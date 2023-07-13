import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
import time
import glob
from pathlib import Path

RAW_DATA_PATH = "C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~raw_data/~from_Box_ASB_rename/"
MAPPINGS_PATH = "C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~raw_data/~mappings"
SAVE_PATH = "C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~outputs/" + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '/'

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)


def main():
    startTime = time.time()
    os.chdir('C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~raw_data/~from_Box_ASB_rename')
    # run_rename()
    files = os.listdir()

    dtypes = {'ID': 'str',
              'Project_Name': 'str',
              'Cabinet': 'str',
              'MarketType2': 'str',
              'Wirecenter_Region1': 'str',
              'Full_Address1': 'str',
              'Serviceability2': 'str',
              'Serviceable_Date1': 'str',
              'CreatedOn1': 'str',
              'ServiceLocationCreatedBy1': 'str',
              'FundType1': 'str',
              'FundTypeID1': 'str',
              'Omnia_SrvItemLocationID': 'str',
              'AccountLocationStatus1': 'str',
              'Account_Service_Activation_Date1': 'str',
              'Account_Service_Deactivation_Date1': 'str',
              'AccountType': 'str',
              'AccountGroup': 'str',
              'AccountCode1': 'str',
              'AccountName1': 'str',
              'BillingYearYYYY': 'str',
              'BillingMonthMMM': 'str',
              'ChargeAmount': 'str',
              'PromotionAmount': 'str',
              'DiscountAmount': 'str',
              'Net': 'str',
              'ServiceAddress1': 'str',
              'ServiceAddress2': 'str',
              'City': 'str',
              'State_Abbreviation': 'str',
              'Postal_Code': 'str'}
    parse_dates = ['CreatedOn1', 'Serviceable_Date1', 'ServiceLocationCreatedBy1', 'Account_Service_Activation_Date1',
                   'Account_Service_Deactivation_Date1']

    full_report = pd.DataFrame(columns=['Project_Name', 'Market', 'Serviceable Date', 'CRM Addresses',
                                        'Competitive Addresses', 'Underserved Addresses', 'Hybrid Addresses', 'Total Serviceable Addresses',
                                        'Competitive Customers', 'Underserved Customers', 'Hybrid Customers', 'Total Active Customers'])

    mappings = pd.read_csv(max(glob.glob(MAPPINGS_PATH + '/*'), key=os.path.getctime), parse_dates=['Mapped Serv Date'])
    mapping_addition_counter = 0

    for file in files:
        # read in raw data
        omnia_data_raw = pd.read_csv(RAW_DATA_PATH + file, dtype=dtypes, parse_dates=parse_dates)

        # kill columns we don't use
        omnia_data = omnia_data_raw.drop(['ID', 'ServiceLocationCreatedBy1', 'FundType1', 'FundTypeID1', 'AccountGroup',
                                          'BillingYearYYYY', 'BillingMonthMMM', 'ChargeAmount', 'PromotionAmount',
                                          'Net', 'DiscountAmount', 'ServiceAddress1', 'ServiceAddress2', 'City',
                                          'State_Abbreviation', 'Postal_Code'], axis=1)

        # convert columns that should be numeric from string
        to_numeric_cols = ['Omnia_SrvItemLocationID', 'AccountCode1']
        omnia_data[to_numeric_cols] = omnia_data[to_numeric_cols].apply(pd.to_numeric, errors='coerce')

        # derive market identifier from wirecenter
        omnia_data['Wirecenter_Region1'] = omnia_data['Wirecenter_Region1'].astype(str)
        omnia_data['Market'] = omnia_data['Wirecenter_Region1'].apply(define_market)

        # Convert serviceability tag from Yes / No to numeric binary
        omnia_data.Serviceability2.replace(('Yes', 'No'), (1, 0), inplace=True)

        # Determine whether customer has already deactivated per deactivation date tag vs today
        omnia_data['Deactivated'] = omnia_data['Account_Service_Deactivation_Date1'] < datetime.now()

        # check if any projects aren't in mappings file, add to mappings file
        if len(list(set(omnia_data["Project_Name"]).difference(mappings["Project_Name"]))) > 0:
            mappings_to_add = omnia_data.loc[omnia_data["Project_Name"].isin(list(set(omnia_data["Project_Name"]).difference(mappings["Project_Name"])))][['Project_Name', 'Wirecenter_Region1', 'Serviceable_Date1', 'MarketType2']].drop_duplicates()
            mappings_to_add['Mapped Market'] = mappings_to_add['Wirecenter_Region1'].apply(define_market)
            mappings_to_add.drop('Wirecenter_Region1', axis=1, inplace=True)
            mappings_to_add = consolidate_market_type(mappings_to_add)
            mappings_to_add = mappings_to_add.groupby(['Project_Name', 'Mapped Market', 'MarketType2'])["Serviceable_Date1"].min().sort_values(ascending=False).reset_index(name="Serviceable Date")
            mappings_to_add = mappings_to_add[~(mappings_to_add["Serviceable Date"].isnull())]
            mappings_to_add['Mapped Projects'] = mappings_to_add['Project_Name']
            mappings_to_add['MarketType2'].replace({'Urban': 'Competitive', 'Rural': 'Underserved'}, inplace=True)
            mappings_to_add.rename(columns={"Project_Name": "Project_Name",
                                            "Mapped Projects": "Mapped Projects",
                                            "Mapped Market": "Mapped Market",
                                            "Serviceable Date": "Mapped Serv Date",
                                            "MarketType2": "Mapped Market Type"}, inplace=True)
            mappings_to_add = mappings_to_add.reindex(columns=mappings.columns)
            if mappings_to_add.empty:
                pass
            else:
                mappings = pd.concat([mappings, mappings_to_add], ignore_index=False)
                mapping_addition_counter += 1
                print(mappings_to_add["Project_Name"])

        omnia_data = omnia_data.merge(mappings[['Project_Name', 'Mapped Projects', 'Mapped Market', 'Mapped Market Type', 'Mapped Serv Date']], how='left')

        report_df = omnia_data.groupby(['Mapped Projects', 'Mapped Market', 'Mapped Market Type', 'Mapped Serv Date'])["Full_Address1"].size().reset_index(name="CRM Addresses")

        # if CRM addresses changes by more than a threshold, duplicate the row and rename the project to "-A" or something
        # need to figure out where to add this logic and if the remaining joins will still work
        # feel like they should because it'll be a valid entry across all identifiers

        addresses_by_type = omnia_data.groupby(['Mapped Projects', 'Mapped Market', 'Serviceability2', 'Mapped Market Type']).size().reset_index()
        customers_by_type = omnia_data.groupby(['Mapped Projects', 'Mapped Market', 'Serviceability2', 'Mapped Market Type', 'Deactivated'])['Account_Service_Activation_Date1'].count().reset_index()

        competitive_addresses = addresses_by_type.loc[((addresses_by_type.Serviceability2 == 1) & (addresses_by_type['Mapped Market Type'] == 'Competitive'))]
        competitive_addresses = competitive_addresses.drop(['Serviceability2', 'Mapped Market Type'], axis=1)
        competitive_addresses.rename(columns={"Mapped Projects": "Mapped Projects",
                                              "Mapped Market": "Mapped Market",
                                              0: "Competitive Addresses"}, inplace=True)

        underserved_addresses = addresses_by_type.loc[((addresses_by_type.Serviceability2 == 1) & (addresses_by_type['Mapped Market Type'] == 'Underserved'))]
        underserved_addresses = underserved_addresses.drop(['Serviceability2', 'Mapped Market Type'], axis=1)
        underserved_addresses.rename(columns={"Mapped Projects": "Mapped Projects",
                                              "Mapped Market": "Mapped Market",
                                              0: "Underserved Addresses"}, inplace=True)

        hybrid_addresses = addresses_by_type.loc[((addresses_by_type.Serviceability2 == 1) & (addresses_by_type['Mapped Market Type'] == 'Hybrid'))]
        hybrid_addresses = hybrid_addresses.drop(['Serviceability2', 'Mapped Market Type'], axis=1)
        hybrid_addresses.rename(columns={"Mapped Projects": "Mapped Projects",
                                         "Mapped Market": "Mapped Market",
                                         0: "Hybrid Addresses"}, inplace=True)

        competitive_customers = customers_by_type.loc[((customers_by_type.Serviceability2 == 1) &
                                                       (customers_by_type['Mapped Market Type'] == 'Competitive') &
                                                       (customers_by_type.Deactivated == False))]
        competitive_customers = competitive_customers.drop(['Serviceability2', 'Deactivated', 'Mapped Market Type'], axis=1)
        competitive_customers.rename(columns={"Mapped Projects": "Mapped Projects",
                                              "Mapped Market": "Mapped Market",
                                              "Account_Service_Activation_Date1": "Competitive Customers"}, inplace=True)

        underserved_customers = customers_by_type.loc[((customers_by_type.Serviceability2 == 1) &
                                                       (customers_by_type['Mapped Market Type'] == 'Underserved') &
                                                       (customers_by_type.Deactivated == False))]
        underserved_customers = underserved_customers.drop(['Serviceability2', 'Deactivated', 'Mapped Market Type'], axis=1)
        underserved_customers.rename(columns={"Mapped Projects": "Mapped Projects",
                                              "Mapped Market": "Mapped Market",
                                              "Account_Service_Activation_Date1": "Underserved Customers"}, inplace=True)

        hybrid_customers = customers_by_type.loc[((customers_by_type.Serviceability2 == 1) &
                                                  (customers_by_type['Mapped Market Type'] == 'Hybrid') &
                                                  (customers_by_type.Deactivated == False))]
        hybrid_customers = hybrid_customers.drop(['Serviceability2', 'Deactivated', 'Mapped Market Type'], axis=1)
        hybrid_customers.rename(columns={"Mapped Projects": "Mapped Projects",
                                         "Mapped Market": "Mapped Market",
                                         "Account_Service_Activation_Date1": "Hybrid Customers"}, inplace=True)

        report_df = report_df.merge(competitive_addresses, how='left')
        report_df = report_df.merge(underserved_addresses, how='left')
        report_df = report_df.merge(hybrid_addresses, how='left')
        report_df = report_df.merge(competitive_customers, how='left')
        report_df = report_df.merge(underserved_customers, how='left')
        report_df = report_df.merge(hybrid_customers, how='left')
        report_df = report_df.fillna(0)
        report_df.insert(8, 'Total Serviceable Addresses', report_df['Competitive Addresses'] + report_df['Underserved Addresses'] + report_df['Hybrid Addresses'])
        report_df.insert(len(report_df.columns), 'Total Active Customers', report_df['Competitive Customers'] + report_df['Underserved Customers'] + report_df['Hybrid Customers'])
        report_df['Source File'] = datetime.strptime(file[:10], '%Y-%m-%d').date()
        print(file)
        full_report = pd.concat([full_report, report_df], ignore_index=False)

    revert_point = full_report.copy()
    full_report = full_report.sort_values(by=['Mapped Projects', 'Source File'])
    full_report = full_report.groupby(['Mapped Projects', 'Mapped Market', 'Mapped Market Type', 'Mapped Serv Date', 'Source File']).sum().reset_index()

    cleaned_full_report = pd.DataFrame(columns=full_report.columns)

    names = full_report["Mapped Projects"].unique()

    for name in names:
        named_subset = full_report[full_report["Mapped Projects"] == name].sort_values(by="Source File")
        cleaned_subset = clean_data(named_subset)
        cleaned_full_report = pd.concat([cleaned_full_report, cleaned_subset], ignore_index=False)

    # add penetration calculations
    full_report['Competitive Penetration'] = full_report['Competitive Customers'] / full_report['Competitive Addresses']
    full_report['Underserved Penetration'] = full_report['Underserved Customers'] / full_report['Underserved Addresses']
    full_report['Hybrid Penetration'] = full_report['Hybrid Customers'] / full_report['Hybrid Addresses']
    full_report['Total Penetration'] = full_report['Total Active Customers'] / full_report['Total Serviceable Addresses']
    cleaned_full_report['Competitive Penetration'] = cleaned_full_report['Competitive Customers'] / cleaned_full_report['Competitive Addresses']
    cleaned_full_report['Underserved Penetration'] = cleaned_full_report['Underserved Customers'] / cleaned_full_report['Underserved Addresses']
    cleaned_full_report['Hybrid Penetration'] = cleaned_full_report['Hybrid Customers'] / cleaned_full_report['Hybrid Addresses']
    cleaned_full_report['Total Penetration'] = cleaned_full_report['Total Active Customers'] / cleaned_full_report['Total Serviceable Addresses']
    full_report = full_report.fillna(0)
    cleaned_full_report = cleaned_full_report.fillna(0)

    # save outputs to excel
    create_directory()
    full_report.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_full_report_output.xlsx', index=False)
    cleaned_full_report.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_cleaned_full_report_output.xlsx', index=False)
    if mapping_addition_counter > 0:
        mappings.to_csv(MAPPINGS_PATH + '/' + datetime.now().strftime("%Y.%m.%d") + '_project_mappings.csv', index=False)
    if len(full_report) > len(cleaned_full_report):
        dropped_projects = full_report[~full_report.isin(cleaned_full_report)].dropna()
        dropped_projects.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_dropped_projects.xlsx', index=False)
    print('Complete - {0:0.1f} seconds'.format(time.time() - startTime))


def define_market(wcregion):
    if wcregion == 'Inactive':
        return 'Inactive'
    elif wcregion == '':
        return 'Unknown'
    elif wcregion[:3] in ['Virginia', 'BRI', 'CPC', 'DUF']:
        return 'VATN'
    elif wcregion[:3] in ['BLD', 'ISL', 'NFL']:
        return 'Island'
    elif wcregion[:3] in ['SWG', 'TAL', 'ALA', 'OPL']:
        return 'ALAGA'
    elif wcregion[:3] == 'HAG':
        return 'HAG'
    elif wcregion[:3] == 'MCH':
        return 'MICH'
    elif wcregion[:3] == 'OHI':
        return 'OHIO'
    elif wcregion[:3] == 'NGA':
        return 'NWGA'
    elif wcregion[:3] == 'NYK':
        return 'NY'
    elif wcregion[:3] == 'WMI':
        return 'WMICH'
    elif wcregion[:3] == 'NGN':
        return 'NGN'
    else:
        return 'Unknown'


def get_clean_serviceable_date(serviceability, serv_date, active_date, clean_created_date):
    if serviceability == 0 and pd.isnull(active_date):
        return pd.NaT
        # return np.datetime64('NaT')
    elif serviceability == 0 and not pd.isnull(active_date):
        return clean_created_date
    elif pd.isnull(serv_date):
        return clean_created_date
    elif active_date < serv_date:
        return clean_created_date
    else:
        return serv_date


def diff_month(d2):
    d1 = datetime.now()
    return min(36, (d1.year - d2.year) * 12 + d1.month - d2.month)


def run_rename():
    os.chdir('C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~raw_data/~from_Box_ASB_rename')
    for filename in os.listdir("."):
        os.rename(filename,
                  str(datetime.fromtimestamp(os.path.getctime(filename)).strftime('%Y-%m-%d') + '.csv'))


def clean_data(df):
    df = df.copy()
    if (df["Mapped Serv Date"] == "1/0/00").all(axis=0):
        print("0 Serv Date: " + df["Mapped Projects"].iloc[0])
        return
    else:
        df["Mapped Serv Date"] = pd.to_datetime(df["Mapped Serv Date"]).apply(datetime.date)
        df = df[~(df["Source File"] < df["Mapped Serv Date"])]

    if all(i <= 24 for i in df["Total Serviceable Addresses"]):
        print("All <=24: " + df["Mapped Projects"].iloc[0])
        return

    for idx, value in enumerate(df["Total Serviceable Addresses"]):
        if value > 25:
            df = df[idx:]
            break

    if df.iloc[-1]["Total Serviceable Addresses"] <= 10:
        print("Last <=10:" + df["Mapped Projects"].iloc[0])
        pass
    elif df.iloc[-1]["Mapped Market"] == "NY":
        df["Mapped Serv Date"] = pd.to_datetime(df["Mapped Serv Date"])
        return df
    elif df.iloc[-1]["Underserved Addresses"] / df.iloc[-1]["Total Serviceable Addresses"] < 0.15:
        df["Competitive Addresses"] = df["Competitive Addresses"] + df["Underserved Addresses"]
        df["Competitive Customers"] = df["Competitive Customers"] + df["Underserved Customers"]
        df["Underserved Addresses"] = np.nan
        df["Underserved Customers"] = np.nan
        df["Mapped Serv Date"] = pd.to_datetime(df["Mapped Serv Date"])
        return df
    elif df.iloc[-1]["Competitive Addresses"] / df.iloc[-1]["Total Serviceable Addresses"] < 0.15:
        df["Underserved Addresses"] = df["Competitive Addresses"] + df["Underserved Addresses"]
        df["Underserved Customers"] = df["Competitive Customers"] + df["Underserved Customers"]
        df["Competitive Addresses"] = np.nan
        df["Competitive Customers"] = np.nan
        df["Mapped Serv Date"] = pd.to_datetime(df["Mapped Serv Date"])
        return df


def consolidate_market_type(df):
    # Calculate the count of each Project_Name and MarketType2 pair
    counts = df.groupby(["Project_Name", "MarketType2"]).size().reset_index(name="Count")

    # Find the pairs with the maximum count for each project
    max_counts = counts.groupby("Project_Name")["Count"].idxmax()
    max_pairs = counts.loc[max_counts, ["Project_Name", "MarketType2"]]

    # Merge the maximum pairs back to the original DataFrame
    merged_df = df.merge(max_pairs, on="Project_Name", how="left")

    # Replace the MarketType2 column with the consolidated label
    merged_df["MarketType2"] = merged_df["MarketType2_y"].fillna(merged_df["MarketType2_x"])

    # Drop unnecessary columns
    merged_df = merged_df.drop(["MarketType2_x", "MarketType2_y"], axis=1)

    return merged_df.drop_duplicates()


# creates folder for saving outputs
def create_directory():
    if not os.path.isdir(Path(SAVE_PATH)):
        Path(SAVE_PATH).mkdir(exist_ok=False)
        print("Created new output folder at " + SAVE_PATH)
    else:
        print("No folder created. Folder already exists at " + SAVE_PATH)


if __name__ == '__main__':
    main()
