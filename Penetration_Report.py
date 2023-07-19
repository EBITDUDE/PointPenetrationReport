import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import time
import glob
from pathlib import Path
import dateutil.relativedelta as relativedelta

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
    parse_dates = ['CreatedOn1', 'Serviceable_Date1', 'Account_Service_Activation_Date1', 'Account_Service_Deactivation_Date1']

    full_report = pd.DataFrame(columns=['Project_Name', 'Market', 'Serviceable Date', 'CRM Addresses',
                                        'Competitive Addresses', 'Underserved Addresses', 'Hybrid Addresses', 'Total Serviceable Addresses',
                                        'Competitive Customers', 'Underserved Customers', 'Hybrid Customers', 'Total Active Customers'])

    mappings = pd.read_csv(max(glob.glob(MAPPINGS_PATH + '/*'), key=os.path.getctime), parse_dates=['Mapped Serv Date'])
    mapping_addition_counter = 0

    for file in files:
        # read in raw data
        omnia_data_raw = pd.read_csv(RAW_DATA_PATH + file, dtype=dtypes, parse_dates=parse_dates)

        # kill columns we don't use
        omnia_data = omnia_data_raw.drop(['ID', 'Full_Address1', 'ServiceLocationCreatedBy1', 'FundType1',
                                          'FundTypeID1', 'AccountLocationStatus1', 'AccountGroup',
                                          'BillingYearYYYY', 'BillingMonthMMM', 'ChargeAmount', 'PromotionAmount',
                                          'Net', 'DiscountAmount', 'ServiceAddress1', 'ServiceAddress2', 'City',
                                          'State_Abbreviation', 'Postal_Code'], axis=1, errors='ignore')

        # convert columns that should be numeric from string
        to_numeric_cols = ['Omnia_SrvItemLocationID', 'AccountCode1']
        omnia_data[to_numeric_cols] = omnia_data[to_numeric_cols].apply(pd.to_numeric, errors='coerce')

        # derive market identifier from wirecenter
        omnia_data['Wirecenter_Region1'] = omnia_data['Wirecenter_Region1'].astype(str)
        omnia_data['Market'] = omnia_data['Wirecenter_Region1'].apply(define_market)

        # convert serviceability tag from Yes / No to numeric binary
        omnia_data.Serviceability2.replace(('Yes', 'No'), (1, 0), inplace=True)

        # determine whether customer has already deactivated per deactivation date tag vs date of source file
        omnia_data['Deactivated'] = omnia_data['Account_Service_Deactivation_Date1'] < pd.Timestamp(datetime.strptime(file[:10], '%Y-%m-%d').date())

        omnia_data = omnia_data[omnia_data['Serviceability2'] != 0]

        omnia_data = omnia_data[omnia_data['Deactivated'] == False]

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

        report_df = omnia_data.groupby(['Mapped Projects', 'Mapped Market', 'Mapped Market Type', 'Mapped Serv Date'])["Omnia_SrvItemLocationID"].size().reset_index(name="CRM Addresses")
        report_df["CRM Addresses"] = report_df["CRM Addresses"].astype(float)

        addresses_by_type = omnia_data.groupby(['Mapped Projects', 'Mapped Market', 'Serviceability2', 'Mapped Market Type']).size().reset_index()
        customers_by_type = omnia_data.groupby(['Mapped Projects', 'Mapped Market', 'Serviceability2', 'Mapped Market Type', 'Deactivated'])['Account_Service_Activation_Date1'].count().reset_index()

        # create datasets of competitive, underserved, and hybrid addresses by the tags from the mappings file
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

        # create datasets of competitive, underserved, and hybrid customer counts by the tags from the mappings file
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

        # merge datasets created above onto master report
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
    penetration_cols = ['Competitive Penetration', 'Underserved Penetration', 'Hybrid Penetration', 'Total Penetration']
    customer_cols = ['Competitive Customers', 'Underserved Customers', 'Hybrid Customers', 'Total Active Customers']
    address_cols = ['Competitive Addresses', 'Underserved Addresses', 'Hybrid Addresses', 'Total Serviceable Addresses']

    for i in range(len(penetration_cols)):
        full_report[penetration_cols[i]] = full_report[customer_cols[i]] / full_report[address_cols[i]]
        cleaned_full_report[penetration_cols[i]] = cleaned_full_report[customer_cols[i]] / cleaned_full_report[address_cols[i]]

    full_report = full_report.fillna(0)
    cleaned_full_report = cleaned_full_report.fillna(0)
    full_report['Mapped Serv Date'] = full_report['Mapped Serv Date'].apply(lambda x: x.date())
    cleaned_full_report['Mapped Serv Date'] = cleaned_full_report['Mapped Serv Date'].apply(lambda x: x.date())

    dates = cleaned_full_report["Source File"].sort_values().drop_duplicates()
    age_range = range(1, 37)

    shortened_full_report = pd.DataFrame({
        'Mapped Projects': mappings['Mapped Projects'].repeat(len(age_range)),
        'Mapped Market': mappings['Mapped Market'].repeat(len(age_range)),
        'Mapped Serv Date': mappings['Mapped Serv Date'].repeat(len(age_range)).apply(lambda x: x.date()),
        'Mapped Market Type': mappings['Mapped Market Type'].repeat(len(age_range)),
        'Age': list(age_range) * len(mappings)
    })

    shortened_full_report = shortened_full_report.drop_duplicates()
    shortened_full_report['Source File'] = shortened_full_report.apply(lambda row: find_closest_date(dates, row['Mapped Serv Date'], row['Age']), axis=1)
    shortened_full_report = shortened_full_report.dropna()
    cols = shortened_full_report.columns.tolist()
    cols = cols[:2] + [cols[3]] + [cols[2]] + cols[-2:]
    shortened_full_report = shortened_full_report[cols]
    shortened_full_report = shortened_full_report.merge(cleaned_full_report)
    shortened_full_report = shortened_full_report.sort_values(by=['Mapped Projects', 'Source File'])

    # if last value is less than first value, replace all with lower last value - helps control for address audits
    final_report = pd.DataFrame(columns=shortened_full_report.columns)
    names = shortened_full_report["Mapped Projects"].unique()
    for name in names:
        named_subset = shortened_full_report[shortened_full_report["Mapped Projects"] == name].sort_values(by="Source File")
        named_subset['Competitive Addresses'] = replace_with_last_value(named_subset['Competitive Addresses'])
        named_subset['Underserved Addresses'] = replace_with_last_value(named_subset['Underserved Addresses'])
        named_subset['Hybrid Addresses'] = replace_with_last_value(named_subset['Hybrid Addresses'])
        named_subset['Total Serviceable Addresses'] = replace_with_last_value(named_subset['Total Serviceable Addresses'])
        final_report = pd.concat([final_report, named_subset], ignore_index=False)
        # shouldn't clean if name matches one of the two i found

    # add penetration calculations
    for i in range(len(penetration_cols)):
        final_report[penetration_cols[i]] = final_report[customer_cols[i]] / final_report[address_cols[i]]
    final_report = final_report.fillna(0)

    # create save folder and save outputs to excel
    create_directory()

    full_report.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_full_output.xlsx', index=False)
    cleaned_full_report.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_cleaned_output.xlsx', index=False)
    final_report.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_shortened_cleaned_output.xlsx', index=False)
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


def find_closest_date(date_list, base_date, offset_months):
    if pd.isnull(base_date):
        return None

    prior_target_date = max(base_date, base_date + relativedelta.relativedelta(months=offset_months - 1))
    target_date = base_date + relativedelta.relativedelta(months=offset_months)

    if target_date <= date.today():
        valid_dates = [i for i in date_list if target_date >= i > base_date and i > prior_target_date]
        if valid_dates:
            closest_date = max(valid_dates)
            return closest_date
    else:
        return None


def replace_with_last_value(df):
    df = df.copy()
    last_value = df.iloc[-1]
    first_value = df.iloc[0]

    if last_value < first_value:
        df = df.apply(lambda x: last_value)

    return df


if __name__ == '__main__':
    main()
