import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time
from dateutil.relativedelta import relativedelta
import seaborn as sns

RAW_DATA_PATH = "C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~raw_data/~from_Box_ASB_rename/"
MAPPINGS_PATH = "C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~raw_data/"
SAVE_PATH = "C:/Users/alexander.bennett/PycharmProjects/Point Penetration Report/~outputs/"

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)


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

    mappings = pd.read_csv(MAPPINGS_PATH + 'project_mappings_v4.csv', parse_dates=['Mapped Serv Date'])
    market_types = mappings[['Project_Name', 'Mapped Market Type']]

    for file in files:
        omnia_data_raw = pd.read_csv(RAW_DATA_PATH + file,
                                     dtype=dtypes,
                                     parse_dates=parse_dates)

        omnia_data = omnia_data_raw.drop(['ServiceLocationCreatedBy1', 'FundType1', 'FundTypeID1', 'AccountGroup',
                                          'BillingYearYYYY', 'BillingMonthMMM', 'ChargeAmount', 'PromotionAmount',
                                          'DiscountAmount', 'ServiceAddress1', 'ServiceAddress2'], axis=1)

        to_numeric_cols = ['Omnia_SrvItemLocationID', 'AccountCode1', 'Net', 'Postal_Code']
        omnia_data[to_numeric_cols] = omnia_data[to_numeric_cols].apply(pd.to_numeric, errors='coerce')

        omnia_data['Wirecenter_Region1'] = omnia_data['Wirecenter_Region1'].astype(str)
        omnia_data['Market'] = omnia_data['Wirecenter_Region1'].apply(define_market)
        omnia_data['CleanCreatedOn'] = omnia_data[['CreatedOn1', 'Account_Service_Activation_Date1']].min(axis=1)
        omnia_data['CleanServiceableDate'] = omnia_data.apply(lambda x: get_clean_serviceable_date(x.Serviceability2,
                                                                                                   x.Serviceable_Date1,
                                                                                                   x.Account_Service_Activation_Date1,
                                                                                                   x.CleanCreatedOn), axis=1)
        omnia_data.Serviceability2.replace(('Yes', 'No'), (1, 0), inplace=True)
        omnia_data['Deactivated'] = omnia_data['Account_Service_Deactivation_Date1'] < datetime.now()

        # should check if any projects aren't in mappings file

        omnia_data = omnia_data.merge(market_types, how='left')

        report_df = omnia_data.groupby(['Project_Name', 'Market', 'Mapped Market Type'])["CleanServiceableDate"].min().sort_values(ascending=False).reset_index(name="Serviceable Date")
        CRM_addresses = omnia_data.groupby(['Project_Name', 'Market'])["Full_Address1"].size().reset_index(name="CRM Addresses")

        addresses_by_type = omnia_data.groupby(['Project_Name', 'Market', 'Serviceability2', 'Mapped Market Type']).size().reset_index()
        competitive_addresses = addresses_by_type.loc[((addresses_by_type.Serviceability2 == 1) & (addresses_by_type['Mapped Market Type'] == 'Competitive'))]
        competitive_addresses = competitive_addresses.drop(['Serviceability2', 'Mapped Market Type'], axis=1)
        competitive_addresses.rename(columns={"Project_Name": "Project_Name",
                                              "Market": "Market",
                                              0: "Competitive Addresses"},
                                     inplace=True)

        underserved_addresses = addresses_by_type.loc[((addresses_by_type.Serviceability2 == 1) & (addresses_by_type['Mapped Market Type'] == 'Underserved'))]
        underserved_addresses = underserved_addresses.drop(['Serviceability2', 'Mapped Market Type'], axis=1)
        underserved_addresses.rename(columns={"Project_Name": "Project_Name",
                                              "Market": "Market",
                                              0: "Underserved Addresses"},
                                     inplace=True)

        hybrid_addresses = addresses_by_type.loc[((addresses_by_type.Serviceability2 == 1) & (addresses_by_type['Mapped Market Type'] == 'Hybrid'))]
        hybrid_addresses = hybrid_addresses.drop(['Serviceability2', 'Mapped Market Type'], axis=1)
        hybrid_addresses.rename(columns={"Project_Name": "Project_Name",
                                         "Market": "Market",
                                         0: "Hybrid Addresses"},
                                inplace=True)

        customers_by_type = omnia_data.groupby(['Project_Name', 'Market', 'Serviceability2', 'Mapped Market Type',
                                                'Deactivated'])['Account_Service_Activation_Date1'].count().reset_index()
        competitive_customers = customers_by_type.loc[((customers_by_type.Serviceability2 == 1) &
                                                       (customers_by_type['Mapped Market Type'] == 'Competitive') &
                                                       (customers_by_type.Deactivated == False))]
        competitive_customers = competitive_customers.drop(['Serviceability2', 'Deactivated', 'Mapped Market Type'], axis=1)
        competitive_customers.rename(columns={"Project_Name": "Project_Name",
                                              "Market": "Market",
                                              'Account_Service_Activation_Date1': "Competitive Customers"},
                                     inplace=True)

        underserved_customers = customers_by_type.loc[((customers_by_type.Serviceability2 == 1) &
                                                       (customers_by_type['Mapped Market Type'] == 'Underserved') &
                                                       (customers_by_type.Deactivated == False))]
        underserved_customers = underserved_customers.drop(['Serviceability2', 'Deactivated', 'Mapped Market Type'], axis=1)
        underserved_customers.rename(columns={"Project_Name": "Project_Name",
                                              "Market": "Market",
                                              'Account_Service_Activation_Date1': "Underserved Customers"},
                                     inplace=True)

        hybrid_customers = customers_by_type.loc[((customers_by_type.Serviceability2 == 1) &
                                                       (customers_by_type['Mapped Market Type'] == 'Hybrid') &
                                                       (customers_by_type.Deactivated == False))]
        hybrid_customers = hybrid_customers.drop(['Serviceability2', 'Deactivated', 'Mapped Market Type'], axis=1)
        hybrid_customers.rename(columns={"Project_Name": "Project_Name",
                                         "Market": "Market",
                                         'Account_Service_Activation_Date1': "Hybrid Customers"},
                                inplace=True)

        report_df = report_df.merge(CRM_addresses)
        report_df = report_df.merge(competitive_addresses, how='left')
        report_df = report_df.merge(underserved_addresses, how='left')
        report_df = report_df.merge(hybrid_addresses, how='left')
        report_df = report_df.merge(competitive_customers, how='left')
        report_df = report_df.merge(underserved_customers, how='left')
        report_df = report_df.merge(hybrid_customers, how='left')
        report_df = report_df.fillna(0)
        report_df.insert(8, 'Total Serviceable Addresses',
                         report_df['Competitive Addresses'] + report_df['Underserved Addresses'] + report_df['Hybrid Addresses'])
        report_df.insert(len(report_df.columns), 'Total Active Customers',
                         report_df['Competitive Customers'] + report_df['Underserved Customers'] + report_df['Hybrid Customers'])
        report_df['Source File'] = datetime.strptime(file[:10], '%Y-%m-%d').date()
        print(file)
        full_report = pd.concat([full_report, report_df], ignore_index=False)
        # od_to_output = pd.concat([od_to_output, omnia_data.loc[:, ['Project_Name', 'Market', 'MarketType2']]], ignore_index=False)

    full_report = full_report.sort_values(by=['Project_Name', 'Source File'])
    full_report = full_report.merge(mappings, how='left')
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

    # save outputs to excel
    full_report.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_full_report_output.xlsx', index=False)
    cleaned_full_report.to_excel(SAVE_PATH + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '_cleaned_full_report_output.xlsx', index=False)
    print('Complete - {0:0.1f} seconds'.format(time.time() - startTime))

    # have dataframe with all days between first date and today
    # merge onto this df so you can hae smooth monthly points
    # fill gaps with most recent data point
    # or could just look for date one month out and if it's not there just take most recent one


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
        return
    else:
        df["Mapped Serv Date"] = pd.to_datetime(df["Mapped Serv Date"]).apply(datetime.date)
        df = df[~(df["Source File"] < df["Mapped Serv Date"])]

    if all(i <= 24 for i in df["Total Serviceable Addresses"]):
        return

    for idx, value in enumerate(df["Total Serviceable Addresses"]):
        if value > 25:
            df = df[idx:]
            break

    # should count if there are any rows with all zero and then either print the name or remove
    # should return names of projects getting cut - maybe call new function and append project name to list if getting killed in any of these

    if df.iloc[-1]["Total Serviceable Addresses"] <= 10:
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






if __name__ == '__main__':
    main()
