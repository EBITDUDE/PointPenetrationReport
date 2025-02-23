import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import time
import glob

RAW_DATA_PATH = "C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~EDW_data/"
SAVE_PATH = "C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~outputs/" + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '/'

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)

def weighted_avg_date(group):
    # Convert to datetime *before* calculating the mean timestamp
    timestamps = pd.to_datetime(group['SERVICEABLE_DT']).astype(np.int64) // 10**9 # Convert to unix timestamps
    avg_timestamp = timestamps.mean()
    return pd.to_datetime(avg_timestamp, unit='s') # Convert back to datetime

def get_most_common_market(group):
    return group['MARKET_SUMMARY_NAME'].mode().iloc[0]

def main():
    os.chdir('C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~EDW_data/')
    files = os.listdir()

    # Read raw data files
    # raw_data = pd.read_csv(max(glob.glob(RAW_DATA_PATH + '/*'), 
    raw_data = pd.read_csv(RAW_DATA_PATH + '/AllLocsSubs_20250217.csv')
    raw_occupancy_data = pd.read_csv(RAW_DATA_PATH + '/GTCR_ServiceLocsAndSubTrend_20250221.csv')

    # Create subsets of raw data
    subset_raw_data = raw_data.loc[:, ['LOCATION_ID', 'SERVICEABLE_DT', 
                                      'IS_SERVICEABLE_FLAG', 'MARKET_SUMMARY_NAME', 
                                      'COMPETITIVE_TYPE_NAME', 'CABINET_KEY', 'VETRO_CABINET_ID']]
    subset_raw_data = subset_raw_data.drop_duplicates()

    subset_raw_occupancy_data = raw_occupancy_data.iloc[:-2, :][['LOCATION_ID', 'SERVICEABLE_DT', 
                                'IS_SERVICEABLE_FLAG', 'MARKET_SUMMARY_NAME', 
                                'COMPETITIVE_TYPE_NAME', 'CABINET_KEY', 'VETRO_CABINET_ID'] 
                                + list(raw_occupancy_data.columns[34:])]
    subset_raw_occupancy_data = subset_raw_occupancy_data.drop_duplicates()

    # 1. Drop non-serviceable locations
    subset_raw_data = subset_raw_data[subset_raw_data['IS_SERVICEABLE_FLAG'] == 1]

    # 2. Handle 9999 dates and convert to datetime
    subset_raw_data['SERVICEABLE_DT'] = subset_raw_data['SERVICEABLE_DT'].str.replace('9999-12-31', '2100-12-31')
    subset_raw_data['SERVICEABLE_DT'] = pd.to_datetime(subset_raw_data['SERVICEABLE_DT'])
    
    # Calculate weighted average serviceable date
    weighted_dates = subset_raw_data.groupby('VETRO_CABINET_ID').apply(weighted_avg_date)
    subset_raw_data['WEIGHTED_AVG_SERVICEABLE_DT'] = subset_raw_data['VETRO_CABINET_ID'].map(weighted_dates)

    # 3. Merge occupancy data
    subcount_cols = [col for col in subset_raw_occupancy_data.columns if col.startswith('SUBCOUNT_')]
    merged_data = subset_raw_data.merge(
        subset_raw_occupancy_data[['LOCATION_ID'] + subcount_cols],
        on='LOCATION_ID',
        how='left'
    )

    # 4. Create conformed market summary name
    market_mapping = merged_data.groupby('VETRO_CABINET_ID').apply(get_most_common_market)
    merged_data['CONFORMED_MARKET_SUMMARY_NAME'] = merged_data['VETRO_CABINET_ID'].map(market_mapping)

    # 5. Create subs_by_cabinet dataframe
    # Check for duplicates first
    group_cols = ['VETRO_CABINET_ID', 'CONFORMED_MARKET_SUMMARY_NAME', 'WEIGHTED_AVG_SERVICEABLE_DT', 'COMPETITIVE_TYPE_NAME']
    duplicates = merged_data[group_cols].groupby('VETRO_CABINET_ID').agg({
        'CONFORMED_MARKET_SUMMARY_NAME': 'nunique',
        'WEIGHTED_AVG_SERVICEABLE_DT': 'nunique',
        'COMPETITIVE_TYPE_NAME': 'nunique'
    })
    
    duplicate_cabinets = duplicates[
        (duplicates['CONFORMED_MARKET_SUMMARY_NAME'] > 1) |
        (duplicates['WEIGHTED_AVG_SERVICEABLE_DT'] > 1) |
        (duplicates['COMPETITIVE_TYPE_NAME'] > 1)
    ].index.tolist()

    if duplicate_cabinets:
        print(f"Warning: The following VETRO_CABINET_IDs have multiple values for grouping columns: {duplicate_cabinets}")

    # Create final subs_by_cabinet dataframe
    subs_by_cabinet = merged_data.groupby('VETRO_CABINET_ID').agg({
        'CONFORMED_MARKET_SUMMARY_NAME': 'first',
        'WEIGHTED_AVG_SERVICEABLE_DT': 'first',
        'COMPETITIVE_TYPE_NAME': 'first',
        **{col: 'sum' for col in subcount_cols}
    }).reset_index()

    # Create output directory if it doesn't exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save processed data
    subs_by_cabinet.to_csv(SAVE_PATH + 'subs_by_cabinet.csv', index=False)
    print(f"\nProcessed data saved to: {SAVE_PATH}")
    
    return subs_by_cabinet, merged_data

if __name__ == "__main__":
    subs_by_cabinet, merged_data = main()
    print("\nSample of subs_by_cabinet dataframe:")
    print(subs_by_cabinet.head())