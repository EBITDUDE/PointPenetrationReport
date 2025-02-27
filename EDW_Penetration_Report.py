import os
import pandas as pd
import numpy as np
from datetime import datetime

RAW_DATA_PATH = "C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~EDW_data/"
SAVE_PATH = "C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~outputs/" + datetime.now().strftime("%Y.%m.%d_%I%M%p") + '/'

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)

def weighted_avg_date(group):
    try:
        timestamps = pd.to_datetime(group['CABINET_SERVICEABLE_DT'], errors='coerce').dropna().astype(np.int64) // 10**9
        avg_timestamp = timestamps.mean()
        result = pd.to_datetime(avg_timestamp, unit='s', errors='raise')
        return result
    except OverflowError as e:
        print(f"OverflowError for {group['VETRO_CABINET_ID'].iloc[0]}: {e}")
        return pd.NaT
    except Exception as e:
        print(f"General Error for {group['VETRO_CABINET_ID'].iloc[0]}: {e}")
        return pd.NaT

def get_most_common_market(group):
    return group['MARKET_SUMMARY_NAME'].mode().iloc[0]

def replace_9999_with_2100(df, column_name):
    def replace_year(date_str):
        if isinstance(date_str, str) and '9999' in date_str:
            parts = date_str.split('-')
            if len(parts) == 3 and parts[0] == '9999':
                return '2100-' + '-'.join(parts[1:])  # Reconstruct with 2100
        return date_str

    df[column_name] = df[column_name].apply(replace_year)
    return df

def main():
    os.chdir('C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~EDW_data/')
    files = os.listdir()

    # Read raw data files
    raw_occupancy_data = pd.read_csv(RAW_DATA_PATH + '/GTCR_ServiceLocsAndSubTrend_20250221B.csv')

    # Create subsets of raw data
    subset_raw_occupancy_data = raw_occupancy_data.iloc[:-2, :][['LOCATION_ID', 'CABINET_SERVICEABLE_DT',
                                            'IS_SERVICEABLE_FLAG', 'MARKET_SUMMARY_NAME', 
                                            'COMPETITIVE_TYPE_NAME', 'CABINET_KEY', 'VETRO_CABINET_ID'] 
                                            + list(raw_occupancy_data.columns[34:])]
    subset_raw_occupancy_data = subset_raw_occupancy_data.drop_duplicates()

    # 1. Drop non-serviceable locations
    subset_raw_occupancy_data = subset_raw_occupancy_data[subset_raw_occupancy_data['IS_SERVICEABLE_FLAG'] == 1]

    # 2. Handle 9999 dates and convert to datetime
    subset_raw_occupancy_data = replace_9999_with_2100(subset_raw_occupancy_data, 'CABINET_SERVICEABLE_DT')
    subset_raw_occupancy_data['CABINET_SERVICEABLE_DT'] = pd.to_datetime(subset_raw_occupancy_data['CABINET_SERVICEABLE_DT'], errors='coerce')

    # Calculate weighted average serviceable date
    weighted_dates = subset_raw_occupancy_data.groupby('VETRO_CABINET_ID').apply(weighted_avg_date)
    subset_raw_occupancy_data['WEIGHTED_AVG_SERVICEABLE_DT'] = subset_raw_occupancy_data['VETRO_CABINET_ID'].map(weighted_dates)

    # 3. Merge occupancy data
    subcount_cols = [col for col in subset_raw_occupancy_data.columns if col.startswith('SUBCOUNT_')]
    merged_data = subset_raw_occupancy_data.copy()

    # 4. Create conformed market summary name (for non-blank VETRO_CABINET_IDs)
    non_blank_vetro = merged_data.dropna(subset=['VETRO_CABINET_ID'])
    market_mapping = non_blank_vetro.groupby('VETRO_CABINET_ID').apply(get_most_common_market)
    merged_data['CONFORMED_MARKET_SUMMARY_NAME'] = merged_data['VETRO_CABINET_ID'].map(market_mapping)

    # For blank VETRO_CABINET_IDs, retain the original MARKET_SUMMARY_NAME
    merged_data.loc[merged_data['VETRO_CABINET_ID'].isnull(), 'CONFORMED_MARKET_SUMMARY_NAME'] = merged_data.loc[merged_data['VETRO_CABINET_ID'].isnull(), 'MARKET_SUMMARY_NAME']

    # Add the count of VETRO_CABINET_ID instances
    counts = merged_data['VETRO_CABINET_ID'].value_counts()
    merged_data['VETRO_CABINET_ID_COUNT'] = merged_data['VETRO_CABINET_ID'].map(counts)


    # 5. Create subs_by_cabinet dataframe
    group_cols = ['VETRO_CABINET_ID', 'CONFORMED_MARKET_SUMMARY_NAME', 'WEIGHTED_AVG_SERVICEABLE_DT', 'VETRO_CABINET_ID_COUNT','COMPETITIVE_TYPE_NAME']
    duplicates = merged_data[group_cols].groupby('VETRO_CABINET_ID').agg({
        'CONFORMED_MARKET_SUMMARY_NAME': 'nunique',
        'WEIGHTED_AVG_SERVICEABLE_DT': 'nunique',
        'VETRO_CABINET_ID_COUNT':'nunique',
        'COMPETITIVE_TYPE_NAME': 'nunique'
    })
    
    duplicate_cabinets = duplicates[
        (duplicates['CONFORMED_MARKET_SUMMARY_NAME'] > 1) |
        (duplicates['WEIGHTED_AVG_SERVICEABLE_DT'] > 1) |
        (duplicates['COMPETITIVE_TYPE_NAME'] > 1) |
        (duplicates['VETRO_CABINET_ID_COUNT'] > 1)
    ].index.tolist()

    if duplicate_cabinets:
        print(f"Warning: The following VETRO_CABINET_IDs have multiple values for grouping columns: {duplicate_cabinets}")

    blank_vetro = merged_data[merged_data['VETRO_CABINET_ID'].isnull()]
    non_blank_vetro = merged_data.dropna(subset=['VETRO_CABINET_ID'])

    subs_by_cabinet_non_blank = non_blank_vetro.groupby('VETRO_CABINET_ID').agg({
        'CONFORMED_MARKET_SUMMARY_NAME': 'first',
        'WEIGHTED_AVG_SERVICEABLE_DT': 'first',
        'VETRO_CABINET_ID_COUNT': 'first',
        'COMPETITIVE_TYPE_NAME': 'first',
        **{col: 'sum' for col in subcount_cols}
    }).reset_index()

    if not blank_vetro.empty:
        # Calculate counts before grouping
        blank_vetro_counts = blank_vetro.groupby('CONFORMED_MARKET_SUMMARY_NAME').size().reset_index(name='VETRO_CABINET_ID_COUNT')

        subs_by_market_blank = blank_vetro.groupby('CONFORMED_MARKET_SUMMARY_NAME').agg({
            'VETRO_CABINET_ID': lambda x: None,
            'WEIGHTED_AVG_SERVICEABLE_DT': lambda x: pd.NaT,
            'COMPETITIVE_TYPE_NAME': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            **{col: 'sum' for col in subcount_cols}
        }).reset_index()

        # Merge counts back into the grouped data
        subs_by_market_blank = pd.merge(subs_by_market_blank, blank_vetro_counts, on='CONFORMED_MARKET_SUMMARY_NAME')

        subs_by_cabinet = pd.concat([subs_by_cabinet_non_blank, subs_by_market_blank], ignore_index=True)
    else:
        subs_by_cabinet = subs_by_cabinet_non_blank

    # Create output directory if it doesn't exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Format WEIGHTED_AVG_SERVICEABLE_DT and sort
    subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'] = pd.to_datetime(subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'])
    subs_by_cabinet = subs_by_cabinet.sort_values(by=['WEIGHTED_AVG_SERVICEABLE_DT', 'VETRO_CABINET_ID'])
    subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'] = subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'].dt.strftime('%m/%d/%Y')

    # Reorder columns
    desired_order = ['CONFORMED_MARKET_SUMMARY_NAME', 'WEIGHTED_AVG_SERVICEABLE_DT', 'COMPETITIVE_TYPE_NAME', 'VETRO_CABINET_ID', 'VETRO_CABINET_ID_COUNT'] + [col for col in subs_by_cabinet.columns if col not in ['CONFORMED_MARKET_SUMMARY_NAME', 'WEIGHTED_AVG_SERVICEABLE_DT', 'COMPETITIVE_TYPE_NAME', 'VETRO_CABINET_ID', 'VETRO_CABINET_ID_COUNT']]
    subs_by_cabinet = subs_by_cabinet[desired_order]
    
    # Save processed data
    subs_by_cabinet.to_csv(SAVE_PATH + 'subs_by_cabinet.csv', index=False)
    print(f"\nProcessed data saved to: {SAVE_PATH}")
    
    return subs_by_cabinet, merged_data

if __name__ == "__main__":
    subs_by_cabinet, merged_data = main()
    print("\nSample of subs_by_cabinet dataframe:")
    print(subs_by_cabinet.head())