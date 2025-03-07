import os
import pandas as pd
import numpy as np
import logging
import concurrent.futures
from datetime import datetime

# =========================================================================
# CONFIGURATION SECTION
# All configurable parameters are defined here for easy modification
# =========================================================================
CONFIG = {
    'RAW_DATA_PATH': "C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~EDW_data/",
    'SAVE_PATH_BASE': "C:/Users/alexander.bennett/PycharmProjects/PointPenetrationReport/~outputs/",
    'RAW_DATA_FILENAME': "GTCR_ServiceLocsAndSubTrend_20250306g.csv",
    'CHUNK_SIZE': 10000,  # For processing large files in chunks
    'USE_PARALLEL': False,  # Set to True to enable parallel processing
    'MAX_WORKERS': 4,  # Number of parallel workers (if parallel processing is enabled)
    'DTYPE_DICT': {
        'FUND_TYPE': str,                  # Column 14
        'NEW_CABINET_NAME': str,           # Column 4
        'CABINET_SERVICEABLE_DT': str,     # Column 5 (date, converted later)
        'COMPETITIVE_TYPE_NAME': str,      # Column 10
        'LATITUDE': str,                   # Column 18
        'PROJECT_CODE': str,               # Column 24
        'PROJECT_NUMBER': str,             # Column 26
        'PROJECT_SERVICEABLE_DT': str      # Column 27 (date, converted later)
        }
}

# Set up the save path with timestamp
SAVE_PATH = os.path.join(CONFIG['SAVE_PATH_BASE'], datetime.now().strftime("%Y.%m.%d_%I%M%p"))

# Configure pandas display settings
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)

# =========================================================================
# LOGGING SETUP
# Configures logging to both console and file
# =========================================================================
def setup_logging():
    """
    Set up logging to both console and file.
    Returns a configured logger object.
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.join(SAVE_PATH, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'processing.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('edw_penetration_report')

# =========================================================================
# DATA VALIDATION FUNCTIONS
# These functions check the input data for validity
# =========================================================================
def validate_input_data(df):
    """
    Validate that the input dataframe has all required columns and correct data types.
    
    Args:
        df (pandas.DataFrame): The input dataframe to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If required columns are missing
    """
    # Define required columns
    required_columns = [
        'LOCATION_ID', 'CABINET_SERVICEABLE_DT', 'IS_SERVICEABLE_FLAG', 
        'MARKET_SUMMARY_NAME', 'COMPETITIVE_TYPE_NAME', 
        'CABINET_KEY', 'VETRO_CABINET_ID'
    ]
    
    # Check for missing columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # At least one SUBCOUNT column should exist
    subcount_cols = [col for col in df.columns if col.startswith('SUBCOUNT_')]
    if not subcount_cols:
        raise ValueError("No SUBCOUNT_* columns found in the data")
    
    # Validate data types
    if not all(df['IS_SERVICEABLE_FLAG'].isin([0, 1])):
        logging.warning("IS_SERVICEABLE_FLAG contains values other than 0 or 1")
    
    return True

# =========================================================================
# DATE HANDLING FUNCTIONS
# Functions for processing and manipulating date fields
# =========================================================================
def replace_9999_with_2100(df, column_name):
    """
    Replace '9999' years with '2100' in date strings.
    Uses vectorized operations for better performance.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the date column
        column_name (str): The name of the column containing dates
        
    Returns:
        pandas.DataFrame: The updated dataframe
    """
    # Only process if the column is string type (object)
    if df[column_name].dtype == 'object':
        # Replace 9999 with 2100 at the beginning of the string
        df[column_name] = df[column_name].astype(str).str.replace(r'^9999-', '2100-', regex=True)
    return df

def weighted_avg_date(group):
    """
    Calculate the weighted average date for a group of records.
    
    Args:
        group (pandas.DataFrame): A group of records
        
    Returns:
        pandas.Timestamp: The weighted average date, or NaT if calculation fails
    """
    # Get cabinet ID for error reporting
    cabinet_id = group['VETRO_CABINET_ID'].iloc[0] if len(group['VETRO_CABINET_ID']) > 0 else "unknown"
    
    # Filter out invalid dates before conversion to improve performance
    valid_dates = group['CABINET_SERVICEABLE_DT'].dropna()
    if valid_dates.empty:
        logging.warning(f"No valid dates for cabinet {cabinet_id}")
        return pd.NaT
        
    try:
        # Convert dates to timestamps (nanoseconds since epoch)
        timestamps = pd.to_datetime(valid_dates, errors='coerce').dropna().astype(np.int64) // 10**9
        if timestamps.empty:
            logging.warning(f"No valid timestamps after conversion for cabinet {cabinet_id}")
            return pd.NaT
            
        # Calculate mean timestamp and convert back to datetime
        avg_timestamp = timestamps.mean()
        return pd.to_datetime(avg_timestamp, unit='s')
    except OverflowError as e:
        logging.error(f"OverflowError for cabinet {cabinet_id}: {e}")
        return pd.NaT
    except Exception as e:
        logging.error(f"General error calculating weighted date for cabinet {cabinet_id}: {e}")
        return pd.NaT

# =========================================================================
# DATA AGGREGATION FUNCTIONS
# Functions for aggregating and summarizing data
# =========================================================================
def get_most_common_market(group):
    """
    Get the most common market name from a group.
    More efficient than using mode() for larger datasets.
    
    Args:
        group (pandas.DataFrame): A group of records
        
    Returns:
        str: The most common market name
    """
    # Count occurrences of each market name
    market_counts = group['MARKET_SUMMARY_NAME'].value_counts()
    if market_counts.empty:
        return None
    # Return the market name with the highest count
    return market_counts.index[0]

def process_cabinet_group(group_data, subcount_cols):
    """
    Process a single cabinet group for parallel processing.
    
    Args:
        group_data (pandas.DataFrame): Data for a single cabinet
        subcount_cols (list): List of subscription count columns
        
    Returns:
        dict: Aggregated data for the cabinet
    """
    # Initialize result with non-numeric columns
    result = {
        'CONFORMED_MARKET_SUMMARY_NAME': group_data['CONFORMED_MARKET_SUMMARY_NAME'].iloc[0],
        'WEIGHTED_AVG_SERVICEABLE_DT': group_data['WEIGHTED_AVG_SERVICEABLE_DT'].iloc[0],
        'VETRO_CABINET_ID_COUNT': group_data['VETRO_CABINET_ID_COUNT'].iloc[0],
        'COMPETITIVE_TYPE_NAME': group_data['COMPETITIVE_TYPE_NAME'].iloc[0],
    }
    
    # Add sum of subscription counts
    for col in subcount_cols:
        result[col] = group_data[col].sum()
        
    return result

def process_blank_vetro(blank_vetro, subcount_cols):
    """
    Process dataframe rows with blank VETRO_CABINET_ID values.
    
    Args:
        blank_vetro (pandas.DataFrame): DataFrame with blank VETRO_CABINET_ID values
        subcount_cols (list): List of subscription count columns
        
    Returns:
        pandas.DataFrame: Processed data for blank VETRO_CABINET_ID records
    """
    if blank_vetro.empty:
        return pd.DataFrame()
        
    # Calculate counts of records per market
    blank_vetro_counts = blank_vetro.groupby('CONFORMED_MARKET_SUMMARY_NAME').size().reset_index(name='VETRO_CABINET_ID_COUNT')
    
    # Group and aggregate data
    subs_by_market_blank = blank_vetro.groupby('CONFORMED_MARKET_SUMMARY_NAME').agg({
        'VETRO_CABINET_ID': lambda x: None,  # Keep VETRO_CABINET_ID as None
        'WEIGHTED_AVG_SERVICEABLE_DT': lambda x: pd.NaT,  # No date for blank records
        'COMPETITIVE_TYPE_NAME': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        **{col: 'sum' for col in subcount_cols}  # Sum subscription counts
    }).reset_index()
    
    # Merge counts back into the aggregated data
    return pd.merge(subs_by_market_blank, blank_vetro_counts, on='CONFORMED_MARKET_SUMMARY_NAME')

# =========================================================================
# LARGE FILE PROCESSING FUNCTIONS
# Functions for handling large data files
# =========================================================================
def process_data_in_chunks(file_path, dtype_dict, chunk_size=10000):
    """
    Process a large CSV file in chunks to manage memory usage.
    
    Args:
        file_path (str): Path to the CSV file
        chunk_size (int): Number of rows to process at once
        
    Returns:
        pandas.DataFrame: The processed dataframe
    """
    logger = logging.getLogger('edw_penetration_report')
    logger.info(f"Processing file {file_path} in chunks of {chunk_size} rows")
    
    chunks = []
    for i, chunk in enumerate(pd.read_csv(file_path, dtype=dtype_dict, chunksize=chunk_size)):
        logger.info(f"Processing chunk {i+1}")

        # Clean LATITUDE column: remove trailing commas and convert to float
        if 'LATITUDE' in chunk.columns:
            chunk['LATITUDE'] = pd.to_numeric(
                chunk['LATITUDE'].astype(str).str.rstrip(','),
                errors='coerce'
            )
        
        # Get subcount columns
        subcount_cols = [c for c in chunk.columns if c.startswith('SUBCOUNT_')]
        
        # Select required columns
        base_columns = [
            'LOCATION_ID', 'CABINET_SERVICEABLE_DT', 'IS_SERVICEABLE_FLAG', 
            'MARKET_SUMMARY_NAME', 'COMPETITIVE_TYPE_NAME', 'CABINET_KEY', 
            'VETRO_CABINET_ID'
        ]
        
        processed_chunk = chunk[base_columns + subcount_cols]
        processed_chunk = processed_chunk[processed_chunk['IS_SERVICEABLE_FLAG'] == 1]
        
        chunks.append(processed_chunk)
    
    # Combine all processed chunks
    logger.info(f"Combining {len(chunks)} processed chunks")
    return pd.concat(chunks, ignore_index=True)
# =========================================================================
# MAIN PROCESSING FUNCTION
# The main function that orchestrates the data processing workflow
# =========================================================================
def main():
    """
    Main function that processes the data and generates the penetration report.
    
    Returns:
        tuple: (subs_by_cabinet DataFrame, merged_data DataFrame)
    """
    # Set up logging
    logger = setup_logging()
    logger.info("Starting EDW Penetration Report generation")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        logger.info(f"Created output directory: {SAVE_PATH}")
    
    # Define the full path to the raw data file
    raw_data_file = os.path.join(CONFIG['RAW_DATA_PATH'], CONFIG['RAW_DATA_FILENAME'])
    logger.info(f"Reading raw data from: {raw_data_file}")
    
    # Read raw data - either all at once or in chunks
    if CONFIG['CHUNK_SIZE'] > 0:
        raw_occupancy_data = process_data_in_chunks(raw_data_file, CONFIG['DTYPE_DICT'], CONFIG['CHUNK_SIZE'])
    else:
        raw_occupancy_data = pd.read_csv(raw_data_file)
    
    # Validate the input data
    try:
        validate_input_data(raw_occupancy_data)
        logger.info("Input data validation passed")
    except ValueError as e:
        logger.error(f"Input data validation failed: {e}")
        return None, None
    
    # Get list of subscription count columns
    subcount_cols = [col for col in raw_occupancy_data.columns if col.startswith('SUBCOUNT_')]
    logger.info(f"Found {len(subcount_cols)} subscription count columns")
    
    # Create subset of raw data (excluding last two rows which might be totals)
    base_columns = [
        'LOCATION_ID', 'CABINET_SERVICEABLE_DT', 'IS_SERVICEABLE_FLAG', 
        'MARKET_SUMMARY_NAME', 'COMPETITIVE_TYPE_NAME', 'CABINET_KEY', 
        'VETRO_CABINET_ID'
    ]
    
    subset_raw_occupancy_data = raw_occupancy_data.iloc[:, :][base_columns + subcount_cols]
    subset_raw_occupancy_data = subset_raw_occupancy_data.drop_duplicates()
    logger.info(f"Created subset with {len(subset_raw_occupancy_data)} rows after removing duplicates")
    
    # 1. Drop non-serviceable locations
    subset_raw_occupancy_data = subset_raw_occupancy_data[subset_raw_occupancy_data['IS_SERVICEABLE_FLAG'] == 1]
    logger.info(f"Filtered to {len(subset_raw_occupancy_data)} serviceable locations")
    
    # 2. Handle 9999 dates and convert to datetime
    subset_raw_occupancy_data = replace_9999_with_2100(subset_raw_occupancy_data, 'CABINET_SERVICEABLE_DT')
    subset_raw_occupancy_data['CABINET_SERVICEABLE_DT'] = pd.to_datetime(
        subset_raw_occupancy_data['CABINET_SERVICEABLE_DT'], 
        errors='coerce'
    )
    logger.info("Processed dates: replaced 9999 with 2100 and converted to datetime")
    
    # Calculate weighted average serviceable date
    logger.info("Calculating weighted average serviceable dates")
    weighted_dates = subset_raw_occupancy_data.groupby('VETRO_CABINET_ID').apply(weighted_avg_date)
    subset_raw_occupancy_data['WEIGHTED_AVG_SERVICEABLE_DT'] = subset_raw_occupancy_data['VETRO_CABINET_ID'].map(weighted_dates)
    
    # 3. Create a copy for further processing
    merged_data = subset_raw_occupancy_data.copy()
    
    # 4. Create conformed market summary name (for non-blank VETRO_CABINET_IDs)
    logger.info("Creating conformed market summary names")
    non_blank_vetro = merged_data.dropna(subset=['VETRO_CABINET_ID'])
    market_mapping = non_blank_vetro.groupby('VETRO_CABINET_ID').apply(get_most_common_market)
    merged_data['CONFORMED_MARKET_SUMMARY_NAME'] = merged_data['VETRO_CABINET_ID'].map(market_mapping)
    
    # For blank VETRO_CABINET_IDs, retain the original MARKET_SUMMARY_NAME
    merged_data.loc[merged_data['VETRO_CABINET_ID'].isnull(), 'CONFORMED_MARKET_SUMMARY_NAME'] = (
        merged_data.loc[merged_data['VETRO_CABINET_ID'].isnull(), 'MARKET_SUMMARY_NAME']
    )
    
    # Add the count of VETRO_CABINET_ID instances
    counts = merged_data['VETRO_CABINET_ID'].value_counts()
    merged_data['VETRO_CABINET_ID_COUNT'] = merged_data['VETRO_CABINET_ID'].map(counts)
    
    # 5. Check for duplicates in grouping columns
    logger.info("Checking for duplicate values in grouping columns")
    
    duplicates = merged_data.groupby('VETRO_CABINET_ID').agg({
        'CONFORMED_MARKET_SUMMARY_NAME': 'nunique',
        'WEIGHTED_AVG_SERVICEABLE_DT': 'nunique',
        'VETRO_CABINET_ID_COUNT': 'nunique',
        'COMPETITIVE_TYPE_NAME': 'nunique'
    })
    
    # More efficient duplicate check
    duplicate_cabinets = duplicates[(duplicates > 1).any(axis=1)].index.tolist()
    duplicate_cabinets_set = set(duplicate_cabinets)
    
    if duplicate_cabinets_set:
        logger.warning(f"{len(duplicate_cabinets_set)} VETRO_CABINET_IDs have multiple values for grouping columns")
        if len(duplicate_cabinets_set) <= 10:  # Only log details for a reasonable number
            logger.warning(f"Duplicate cabinet IDs: {list(duplicate_cabinets_set)}")
    
    # Split data into blank and non-blank VETRO_CABINET_ID
    blank_vetro = merged_data[merged_data['VETRO_CABINET_ID'].isnull()]
    non_blank_vetro = merged_data.dropna(subset=['VETRO_CABINET_ID'])
    
    logger.info(f"Processing {len(non_blank_vetro)} records with VETRO_CABINET_ID")
    logger.info(f"Processing {len(blank_vetro)} records without VETRO_CABINET_ID")
    
    # 6. Create subs_by_cabinet dataframe - with option for parallel processing
    if CONFIG['USE_PARALLEL'] and len(non_blank_vetro) > 1000:
        logger.info(f"Using parallel processing with {CONFIG['MAX_WORKERS']} workers")
        
        # Group data by VETRO_CABINET_ID
        grouped = non_blank_vetro.groupby('VETRO_CABINET_ID')
        groups = [(name, group) for name, group in grouped]
        
        # Process groups in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
            future_to_cabinet = {
                executor.submit(process_cabinet_group, group_data, subcount_cols): name 
                for name, group_data in groups
            }
            
            for future in concurrent.futures.as_completed(future_to_cabinet):
                cabinet_id = future_to_cabinet[future]
                try:
                    result = future.result()
                    results.append((cabinet_id, result))
                except Exception as e:
                    logger.error(f"Error processing cabinet {cabinet_id}: {e}")
        
        # Convert results to DataFrame
        subs_by_cabinet_non_blank = pd.DataFrame([
            {**{'VETRO_CABINET_ID': name}, **data} for name, data in results
        ])
    else:
        # Standard processing (non-parallel)
        logger.info("Using standard (non-parallel) processing")
        subs_by_cabinet_non_blank = non_blank_vetro.groupby('VETRO_CABINET_ID').agg({
            'CONFORMED_MARKET_SUMMARY_NAME': 'first',
            'WEIGHTED_AVG_SERVICEABLE_DT': 'first',
            'VETRO_CABINET_ID_COUNT': 'first',
            'COMPETITIVE_TYPE_NAME': 'first',
            **{col: 'sum' for col in subcount_cols}
        }).reset_index()
    
    # Process blank VETRO_CABINET_ID records
    logger.info("Processing records with blank VETRO_CABINET_ID")
    subs_by_market_blank = process_blank_vetro(blank_vetro, subcount_cols)
    
    # Combine blank and non-blank results
    if not subs_by_market_blank.empty:
        logger.info("Combining blank and non-blank VETRO_CABINET_ID results")
        subs_by_cabinet = pd.concat([subs_by_cabinet_non_blank, subs_by_market_blank], ignore_index=True)
    else:
        subs_by_cabinet = subs_by_cabinet_non_blank
    
    # Format WEIGHTED_AVG_SERVICEABLE_DT and sort
    logger.info("Formatting dates and sorting results")
    subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'] = pd.to_datetime(subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'])
    subs_by_cabinet.sort_values(
        by=['WEIGHTED_AVG_SERVICEABLE_DT', 'VETRO_CABINET_ID'], 
        inplace=True
        )
    subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'] = subs_by_cabinet['WEIGHTED_AVG_SERVICEABLE_DT'].dt.strftime('%m/%d/%Y')
    
    # Reorder columns
    desired_order = [
        'CONFORMED_MARKET_SUMMARY_NAME', 'WEIGHTED_AVG_SERVICEABLE_DT', 
        'COMPETITIVE_TYPE_NAME', 'VETRO_CABINET_ID', 'VETRO_CABINET_ID_COUNT'
    ] + [col for col in subs_by_cabinet.columns if col not in [
        'CONFORMED_MARKET_SUMMARY_NAME', 'WEIGHTED_AVG_SERVICEABLE_DT', 
        'COMPETITIVE_TYPE_NAME', 'VETRO_CABINET_ID', 'VETRO_CABINET_ID_COUNT'
    ]]
    
    subs_by_cabinet = subs_by_cabinet[desired_order]
    
    # Save processed data
    output_file = os.path.join(SAVE_PATH, 'subs_by_cabinet.csv')
    subs_by_cabinet.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to: {output_file}")
    
    return subs_by_cabinet, merged_data

# =========================================================================
# ENTRY POINT
# This code runs when the script is executed directly
# =========================================================================
if __name__ == "__main__":
    # Run the main function and capture the results
    subs_by_cabinet, merged_data = main()
    
    # Display a sample of the results
    if subs_by_cabinet is not None:
        print("\nSample of subs_by_cabinet dataframe:")
        print(subs_by_cabinet.head())
