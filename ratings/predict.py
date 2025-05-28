import pandas as pd
import numpy as np
import argparse
import requests
import joblib
import json
import io
import os
from datetime import datetime, timedelta
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict TV ratings from a CSV file')
    parser.add_argument('input_file', help='Path to the input CSV file')
    return parser.parse_args()

def load_input_data(file_path):
    """Load the input data from a CSV file"""
    print(f"Loading input data from {file_path}")
    try:
        # Semi-colon separated with proper encoding
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Rename columns to match the expected format
        column_mapping = {
            'Programma': 'description',
            'Zender': 'channel',
            'Datum': 'dateResult',
            'Start': 'startTime',
            'Duur': 'rLength'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all expected columns are present
        required_columns = ['description', 'channel', 'dateResult', 'startTime', 'rLength']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in input file")
        
        return df
    
    except Exception as e:
        print(f"Error loading input data: {e}")
        raise

def convert_datetime_formats(df):
    """Convert date and time formats to match the expected format"""
    print("Converting date and time formats")
    
    # Convert dateResult from d/mm/yyyy to yyyy-mm-ddT00:00:00.000000
    df['dateResult'] = pd.to_datetime(df['dateResult'], format='%d/%m/%Y').dt.strftime('%Y-%m-%dT00:00:00.000000')
    
    # Ensure startTime is in HH:MM:SS format
    df['startTime'] = df['startTime'].apply(lambda x: x if ':' in x else f"{x[:2]}:{x[2:4]}:{x[4:6]}" if len(x) >= 6 else x)
    
    # Ensure rLength is in HH:MM:SS format
    def format_duration(duration):
        # Check if duration is already in correct format
        if ':' in duration:
            parts = duration.split(':')
            if len(parts) == 2:  # MM:SS format
                return f"00:{parts[0]}:{parts[1]}"
            return duration
        return duration  # Return as is if can't parse
    
    df['rLength'] = df['rLength'].apply(format_duration)
    
    return df

def load_genre_data():
    """Load genre data from GitHub"""
    url = 'https://raw.githubusercontent.com/emilegezels/project-broadcast/refs/heads/main/kijkcijfers_genres.csv'

    try:
        # Fetch data from URL
        print("Fetching genre data from GitHub...")
        response = requests.get(url)
        response.raise_for_status()

        # Parse genre data
        data = pd.read_csv(
            StringIO(response.text),
            delimiter=';'
        )

        print(f"Genre data loaded: {data.shape}")
        return data

    except Exception as e:
        print(f"Error loading genre data: {e}")
        return pd.DataFrame()

def load_weather_data():
    """Load weather data from GitHub"""
    url = 'https://raw.githubusercontent.com/emilegezels/project-broadcast/refs/heads/main/aws_daily.csv'

    try:
        # Fetch data from URL
        print("Fetching weather data from GitHub...")
        response = requests.get(url)
        response.raise_for_status()

        # Parse weather data
        weather_raw = pd.read_csv(
            StringIO(response.text),
            parse_dates=['timestamp']
        )

        # Process weather data
        weather_raw['date'] = pd.to_datetime(weather_raw['timestamp']).dt.date

        # Keep only needed columns
        columns_to_keep = ['date', 'temp_max']
        weather_df = weather_raw[columns_to_keep]

        # Group by date and calculate averages
        data = weather_df.groupby('date').mean(numeric_only=True).reset_index()

        print(f"Weather data loaded: {data.shape}")
        return data

    except Exception as e:
        print(f"Error loading weather data: {e}")
        return pd.DataFrame()

def merge_genre_data(X, genre_data):
    """Merge genre data with input dataframe"""
    # Check if genre data is available
    if genre_data is None or len(genre_data) == 0:
        print("Warning: Genre data not available, skipping merge")
        return X

    # Check if required columns exist
    if 'description' not in X.columns or 'channel' not in X.columns:
        print("Warning: Required columns missing for genre merge")
        return X

    # Make a copy to avoid modifying the original
    df = X.copy()

    # Merge genre data
    result = pd.merge(df, genre_data, on=['description', 'channel'], how='left')

    return result

def merge_weather_data(X, weather_data):
    """Merge weather data with input dataframe"""
    # Check if weather data is available
    if weather_data is None or len(weather_data) == 0:
        print("Warning: Weather data not available, skipping merge")
        return X

    # Check if required columns exist
    if 'date' not in X.columns:
        print("Warning: Date column missing, cannot merge weather data")
        return X

    # Make a copy to avoid modifying the original
    df = X.copy()

    # Ensure date is in correct format
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Merge weather data
    result = pd.merge(df, weather_data, on='date', how='left')

    return result

def clean_description_field(X):
    """Clean the description field by removing commas"""
    print("Cleaning description field")

    df = X.copy()

    # Replace commas in description
    if 'description' in df.columns:
        df['description'] = df['description'].str.replace(',', ' ')

    return df

def clean_and_standardize_channels(X):
    """Cleans and standardizes channel names"""
    print("Converting channels")

    # Define channel mappings
    channel_mappings = {
        # Canvas variations
        'VRT CANVAS': 'CANVAS',

        # VRT 1 / EEN variations
        'VRT 1': 'EEN',

        # PLAY channels (formerly VIER, VIJF, ZES)
        'VIER': 'PLAY4',
        'VIJF': 'PLAY5',
        'ZES': 'PLAY6',

        # VTM family
        'VITAYA': 'VTM3',
        'CAZ': 'VTM4',
        'OP 12': 'VTM2',
        'Q2': 'VTM2',

        # Sports channels
        'ELEVEN PRO LEAGUE 1 NL': 'DAZN PRO LEAGUE 1 (NL)',
    }

    df = X.copy()

    # Convert all channel names to uppercase for consistency
    df['channel'] = df['channel'].str.upper()

    # Filter out composite channels
    mask = df['channel'].str.contains('/|,', regex=True, na=False)
    df = df[~mask]

    # Apply the mappings to standardize channel names
    df['channel'] = df['channel'].replace(channel_mappings)

    return df

def extract_time_features(X):
    """
    Extracts and transforms time-related features from the data
    """
    print("Extracting time features")

    df = X.copy()

    # Helper function to convert time strings to seconds
    def time_str_to_seconds(time_str):
        """Convert a time string in format 'HH:MM:SS' to total seconds"""
        try:
            hours, minutes, seconds = map(int, time_str.split(':'))
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, AttributeError) as e:
            print(f"Error converting time {time_str} to seconds: {e}")
            return 0

    # Clean start time
    df['startTime'] = df['startTime'].replace('', np.nan)
    df['startTime'] = df['startTime'].replace('Start time', np.nan)
    df = df.dropna(subset=['startTime'])

    # Convert time strings to seconds
    df['startTimeSeconds'] = df['startTime'].apply(time_str_to_seconds)
    df['durationSeconds'] = df['rLength'].apply(time_str_to_seconds)
    df['endTimeSeconds'] = df['startTimeSeconds'] + df['durationSeconds']

    # Extract date features
    df['date'] = pd.to_datetime(df['dateResult']).dt.date
    df['year'] = pd.to_datetime(df['dateResult']).dt.year
    df['dayofweek'] = pd.to_datetime(df['dateResult']).dt.dayofweek + 1

    # Keep only relevant columns
    cols_to_keep = [
        'date', 'year', 'dayofweek', 'startTimeSeconds',
        'durationSeconds', 'endTimeSeconds', 'description',
        'channel', 
    ]

    # Filter to only include columns that exist in the dataframe
    existing_cols = [col for col in cols_to_keep if col in df.columns]

    return df[existing_cols]

def add_primetime_features(X):
    """Adds primetime feature"""
    print("Adding primetime feature")

    df = X.copy()

    # Create hour feature from startTimeSeconds
    df['hour'] = (df['startTimeSeconds'] // 3600).astype(int)

    # Calculate is_primetime (19:00-21:00)
    df['is_primetime'] = ((df['hour'] >= 19) & (df['hour'] <= 21)).astype(int)

    return df

def episode_number_transformer(X, min_occurrences=5):
    """
    Calculate episode sequence number for recurring programs
    """
    print("Calculating episode numbers for recurring programs")
    
    df = X.copy()

    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Initialize episode_in_season feature
    df['episode_in_season'] = 0

    # Identify recurring programs
    description_counts = df['description'].value_counts()
    recurring_mask = description_counts >= min_occurrences
    if recurring_mask.sum() == 0:
        return df  # No recurring series found

    recurring_descriptions = description_counts[recurring_mask].index

    # Pre-filter to only recurring descriptions to reduce dataset size
    recurring_df = df[df['description'].isin(recurring_descriptions)].copy()
    recurring_df['desc_date'] = recurring_df.apply(
        lambda x: (x['description'], x['date']), axis=1
    )

    # Group by description and compute all at once
    groups = recurring_df.groupby('description')

    # Process each description group
    for desc, group in groups:
        if len(group) < min_occurrences:
            continue

        # Sort by date
        sorted_group = group.sort_values('date')

        # Calculate days between broadcasts
        sorted_group['prev_date'] = sorted_group['date'].shift(1)
        sorted_group['days_since_last'] = (
            sorted_group['date'] - sorted_group['prev_date']
        ).dt.days

        # Identify season breaks
        typical_gap = sorted_group['days_since_last'].median()
        season_break_threshold = max(typical_gap * 3, 14)

        # Mark season starts and assign season numbers
        sorted_group['season_start'] = (
            (sorted_group['days_since_last'] > season_break_threshold) |
            (sorted_group['days_since_last'].isna())
        )
        sorted_group['season_number'] = sorted_group['season_start'].cumsum()

        # Calculate episode numbers
        sorted_group['episode_in_season'] = sorted_group.groupby('season_number').cumcount() + 1

        # Update original dataframe
        df.loc[sorted_group.index, 'episode_in_season'] = sorted_group['episode_in_season']

    return df

def preprocess_data(input_df):
    """Apply all preprocessing steps to the input data"""
    df = input_df.copy()
    
    # Apply transformations
    df = convert_datetime_formats(df)
    df = clean_description_field(df)
    df = clean_and_standardize_channels(df)
    df = extract_time_features(df)
    df = add_primetime_features(df)
    df = episode_number_transformer(df)
    
    # Load external data
    genre_df = load_genre_data()
    weather_df = load_weather_data()
    
    # Merge external data
    df = merge_genre_data(df, genre_df)
    df = merge_weather_data(df, weather_df)
    
    # Handle missing values in categorical features
    if 'main_category' in df.columns and df['main_category'].isna().any():
        df['main_category'] = df['main_category'].fillna('unknown')
        
    if 'sub_category' in df.columns and df['sub_category'].isna().any():
        df['sub_category'] = df['sub_category'].fillna('unknown')
    
    # Handle missing weather data
    if 'temp_max' in df.columns and df['temp_max'].isna().any():
        # Fill with the median temperature
        median_temp = df['temp_max'].median()
        df['temp_max'] = df['temp_max'].fillna(median_temp)
    
    return df

def load_model():
    """Load the trained model from local file"""
    model_path = 'best_model.pkl'
    print(f"Loading model from {model_path}")
    try:
        # Load the model from local file
        model = joblib.load(model_path)
        print("Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def make_predictions(model, processed_data):
    """Use the model to make predictions"""
    print("Making predictions")
    
    # Columns expected by the model
    expected_columns = [
        'year', 'dayofweek', 'startTimeSeconds', 'durationSeconds', 
        'endTimeSeconds', 'channel', 'hour', 'is_primetime', 
        'episode_in_season', 'main_category', 'sub_category', 'temp_max'
    ]
    
    # Check for missing columns and handle them
    for col in expected_columns:
        if col not in processed_data.columns:
            if col == 'episode_in_season':
                processed_data[col] = 0
            elif col in ['main_category', 'sub_category']:
                processed_data[col] = 'unknown'
            else:
                print(f"Warning: Column {col} is missing and cannot be automatically filled")
    
    # Select only the columns needed for prediction
    # Drop any additional columns that aren't in expected_columns
    pred_columns = [col for col in expected_columns if col in processed_data.columns]
    X = processed_data[pred_columns]
    
    # Make predictions
    try:
        predictions = model.predict(X)
        processed_data['predicted_rateInK'] = predictions.astype(int)
        return processed_data
    except Exception as e:
        print(f"Error during prediction: {e}")
        # More detailed error information for debugging
        print(f"X shape: {X.shape}")
        print(f"X columns: {X.columns.tolist()}")
        print(f"X data types: {X.dtypes}")
        raise

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def display_predictions(predictions_df):
    """Display predictions in the console"""
    print("\n===== PREDICTED TV RATINGS =====\n")
    
    # Create a display-friendly dataframe
    # Ensure we have startTime - it might be missing in the output after transformations
    if 'startTime' not in predictions_df.columns:
        # Try to recreate it from startTimeSeconds
        if 'startTimeSeconds' in predictions_df.columns:
            predictions_df['startTime'] = predictions_df['startTimeSeconds'].apply(seconds_to_time)
    
    # Similarly check for rLength
    if 'rLength' not in predictions_df.columns:
        if 'durationSeconds' in predictions_df.columns:
            predictions_df['rLength'] = predictions_df['durationSeconds'].apply(seconds_to_time)
    
    # Select relevant columns
    display_df = predictions_df[['description', 'channel', 'date', 'startTime', 'rLength', 'predicted_rateInK']].copy()
    
    # Format the date column
    display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%d/%m/%Y')
    
    # Rename columns for display
    display_df.columns = ['Program', 'Channel', 'Date', 'Start Time', 'Duration', 'Predicted Viewers (thousands)']
    
    # Sort by predicted viewers (descending)
    display_df = display_df.sort_values('Predicted Viewers (thousands)', ascending=False)
    
    # Print the formatted table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 0)  # No decimal places for viewer numbers
    
    print(display_df.to_string(index=False))
    print("\n==============================\n")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load and preprocess data
    input_df = load_input_data(args.input_file)
    processed_df = preprocess_data(input_df)
    
    # Load model and make predictions
    model = load_model()
    predictions_df = make_predictions(model, processed_df)
    
    # Display predictions to console
    display_predictions(predictions_df)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()