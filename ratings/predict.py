import subprocess
import sys
import importlib

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = importlib.import_module(package)

install_and_import('nltk')
install_and_import('catboost')

import os
import pandas as pd
import numpy as np
import argparse
import requests
import tempfile
import joblib
import json
import string
import re
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import nltk #ignore warnings about nltk, it will be installed if missing
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#from google.colab import drive
#drive.mount('/content/drive')

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

def convert_datetime_formats(X):
    """Convert date and time formats to match the expected format"""
    print("Converting date and time formats")

    df = X.copy()

    # Convert dateResult from d/mm/yyyy to yyyy-mm-ddT00:00:00.000000
    df['date'] = pd.to_datetime(df['dateResult'], format='%d/%m/%Y').dt.strftime('%Y-%m-%dT00:00:00.000000')
    df.drop(columns=['dateResult'], inplace=True)

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
    url = 'https://raw.githubusercontent.com/emilegezels/project-broadcast/refs/heads/main/ratings/data/kijkcijfers_genres.csv'

    try:
        # Fetch data from URL
        print("Fetching genre data from GitHub...")
        response = requests.get(url)
        response.raise_for_status()

        # Parse genre data
        data = pd.read_csv(
            StringIO(response.text),
            delimiter='|'
        )

        print(f"Genre data loaded: {data.shape}")
        return data

    except Exception as e:
        print(f"Error loading genre data: {e}")
        return pd.DataFrame()

def get_brussels_forecast():
    """
    Retrieve the 14-day weather forecast for Brussels from Meteovista.be
    Returns DataFrame with date and max_temp columns
    """

    # URL for Brussels 14-day forecast
    url = "https://www.meteovista.be/Europa/Belgie/Brussel/4053951/weersverwachting-14dagen"

    # Send HTTP request
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # Extract JSON data from __NEXT_DATA__ script tag
    json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text, re.DOTALL)

    if json_match:
        # Parse JSON
        data = json.loads(json_match.group(1))

        # Extract forecast days
        forecast_days = data['props']['pageProps']['forecastData']

        # Extract dates and max temperatures
        dates = []
        max_temps = []

        for day in forecast_days:
            # Get date from intervalStart (format: 2025-05-20T02:00:00+02:00)
            date_str = day['intervalStart']['formatted'].split('T')[0]  # Just get the date part

            # Get max temperature from afternoon data
            max_temp = day['dayPart']['afternoon']['temperature']['air']

            dates.append(date_str)
            max_temps.append(max_temp)

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'temp_max': max_temps
        })

        return df

    return pd.DataFrame(columns=['date', 'temp_max'])  # Empty DataFrame if data not found

def merge_genre_data(X):
    """Merge genre data with input dataframe"""
    genre_data = load_genre_data()

    # Check if required columns exist
    if 'description' not in X.columns or 'channel' not in X.columns:
        print("Warning: Required columns missing for genre merge")
        return X

    # Make a copy to avoid modifying the original
    df = X.copy()

    df['description'] = df['description'].str.strip()
    df['channel'] = df['channel'].str.strip()

    # Remove duplicates from genre data before merging
    genre_data_clean = genre_data.drop_duplicates(subset=['description', 'channel'], keep='first').copy()
    genre_data_clean['description'] = genre_data_clean['description'].str.strip()
    genre_data_clean['channel'] = genre_data_clean['channel'].str.strip()

    # Merge genre data
    print("Merging genre data.")
    result = pd.merge(df, genre_data_clean, on=['description', 'channel'], how='left')

    if 'main_category' in result.columns:
        result['main_category'] = result['main_category'].astype('category')
    if 'sub_category' in result.columns:
        result['sub_category'] = result['sub_category'].astype('category')

    return result

def merge_weather_data(X):
    """Merge weather data with input dataframe"""
    weather_data = get_brussels_forecast()

    # Check if required columns exist
    if 'date' not in X.columns:
        print("Warning: Date column missing, cannot merge weather data")
        return X

    # Make a copy to avoid modifying the original
    df = X.copy()
    weather = weather_data.copy()

    # Ensure date is in correct format
    df['date'] = pd.to_datetime(df['date']).dt.date
    weather['date'] = pd.to_datetime(weather['date']).dt.date

    #print df info() and print weather_data info()
    print(df.info())
    print(weather.info())

    # Merge weather data
    result = pd.merge(df, weather, on='date', how='left')

    return result

def extract_time_features(X):
   """
   Extracts and transforms time-related features from the data

   This function:
   1. Converts time strings to seconds (start time, duration)
   2. Calculates end time in seconds
   3. Extracts date features (month, season, day of week)

   Parameters:
   -----------
   X : pandas.DataFrame
       Input data containing time-related columns

   Returns:
   --------
   pandas.DataFrame
       DataFrame with extracted time features
   """
   print("Extracting time features.")

   df = X.copy()

   # Helper function to convert time strings to seconds
   def time_str_to_seconds(time_str):
       """Convert a time string in format 'HH:MM:SS' to total seconds"""
       hours, minutes, seconds = map(int, time_str.split(':'))
       return hours * 3600 + minutes * 60 + seconds

   # Clean start time
   df['startTime'] = df['startTime'].replace('', pd.NA)
   df['startTime'] = df['startTime'].replace('Start time', pd.NA)
   df = df.dropna(subset=['startTime'])

   # Convert time strings to seconds
   df['startTimeSeconds'] = df['startTime'].apply(time_str_to_seconds)
   df['durationSeconds'] = df['rLength'].apply(time_str_to_seconds)
   df['endTimeSeconds'] = df['startTimeSeconds'] + df['durationSeconds']
   df['hour'] = (df['startTimeSeconds'] / 3600) % 24
   df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
   df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

   # Extract date features
   df['month'] = pd.to_datetime(df['date']).dt.month
   df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek + 1

   # Extract season as number (for OrdinalEncoder later)
   def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:  # [9, 10, 11]
            return 3  # Autumn

   df['season'] = df['month'].apply(get_season)

   return df

def add_manual_clustering_features(X):
    df = X.copy()

    # K-means inspired clusters (broader ranges)
    kmeans_conditions = [
        (df['hour'] >= 7) & (df['hour'] < 16),  # Daytime cluster (centered ~13h)
        (df['hour'] >= 16) & (df['hour'] <= 23.5)  # Evening cluster (centered ~20h)
    ]
    kmeans_choices = [0, 1]  # 0=daytime, 1=evening
    df['kmeans_cluster'] = np.select(kmeans_conditions, kmeans_choices, default=-1)

    # HDBSCAN inspired clusters (tighter ranges around centroids)
    hdbscan_conditions = [
        (df['hour'] >= 11.5) & (df['hour'] < 14.5),    # Afternoon cluster (13.2h centroid)
        (df['hour'] >= 18) & (df['hour'] < 19.5),  # Early evening (18.6h centroid)
        (df['hour'] >= 19.5) & (df['hour'] < 21),  # Primetime (20.0h centroid)
        (df['hour'] >= 21) & (df['hour'] < 22)     # Late evening (20.6h centroid)
    ]
    hdbscan_choices = [0, 1, 2, 3]
    df['hdbscan_cluster'] = np.select(hdbscan_conditions, hdbscan_choices, default=-1)

    return df

def add_anomaly(X):
    """Add anomaly column with default value of 0 for new predictions"""
    df = X.copy()
    df['anomaly'] = 0  # All new data is considered normal (not anomalous)
    return df

def clean_descriptions(df, text_column='description'):
    """
    Comprehensive text cleaning for TV program descriptions
    """
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

    df = df.copy()

    # Initialize tools
    lemmatizer = WordNetLemmatizer()
    dutch_stopwords = set(stopwords.words('dutch'))  # Assuming Belgian TV descriptions
    english_stopwords = set(stopwords.words('english'))  # Some might be in English
    all_stopwords = dutch_stopwords.union(english_stopwords)
    punctuation = set(string.punctuation)

    # TV-specific words to exclude (common but not meaningful for clustering)
    tv_general_words = ['programma', 'uitzending', 'aflevering', 'seizoen',
                       'episode', 'serie', 'show', 'presentatie', 'live']

    def clean_text(text):
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

        # Tokenize
        tokens = word_tokenize(text)

        # Filter tokens
        tokens = [word for word in tokens if word not in punctuation]
        tokens = [word for word in tokens if word.isalpha()]  # Only alphabetic
        tokens = [word for word in tokens if len(word) > 3]  # Words > 3 chars
        tokens = [word for word in tokens if word not in all_stopwords]
        tokens = [word for word in tokens if word not in tv_general_words]

        # Lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    print(f"Cleaning {text_column} field...")
    df[f'{text_column}_cleaned'] = df[text_column].apply(clean_text)

    # Remove any remaining TV general words and short words
    for word in tv_general_words:
        df[f'{text_column}_cleaned'] = df[f'{text_column}_cleaned'].str.replace(f'\\b{word}\\b', '', regex=True)

    df[f'{text_column}_cleaned'] = df[f'{text_column}_cleaned'].str.replace(r'\b\w{1,3}\b', '', regex=True)
    df[f'{text_column}_cleaned'] = df[f'{text_column}_cleaned'].str.replace(r'\s+', ' ', regex=True).str.strip()

    return df

def preprocessing_pipeline():
    """
    Creates a preprocessing pipeline
    """
    preprocessing_pipeline = Pipeline([
        ('convert_datetime_formats', FunctionTransformer(convert_datetime_formats, validate=False)),
        ('merge_genre_data', FunctionTransformer(merge_genre_data, validate=False)),
        ('merge_weather_data', FunctionTransformer(merge_weather_data, validate=False)),
        ('extract_time_features', FunctionTransformer(extract_time_features, validate=False)),
        ('add_manual_clustering_features', FunctionTransformer(add_manual_clustering_features, validate=False)),
        ('add_anomaly', FunctionTransformer(add_anomaly, validate=False)),
        ('clean_descriptions', FunctionTransformer(clean_descriptions, validate=False)),
    ])

    return preprocessing_pipeline

def load_model():
    """Load the trained model from GitHub URL"""
    model_url = 'https://media.githubusercontent.com/media/emilegezels/project-broadcast/refs/heads/main/Voting_Regressor_20250601_144915.pkl'
    print(f"Downloading model from {model_url}")

    try:
        # Create a temporary file to save the downloaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            # Download the file from the URL
            response = requests.get(model_url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses

            # Write the file content to the temporary file
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

            # Get the temporary file path
            temp_file_path = temp_file.name

        # Load the model from the temporary file
        print(f"Loading model from temporary file {temp_file_path}")
        model = joblib.load(temp_file_path)

        # Clean up the temporary file
        os.unlink(temp_file_path)

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
        'startTimeSeconds',
        'endTimeSeconds',
        'durationSeconds',
        'hour_sin',
        'hour_cos',
        'temp_max',
        'month',
        'dayofweek',
        'kmeans_cluster',
        'hdbscan_cluster',
        'season',
        'channel',
        'main_category',
        'sub_category',
        'description_cleaned',
        'anomaly'
    ]

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

    # Sort by predicted viewers (descending) - Fixed column name
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
    # Use transform() instead of fit_transform() since this is a prediction pipeline
    processed_df = preprocessing_pipeline().transform(input_df)

    print(f"Processed data loaded: {processed_df.shape}")
    print(processed_df.head())
    print("\n==============================\n")
    print(processed_df.info())

    # Load model and make predictions
    model = load_model()
    predictions_df = make_predictions(model, processed_df)

    # Display predictions to console
    display_predictions(predictions_df)

    print("Process completed successfully!")

if __name__ == "__main__":
    main()