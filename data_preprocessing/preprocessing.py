"""
data_preprocessing/preprocessing.py
-------------------------------------
Contains functions for data cleaning and feature engineering, including lag features, rolling statistics,
and end-to-end CSV data loading with scaling and train/test splitting.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_lag_features(df, column_name, lag_sizes):
    """
    Creates lag features for the specified column.

    Parameters:
        df (DataFrame): Input data.
        column_name (str): Column to generate lag features.
        lag_sizes (list): List of lag periods.
    
    Returns:
        DataFrame: DataFrame with new lag features.
    """
    try:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        for lag in lag_sizes:
            df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        return df
    except Exception as e:
        logging.error("Error in create_lag_features: " + str(e))
        raise

def create_rolling_window_features(df, column_name, window_sizes, method='pad'):
    """
    Creates rolling window features (mean and standard deviation) for the specified column.

    Parameters:
        df (DataFrame): Input data.
        column_name (str): Column to compute rolling statistics.
        window_sizes (list): List of window sizes.
        method (str): Method for handling NaN values ('pad' or 'interpolate').
    
    Returns:
        DataFrame: DataFrame with new rolling window features.
    """
    try:
        for window in window_sizes:
            df[f'{column_name}_rolling_mean_{window}'] = df[column_name].rolling(window=window).mean()
            df[f'{column_name}_rolling_std_{window}'] = df[column_name].rolling(window=window).std()
            if method == 'interpolate':
                df[f'{column_name}_rolling_mean_{window}'].interpolate(method='linear', inplace=True)
                df[f'{column_name}_rolling_std_{window}'].interpolate(method='linear', inplace=True)
            elif method == 'pad':
                df[f'{column_name}_rolling_mean_{window}'].fillna(method='pad', inplace=True)
                df[f'{column_name}_rolling_std_{window}'].fillna(method='pad', inplace=True)
            else:
                df[f'{column_name}_rolling_mean_{window}'].fillna(df[column_name].mean(), inplace=True)
                df[f'{column_name}_rolling_std_{window}'].fillna(df[column_name].std(), inplace=True)
        return df
    except Exception as e:
        logging.error("Error in create_rolling_window_features: " + str(e))
        raise

def preprocess_data_with_feature_engineering(data, lag_sizes=[1, 2, 3, 5, 10], window_sizes=[5, 10, 20]):
    """
    Applies feature engineering to the raw data by creating lag and rolling window features.

    Parameters:
        data (DataFrame): Raw input data.
        lag_sizes (list): List of lag periods.
        window_sizes (list): List of window sizes for rolling features.
    
    Returns:
        X (DataFrame): Engineered feature matrix.
        y (Series): Target variable.
    """
    try:
        if data.empty:
            raise ValueError("Input data is empty.")
        
        data.columns = data.columns.str.strip().str.lower()
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            reference_date = data['date'].min()
            data['days_since_reference'] = (data['date'] - reference_date).dt.days
        
        if 'close' not in data.columns:
            raise ValueError("Target column 'close' not found in data.")
        data = create_lag_features(data, 'close', lag_sizes)
        data = create_rolling_window_features(data, 'close', window_sizes)
        data.dropna(inplace=True)
        y = data['close']
        X = data.drop(columns=['close', 'date'], errors='ignore')
        return X, y
    except Exception as e:
        logging.error("Error in preprocess_data_with_feature_engineering: " + str(e))
        raise

def get_scaler(scaler_type):
    """
    Returns a scaler instance based on the scaler type.

    Parameters:
        scaler_type (str): One of "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", or "MaxAbsScaler".
    
    Returns:
        scaler: The appropriate scaler instance.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler
    scaler_type = scaler_type.lower()
    if scaler_type == "standardscaler":
        return StandardScaler()
    elif scaler_type == "minmaxscaler":
        return MinMaxScaler()
    elif scaler_type == "robustscaler":
        return RobustScaler()
    elif scaler_type == "normalizer":
        return Normalizer()
    elif scaler_type == "maxabsscaler":
        return MaxAbsScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

def preprocess_data(data_file_path, scaler_type="StandardScaler", model_type="neural_network", window_size=5):
    """
    Loads a CSV file and applies preprocessing including feature engineering, scaling, and train/test splitting.

    Parameters:
        data_file_path (str): Path to the CSV file.
        scaler_type (str): Type of scaler to use.
        model_type (str): Model type; if "lstm", windowed data is created.
        window_size (int): Number of time steps for LSTM windowing.
    
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed and split data.
    """
    try:
        data = pd.read_csv(data_file_path)
        X, y = preprocess_data_with_feature_engineering(data)
        scaler = get_scaler(scaler_type)
        X_scaled = scaler.fit_transform(X)
        
        if model_type.lower() == "lstm":
            from data_preprocessing.windowing import create_windowed_data
            X_windowed, y_windowed = create_windowed_data(X_scaled, y.values, window_size)
            X_scaled = X_windowed
            y = y_windowed
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in preprocess_data: " + str(e))
        raise
