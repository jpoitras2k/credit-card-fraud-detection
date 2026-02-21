import pandas as pd
from sklearn.preprocessing import StandardScaler


def standard_scaler(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """
    Standardizes one or more columns by subtracting the mean and dividing by the standard deviation.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    *columns (str): The columns to standardize.

    Returns:
    pd.DataFrame: DataFrame with the specified columns standardized.
    """
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df[f"{column}_standardized"] = (df[column] - mean) / std
    return df


def minmax_scaler(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """
    Normalizes one or more columns by scaling the values to a range of [0, 1] (Min-Max Scaling).

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    *columns (str): The columns to normalize.

    Returns:
    pd.DataFrame: DataFrame with the specified columns normalized.
    """
    for column in columns:
        min_value = df[column].min()
        max_value = df[column].max()
        df[f"{column}_normalized"] = (df[column] - min_value) / (max_value - min_value)
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fits a StandardScaler on the training data and transforms both training and testing data.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.

    Returns:
        tuple: (X_train_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
