"""
Data Preprocessing Module for Credit Card Fraud Dataset

This module provides functions for cleaning, transforming, and scaling
the credit card dataset.
"""

import pandas as pd
from ml_framework_project.data_analyzer.scaler import standard_scaler


def preprocess_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the credit card dataset.

    Steps involved:
    1. Drop rows with missing values (though this dataset usually has none).
    2. Standardize 'Time' and 'Amount' features since V1-V28 are PCA scaled.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # 1. Drop missing values just in case
    df_cleaned = df.dropna()

    # 2. Standardize Time and Amount
    if "Time" in df_cleaned.columns and "Amount" in df_cleaned.columns:
        df_scaled = standard_scaler(df_cleaned.copy(), "Time", "Amount")
        # Drop the original columns to only keep scaled
        df_scaled = df_scaled.drop(columns=["Time", "Amount"])
        # Rename the standardized columns back to typical names or keep them as _standardized

    else:
        df_scaled = df_cleaned
    return df_scaled


def shuffle_dataframe(df: pd.DataFrame, random_state: int = None) -> pd.DataFrame:
    """
    Shuffles the rows of the DataFrame.
    """
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def sample_dataframe(
    df: pd.DataFrame, n: int, random_state: int = None
) -> pd.DataFrame:
    """
    Samples n random rows from the DataFrame.
    """
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)
