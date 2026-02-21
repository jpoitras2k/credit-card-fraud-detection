import pandas as pd
import os


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return pd.DataFrame()


def read_excel(file_path: str) -> pd.DataFrame:
    """
    Reads an Excel file and returns a DataFrame.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return pd.DataFrame()


def read_json(file_path: str) -> pd.DataFrame:
    """
    Reads a JSON file and returns a DataFrame.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the JSON file.
    """
    try:
        df = pd.read_json(file_path)
        return df
    except Exception as e:
        print(f"Error reading the JSON file: {e}")
        return pd.DataFrame()


def read_parquet(file_path: str) -> pd.DataFrame:
    """
    Reads a Parquet file and returns a DataFrame.

    Parameters:
    file_path (str): The path to the Parquet file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the Parquet file.
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error reading the Parquet file: {e}")
        return pd.DataFrame()


def read_data(file_path: str) -> pd.DataFrame:
    """
    Reads a data file (CSV, Excel, JSON, Parquet) and returns a DataFrame.

    Parameters:
    file_path (str): The path to the data file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the file.
    """
    if file_path.endswith(".csv"):
        return read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return read_excel(file_path)
    elif file_path.endswith(".json"):
        return read_json(file_path)
    elif file_path.endswith(".parquet"):
        return read_parquet(file_path)
    else:
        print("Unsupported file format.")
        return pd.DataFrame()
