import pandas as pd


def one_hot_encode(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """
    Encodes one or more categorical columns using one-hot encoding.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    *columns (str): The categorical columns to encode.

    Returns:
    pd.DataFrame: DataFrame with the specified columns one-hot encoded.
    """
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
    return df


def label_encode(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """
    Encodes one or more categorical columns using label encoding (assigning a unique integer to each category).

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    *columns (str): The categorical columns to encode.

    Returns:
    pd.DataFrame: DataFrame with the specified columns label encoded.
    """
    for column in columns:
        df[f"{column}_encoded"] = df[column].astype("category").cat.codes
    return df


def ordinal_encode(df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
    """
    Encodes a categorical column using ordinal encoding based on a provided mapping.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): The categorical column to encode.
    mapping (dict): A dictionary mapping categories to integer values.

    Returns:
    pd.DataFrame: DataFrame with the specified column ordinal encoded.
    """
    df[f"{column}_encoded"] = df[column].map(mapping)
    return df
