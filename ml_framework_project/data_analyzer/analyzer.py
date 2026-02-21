import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_eda(df: pd.DataFrame):
    """
    Performs Exploratory Data Analysis (EDA) on the DataFrame.
    Prints summary statistics, info, and correlation matrix.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    """
    print("\n--- Exploratory Data Analysis ---")
    print("\nData Head:")
    print(df.head())

    print("\nData Info:")
    print(df.info())

    print("\nData Description:")
    print(df.describe())

    try:
        # Select only numeric columns for correlation matrix
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            print("\nCorrelation Matrix:")
            print(numeric_df.corr())
        else:
            print("\nNo numeric columns found for correlation matrix.")
    except Exception as e:
        print(f"\nCould not calculate correlation matrix: {e}")


def visualize_data(df: pd.DataFrame):
    """
    Visualizes the data using basic plots and displays them.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    """
    print(f"\n--- Data Visualization ---")
    print(f"Displaying plots...")

    numeric_columns = df.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        except Exception as e:
            print(f"Could not plot histogram for {col}: {e}")

    # Pairplot for a subset of columns if dataset is large, or all if small
    key_columns = ["price_usd", "carat_weight", "depth_mm", "cut_quality_encoded"]
    available_keys = [col for col in key_columns if col in df.columns]

    if available_keys:
        try:
            sns.pairplot(df[available_keys])
            plt.show()
        except Exception as e:
            print(f"Could not create pairplot: {e}")
