import pandas as pd
import os
from ml_framework_project.data_analyzer.data_reader import read_data


def get_creditcard_data() -> pd.DataFrame:
    """
    Loads the credit card fraud dataset.

    Returns:
        pd.DataFrame: The credit card dataset.
    """
    # Assuming this script is at ml_framework_project/datasets/credit_card.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "creditcard.csv")

    # Use the reusable data_reader module
    df = read_data(file_path)

    if df.empty:
        print(
            f"Warning: Could not load data from {file_path}. Please check if the file exists."
        )
    else:
        print(f"Successfully loaded credit card data with shape: {df.shape}")

    return df
