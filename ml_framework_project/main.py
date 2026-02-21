"""
Main execution script for the Credit Card Fraud Detection project.

This script orchestrates the pipeline:
1. Data Loading: Reading the credit card fraud dataset.
2. Data Preprocessing: Scaling Time and Amount features.
3. Classification: Predicting fraudulent transactions.
"""

import pandas as pd
import numpy as np
from ml_framework_project.data_analyzer.data_preprocessing import (
    preprocess_creditcard_data,
    shuffle_dataframe,
)
from ml_framework_project.models.classifier import Classifier, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from ml_framework_project.datasets.credit_card import get_creditcard_data


def fraud_classification_pipeline(df: pd.DataFrame):
    """
    Runs the classification pipeline to predict 'Class' (fraud).

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
    """
    print("\n--- Starting Fraud Classification Pipeline ---")
    if "Class" not in df.columns:
        print("Error: Target variable 'Class' not found.")
        return None

    # We will sample the dataset if it's too large for quick verification, but ideally we use all.
    # The dataset has ~284k rows, which is fine for most algorithms, but we can shuffle it.
    df = shuffle_dataframe(df, random_state=42)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Handle any potential remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split the data into Training (80%) and Testing (20%) sets
    # We use random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Fraud cases in test set: {sum(y_test)}")

    # Models to test: we use PyTorch NN and Random Forest (robust to imbalance)
    models_to_test = ["PyTorch NN", "Random Forest"]
    best_model_name = ""
    best_pr_auc = 0
    best_clf = None

    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        clf = Classifier()

        try:
            clf.fit(X_train, y_train, model_name=model_name, perform_tuning=False)

            # Since fraud is rare, Precision-Recall AUC (PR-AUC) is the best metric
            pr_auc = clf.score(X_test, y_test, metric="pr_auc")
            f1 = clf.score(X_test, y_test, metric="f1")

            print(f"{model_name} PR-AUC: {pr_auc:.4f}")
            print(f"{model_name} F1-Score: {f1:.4f}")

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_model_name = model_name
                best_clf = clf

        except Exception as e:
            print(f"Failed to train {model_name}: {e}")

    if best_clf:
        print(f"\nBest Model: {best_model_name} with PR-AUC: {best_pr_auc:.4f}")
        print(f"Plotting Confusion Matrix for {best_model_name}...")
        y_pred = best_clf.predict(X_test)
        plot_confusion_matrix(
            y_test, y_pred, title=f"Confusion Matrix - {best_model_name}"
        )

    return best_clf


def main():
    """
    Main entry point of the application.
    """
    print("=== Credit Card Fraud Predictor ===")

    # 1. Load Data
    print("\nLoading dataset...")
    df = get_creditcard_data()

    if df.empty:
        print("Exiting pipeline due to missing dataset.")
        return

    # 2. Preprocess Data
    print("\nData Preprocessing: Scaling Time and Amount.")
    df_processed = preprocess_creditcard_data(df)

    # 3. Run Classification
    fraud_classification_pipeline(df_processed)


if __name__ == "__main__":
    main()
