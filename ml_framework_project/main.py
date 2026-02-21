"""
Main execution script for the Credit Card Fraud Detection project.
Curriculum Alignment: RoboGarden Bootcamp

Phase 4 & 5 execution: trains and evaluates classical and deep learning models,
then creates a comparative visualization of their performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_framework_project.data_analyzer.data_preprocessing import (
    preprocess_creditcard_data,
    shuffle_dataframe,
)
from ml_framework_project.models.classifier import Classifier
from ml_framework_project.models.clustering import Clustering
from sklearn.model_selection import train_test_split
from ml_framework_project.datasets.credit_card import get_creditcard_data
import os


def run_project_pipeline(df: pd.DataFrame):
    if "Class" not in df.columns:
        print("Error: Target variable 'Class' not found.")
        return

    # To ensure training happens in a reasonable time for all models, we can sample the dataset
    # while maintaining the fraud ratio, but for exactitude we use the shuffled full dataset.
    df = shuffle_dataframe(df, random_state=42)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Phase 2 & 3: Model Listing
    classification_models = [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Keras ANN",
        "Keras CNN",
        "Keras RNN",
    ]

    results = []

    # Phase 4: Model Training and Evaluation
    for model_name in classification_models:
        print(f"\\n--- Evaluating {model_name} ---")
        clf = Classifier()
        try:
            # Linear Regression isn't a true classifier, so we skip grid search tuning for it
            clf.fit(X_train, y_train, model_name=model_name, perform_tuning=False)

            # Record Metrics
            acc = clf.score(X_test, y_test, metric="accuracy")
            f1 = clf.score(X_test, y_test, metric="f1")
            pr_auc = clf.score(X_test, y_test, metric="pr_auc")

            print(
                f"{model_name} - Acc: {acc:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f}"
            )
            results.append(
                {"Model": model_name, "Accuracy": acc, "F1-Score": f1, "PR-AUC": pr_auc}
            )
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")

    # Adding Unsupervised Clustering Test as requested in curriculum
    print(f"\\n--- Evaluating Unsupervised K-Means ---")
    clu = Clustering()
    try:
        # We sample a smaller set for k-means speed
        X_sample = X_train.sample(n=10000, random_state=42)
        clu.fit(X_sample, model_name="k-Means", n_clusters=2)
        score = clu.score(X_sample, metric="silhouette")
        print(f"K-Means Silhouette Score on sample: {score:.4f}")
    except Exception as e:
        print(f"Failed to train K-Means: {e}")

    # Phase 5: Final Performance Results & Plots
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        print("\\n--- Final Model Comparison ---")
        print(results_df.to_string(index=False))

        # Melt dataframe for seaborn categorical plotting
        results_melted = results_df.melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )

        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=results_melted, x="Model", y="Score", hue="Metric", palette="viridis"
        )
        plt.title("Phase 5: Final Performance Model Comparison (Credit Card Fraud)")
        plt.ylim(0, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if not os.path.exists("plots"):
            os.makedirs("plots")

        plt.savefig("plots/model_comparison.png")
        print("Saved performance plot to 'plots/model_comparison.png'")
        plt.show()


def main():
    print("=== RoboGarden Credit Card Fraud Project ===")

    print("\\nPhase 0 & 1: Data Acquisition and Preprocessing")
    df = get_creditcard_data()

    if df.empty:
        print("Exiting pipeline due to missing dataset.")
        return

    df_processed = preprocess_creditcard_data(df)

    run_project_pipeline(df_processed)


if __name__ == "__main__":
    main()
