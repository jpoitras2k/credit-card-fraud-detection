"""
Main execution script for the Credit Card Fraud Detection project.
Curriculum Alignment: RoboGarden Bootcamp

Phase 4 & 5 execution: trains and evaluates classical and deep learning models,
then creates a comparative visualization of their performance.

Phase 6: Interactive UX - Command line interface to let the user select models.
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


def run_project_pipeline(df: pd.DataFrame, selected_models: list, run_clustering: bool = False):
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

    print(f"\\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    results = []
    best_pr_auc = -1.0
    best_model_name = ""
    best_clf = None

    # Phase 4: Model Training and Evaluation (Supervised)
    if selected_models:
        for model_name in selected_models:
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

                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    best_model_name = model_name
                    best_clf = clf
            except Exception as e:
                print(f"Failed to train {model_name}: {e}")

    # Adding Unsupervised Clustering Test as requested in curriculum
    if run_clustering:
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

        # Save Best Model Logic
        if best_clf:
            print(f"\\n--- Exporting Best Model ---")
            extension = ".keras" if best_clf.is_keras else ".pkl"
            safe_name = best_model_name.replace(" ", "_").lower()
            filepath = os.path.join("saved_models", f"best_model_{safe_name}{extension}")
            print(f"Highest PR-AUC achieved: {best_pr_auc:.4f} ({best_model_name})")
            best_clf.save(filepath)

        # Do not plot if only 1 model was evaluated
        if len(results_df) > 1:
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
            print("\\nSaved performance plot to 'plots/model_comparison.png'")
            plt.show()


def main():
    print("=== RoboGarden Credit Card Fraud Project ===")

    print("\\nPhase 0 & 1: Data Acquisition and Preprocessing...")
    df = get_creditcard_data()

    if df.empty:
        print("Exiting pipeline due to missing dataset.")
        return

    df_processed = preprocess_creditcard_data(df)
    
    all_models = [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Keras ANN",
        "Keras CNN",
        "Keras RNN",
    ]
    classical_models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    keras_models = ["Keras ANN", "Keras CNN", "Keras RNN"]

    while True:
        print("\\n" + "="*50)
        print("          INTERACTIVE MODEL SELECTION MENU          ")
        print("="*50)
        print("1. Run ALL Models (Classical + Keras Deep Learning + Clustering) -- May take a while to run ")
        print("2. Run ONLY Fast Classical Models")
        print("3. Run ONLY Keras Deep Learning Models -- May take a while to run")
        print("4. Run a Specific Custom Model")
        print("5. Run ONLY Unsupervised Clustering (K-Means)")
        print("6. Exit")
        print("="*50)
        
        choice = input("Please select an option (1-6): ").strip()
        
        if choice == '1':
            run_project_pipeline(df_processed, all_models, run_clustering=True)
        elif choice == '2':
            run_project_pipeline(df_processed, classical_models, run_clustering=False)
        elif choice == '3':
            run_project_pipeline(df_processed, keras_models, run_clustering=False)
        elif choice == '4':
            print(f"Available models: {', '.join(all_models)}")
            custom = input("Type the exact name of the model to run: ").strip()
            if custom in all_models:
                 run_project_pipeline(df_processed, [custom], run_clustering=False)
            else:
                 print(f"Invalid model name '{custom}'.")
        elif choice == '5':
            run_project_pipeline(df_processed, [], run_clustering=True)
        elif choice == '6':
            print("Exiting interactive menu. Goodbye!")
            break
        else:
            print("Invalid selection. Please type a number between 1 and 6.")


if __name__ == "__main__":
    main()
