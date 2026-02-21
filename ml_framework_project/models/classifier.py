from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots the confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


class FraudNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class Classifier:
    """
    A generic Classifier class to train and evaluate various machine learning models with hyperparameter tuning.

    Supported Models:
    - Logistic Regression
    - Random Forest
    - SVM (Support Vector Machine)
    - KNN (K-Nearest Neighbors)
    - Decision Tree
    - Naive Bayes
    - ANN (Artificial Neural Network)
    - PyTorch NN (Custom Neural Network using PyTorch)
    """

    def __init__(self):
        self.model = None
        self.is_pytorch = False

    def fit(
        self,
        X_train: pd.DataFrame or np.ndarray,
        y_train: pd.Series or np.ndarray,
        model_name: str = "Logistic Regression",
        perform_tuning: bool = False,
    ):
        self.is_pytorch = model_name == "PyTorch NN"

        if self.is_pytorch:
            X_tensor = torch.tensor(
                X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                dtype=torch.float32,
            )
            y_tensor = torch.tensor(
                y_train.values if isinstance(y_train, pd.Series) else y_train,
                dtype=torch.float32,
            ).unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            self.model = FraudNN(input_dim=X_tensor.shape[1])
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            print(f"Training {model_name} for 10 epochs...")
            self.model.train()
            for epoch in range(10):
                running_loss = 0.0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
            print(f"Model '{model_name}' trained successfully.")
            return

        if model_name == "Logistic Regression":
            self.model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        elif model_name == "SVM":
            self.model = SVC(probability=True)
        elif model_name == "KNN":
            self.model = KNeighborsClassifier()
        elif model_name == "Decision Tree":
            self.model = DecisionTreeClassifier()
        elif model_name == "Naive Bayes":
            self.model = GaussianNB()
        elif model_name == "ANN":
            self.model = MLPClassifier(max_iter=500)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        print(f"Training {model_name}...")
        self.model.fit(X_train, y_train)
        print(f"Model '{model_name}' trained successfully.")

    def predict(self, X_test: pd.DataFrame or np.ndarray) -> np.ndarray:
        if self.model is None:
            raise Exception("Model has not been trained yet. Please call fit() first.")

        if self.is_pytorch:
            self.model.eval()
            X_tensor = torch.tensor(
                X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
                dtype=torch.float32,
            )
            with torch.no_grad():
                probs = self.model(X_tensor).numpy()
                return (probs > 0.5).astype(int).flatten()

        return self.model.predict(X_test)

    def predict_proba(self, X_test: pd.DataFrame or np.ndarray) -> np.ndarray:
        if self.model is None:
            raise Exception("Model has not been trained yet.")

        if self.is_pytorch:
            self.model.eval()
            X_tensor = torch.tensor(
                X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
                dtype=torch.float32,
            )
            with torch.no_grad():
                probs = self.model(X_tensor).numpy().flatten()
                return probs

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        else:
            return self.model.predict(X_test)

    def score(
        self,
        X: pd.DataFrame or np.ndarray,
        y: pd.Series or np.ndarray,
        metric: str = "f1",
    ) -> float:
        if self.model is None:
            raise Exception("Model has not been trained yet. Please call fit() first.")

        y_pred = self.predict(X)

        if metric == "accuracy":
            return accuracy_score(y, y_pred)
        elif metric == "precision":
            return precision_score(y, y_pred, zero_division=0)
        elif metric == "recall":
            return recall_score(y, y_pred, zero_division=0)
        elif metric == "f1":
            return f1_score(y, y_pred, zero_division=0)
        elif metric == "roc_auc":
            y_probs = self.predict_proba(X)
            return roc_auc_score(y, y_probs)
        elif metric == "pr_auc":
            y_probs = self.predict_proba(X)
            return average_precision_score(y, y_probs)
        else:
            raise ValueError(f"Metric '{metric}' is not supported.")
