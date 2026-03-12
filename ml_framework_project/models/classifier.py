from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots the confusion matrix using seaborn heatmap.
    # Visualizing true positives vs false positives is essential. 
    # Just like Dr. Ng says, you can't just rely on accuracy for imbalanced datasets! 
    # We must properly evaluate the F1 score and the PR curve.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def build_keras_ann(input_dim):
    # Building a standard Multi-Layer Perceptron.
    # I added dropout layers because avoiding over-parameterization is key for the bias-variance tradeoff!
    # A single hidden layer might theoretically approximate any continuous function,
    # but deeper architectures yield more robust abstract representations.
    model = keras.Sequential(
        [
            layers.Dense(32, activation="relu", input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model


def build_keras_cnn(input_dim):
    model = keras.Sequential(
        [
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.Conv1D(filters=32, kernel_size=2, activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        # Using Adam over SGD here because adaptive learning rates help us
        # traverse the saddle points in the non-convex loss surface much faster!
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model


def build_keras_rnn(input_dim):
    model = keras.Sequential(
        [
            layers.Reshape((1, input_dim), input_shape=(input_dim,)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model


class Classifier:
    """
    A generic Classifier class to train and evaluate various local models and keras neural networks.

    Supported Models:
    - Linear Regression (thresholded)
    - Logistic Regression
    - Random Forest
    - KNN
    - Decision Tree
    - Keras ANN (Artificial Neural Network)
    - Keras CNN (Convolutional Neural Network)
    - Keras RNN (Recurrent Neural Network)
    
    # I implemented all these models so we can have a proper benchmark. 
    # According to No Free Lunch Theorem, there is no one universal best model. We must empirically evaluate!
    """

    def __init__(self):
        self.model = None
        self.is_keras = False
        self.is_linear_reg = False

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        model_name: str = "Logistic Regression",
        perform_tuning: bool = False,
    ):
        self.is_keras = model_name in ["Keras ANN", "Keras CNN", "Keras RNN"]
        self.is_linear_reg = model_name == "Linear Regression"

        X_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_array = y_train.values if isinstance(y_train, pd.Series) else y_train
        input_dim = X_array.shape[1]

        if self.is_keras:
            if model_name == "Keras ANN":
                self.model = build_keras_ann(input_dim)
            elif model_name == "Keras CNN":
                self.model = build_keras_cnn(input_dim)
            elif model_name == "Keras RNN":
                self.model = build_keras_rnn(input_dim)

            print(
                f"Training {model_name} for 5 epochs (Early Stopping applied internally)..."
            )

            # Implementing Early Stopping to prevent overfitting.
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True
            )

            # Use a validation split to monitor PR-AUC
            self.model.fit(
                X_array,
                y_array,
                epochs=5,
                batch_size=256,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=1,
            )
            print(f"Model '{model_name}' trained successfully.")
            return

        if model_name == "Linear Regression":
            self.model = LinearRegression()
        elif model_name == "Logistic Regression":
            if perform_tuning:
                params = {"C": [0.1, 1.0, 10.0]}
                self.model = GridSearchCV(
                    LogisticRegression(max_iter=1000), params, scoring="accuracy", cv=2
                )
            else:
                self.model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        elif model_name == "KNN":
            if perform_tuning:
                params = {"n_neighbors": [3, 5, 7]}
                self.model = GridSearchCV(
                    KNeighborsClassifier(), params, scoring="accuracy", cv=2
                )
            else:
                self.model = KNeighborsClassifier()
        elif model_name == "Decision Tree":
            if perform_tuning:
                params = {"max_depth": [None, 10, 20]}
                self.model = GridSearchCV(
                    DecisionTreeClassifier(), params, scoring="accuracy", cv=2
                )
            else:
                self.model = DecisionTreeClassifier()
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        print(f"Training {model_name}...")
        self.model.fit(X_train, y_train)

        if perform_tuning and isinstance(self.model, GridSearchCV):
            print(f"Best parameters for {model_name}: {self.model.best_params_}")
            self.model = self.model.best_estimator_

        print(f"Model '{model_name}' trained successfully.")

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # Performing inference. Remember to ensure X_test has the same feature mapping 
        # as the training set, otherwise we'll have input space covariate shift!
        if self.model is None:
            raise Exception("Model has not been trained yet. Please call fit() first.")

        if self.is_keras:
            X_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            probs = self.model.predict(X_array, verbose=0)
            return (probs > 0.5).astype(int).flatten()

        if self.is_linear_reg:
            preds = self.model.predict(X_test)
            return (preds > 0.5).astype(int)

        return self.model.predict(X_test)

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise Exception("Model has not been trained yet.")

        if self.is_keras:
            X_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            probs = self.model.predict(X_array, verbose=0).flatten()
            return probs

        if self.is_linear_reg:
            return self.model.predict(X_test)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        else:
            return self.model.predict(X_test)

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
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

    def save(self, filepath: str):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        if self.is_keras:
            self.model.save(filepath)
        else:
            joblib.dump(self.model, filepath)
        print(f"Model successfully saved to: {filepath}")

