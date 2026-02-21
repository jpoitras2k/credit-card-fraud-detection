import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Regressor:
    """
    A generic Regressor class to train and evaluate various regression models with hyperparameter tuning.

    Supported Models:
    - Linear Regression
    - KNN (K-Nearest Neighbors)
    - Decision Tree
    - Random Forest
    - SVR (Support Vector Regressor)
    - ANN (Artificial Neural Network / MLPRegressor)

    Usage:
        >>> reg = Regressor()
        >>> reg.fit(X_train, y_train, model_name='Linear Regression')
        >>> r2_score = reg.score(X_test, y_test, metric='r2')
        >>> predictions = reg.predict(X_test)
    """

    def __init__(self):
        self.model = None

    def fit(
        self,
        X_train: pd.DataFrame or np.ndarray,
        y_train: pd.Series or np.ndarray,
        model_name: str = "Linear Regression",
        perform_tuning: bool = False,
    ):
        """
        Trains the specified regression model on the provided training data.

        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            The feature matrix for training.
        y_train : pd.Series or np.ndarray
            The target vector for training (true values).
        model_name : str, default='Linear Regression'
            The name of the algorithm to use. Options include:
            'Linear Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVR', 'ANN'.
        perform_tuning : bool, default=False
            If True, uses GridSearchCV to find the best hyperparameters from a predefined grid.
            If False, initializes the model with default parameters.

        Raises:
        -------
        ValueError
            If an unsupported model_name is provided.
        """
        # Trains the specified model on the training data, optionally performing hyperparameter tuning.

        if model_name == "Linear Regression":
            self.model = LinearRegression()
        elif model_name == "KNN":
            if perform_tuning:
                param_grid = {"n_neighbors": range(1, 21)}
                self.model = GridSearchCV(
                    KNeighborsRegressor(), param_grid, cv=3, scoring="r2"
                )
            else:
                self.model = KNeighborsRegressor()
        elif model_name == "Decision Tree":
            if perform_tuning:
                param_grid = {
                    "criterion": [
                        "squared_error",
                        "absolute_error",
                        "poisson",
                        "friedman_mse",
                    ]
                }
                self.model = GridSearchCV(
                    DecisionTreeRegressor(), param_grid, cv=3, scoring="r2"
                )
            else:
                self.model = DecisionTreeRegressor()
        elif model_name == "Random Forest":
            if perform_tuning:
                param_grid = {
                    "n_estimators": [10, 50, 100],
                    "criterion": ["squared_error", "absolute_error"],
                }
                self.model = GridSearchCV(
                    RandomForestRegressor(), param_grid, cv=3, scoring="r2"
                )
            else:
                self.model = RandomForestRegressor()
        elif model_name == "SVR":
            if perform_tuning:
                param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                self.model = GridSearchCV(SVR(), param_grid, cv=3, scoring="r2")
            else:
                self.model = SVR()
        elif model_name == "ANN":
            if perform_tuning:
                param_grid = {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "activation": ["relu", "tanh"],
                    "solver": ["adam", "sgd"],
                    "learning_rate": ["constant", "adaptive"],
                }
                self.model = GridSearchCV(
                    MLPRegressor(max_iter=500), param_grid, cv=3, scoring="r2"
                )
            else:
                self.model = MLPRegressor(max_iter=500)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        print(f"Training {model_name}...")
        self.model.fit(X_train, y_train)

        if perform_tuning and isinstance(self.model, GridSearchCV):
            print(f"Best parameters found: {self.model.best_params_}")
            self.model = self.model.best_estimator_

        print(f"Model '{model_name}' trained successfully.")

    def predict(self, X_test: pd.DataFrame or np.ndarray) -> np.ndarray:
        """
        Predicts values for the testing set.

        Parameters:
        X_test (pd.DataFrame or np.ndarray): Testing features.

        Returns:
        np.ndarray: Predicted values.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet. Please call fit() first.")

        return self.model.predict(X_test)

    def score(
        self,
        X: pd.DataFrame or np.ndarray,
        y: pd.Series or np.ndarray,
        metric: str = "r2",
    ) -> float:
        """
        Evaluates the model performance using the specified metric.

        Parameters:
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): True values.
        metric (str): Metric to calculate. Options: 'r2', 'mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error'.

        Returns:
        float: The calculated score.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet. Please call fit() first.")

        y_pred = self.model.predict(X)

        if metric == "r2" or metric == "r2_score":
            return r2_score(y, y_pred)
        elif metric == "mean_squared_error":
            return mean_squared_error(y, y_pred)
        elif metric == "root_mean_squared_error":
            return np.sqrt(mean_squared_error(y, y_pred))
        elif metric == "mean_absolute_error":
            return mean_absolute_error(y, y_pred)
        else:
            raise ValueError(f"Metric '{metric}' is not supported.")

    def plot_results(
        self,
        y_true: pd.Series or np.ndarray,
        y_pred: pd.Series or np.ndarray,
        model_name: str = "Model",
    ):
        """
        Plots the actual vs predicted values.

        Parameters:
        y_true (pd.Series or np.ndarray): True values.
        y_pred (pd.Series or np.ndarray): Predicted values.
        model_name (str): Name of the model.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)

        # Plot diagonal line for reference
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

        plt.xlabel("Actual Price (USD)")
        plt.ylabel("Predicted Price (USD)")
        plt.title(f"Actual vs Predicted Prices - {model_name}")
        plt.grid(True)

        # Save plot
        import os

        if not os.path.exists("plots"):
            os.makedirs("plots")

        plot_path = f'plots/{model_name.replace(" ", "_")}_results.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.show()
