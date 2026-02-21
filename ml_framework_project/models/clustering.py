import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


class Clustering:
    """
    A generic Clustering class to train and evaluate various clustering models.

    Supported Models:
    - k-Means
    - Agglomerative Hierarchal Clustering
    - Mean Shift Clustering

    Usage:
        >>> clu = Clustering()
        >>> clu.fit(X_train, model_name='k-Means')
        >>> labels = clu.predict(X_test)
        >>> score = clu.score(X_test, labels, metric='silhouette')
    """

    def __init__(self):
        self.model = None
        self.model_name = None

    def fit(
        self, X_train: pd.DataFrame or np.ndarray, model_name: str = "k-Means", **kwargs
    ):
        """
        Trains the specified clustering model on the provided training data.

        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            The feature matrix for training.
        model_name : str, default='k-Means'
            The name of the algorithm to use. Options include:
            'k-Means', 'Agglomerative Hierarchal Clustering', 'Mean Shift Clustering'.
        **kwargs : Additional keyword arguments to pass to the model initialization.

        Raises:
        -------
        ValueError
            If an unsupported model_name is provided.
        """
        self.model_name = model_name

        if model_name == "k-Means":
            # Setting default n_init to 'auto' to avoid future warnings, if not provided
            if "n_init" not in kwargs:
                kwargs["n_init"] = "auto"
            self.model = KMeans(**kwargs)
        elif (
            model_name == "Agglomerative Hierarchal Clustering"
            or model_name == "Agglomerative Hierarchical Clustering"
        ):
            self.model = AgglomerativeClustering(**kwargs)
        elif model_name == "Mean Shift Clustering":
            self.model = MeanShift(**kwargs)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        print(f"Training {self.model_name}...")
        self.model.fit(X_train)
        print(f"Model '{self.model_name}' trained successfully.")

    def predict(self, X_test: pd.DataFrame or np.ndarray) -> np.ndarray:
        """
        Predicts cluster labels for the testing set.
        Note: Some models like Agglomerative Hierarchal Clustering do not support
        predicting on new data after training.

        Parameters:
        X_test (pd.DataFrame or np.ndarray): Testing features.

        Returns:
        np.ndarray: Predicted cluster labels.
        """
        if self.model is None:
            raise Exception("Model has not been trained yet. Please call fit() first.")

        if hasattr(self.model, "predict"):
            return self.model.predict(X_test)
        elif hasattr(self.model, "fit_predict"):
            print(
                f"Warning: Model '{self.model_name}' does not have a predict method for new data. Using fit_predict instead."
            )
            return self.model.fit_predict(X_test)
        else:
            raise AttributeError("The selected model does not support prediction.")

    def score(
        self,
        X: pd.DataFrame or np.ndarray,
        labels: np.ndarray = None,
        metric: str = "silhouette",
    ) -> float:
        """
        Evaluates the clustering performance using the specified metric.

        Parameters:
        X (pd.DataFrame or np.ndarray): Features.
        labels (np.ndarray, optional): Cluster labels. If None, the labels from the trained model are used.
        metric (str): Metric to calculate. Options: 'silhouette', 'calinski_harabasz', 'davies_bouldin'.

        Returns:
        float: The calculated score.
        """
        if self.model is None and labels is None:
            raise Exception(
                "Model has not been trained yet. Please call fit() first or provide labels."
            )

        # If no labels are provided, try to use the labels from the trained model
        if labels is None:
            if hasattr(self.model, "labels_"):
                labels = self.model.labels_
                # If evaluating on data that doesn't match the training data size, we must predict
                if len(labels) != len(X):
                    labels = self.predict(X)
            else:
                labels = self.predict(X)

        # Check if there's only 1 cluster, metrics generally require at least 2 clusters
        if len(set(labels)) <= 1:
            print(
                "Warning: Only one cluster found. Metrics require at least 2 clusters. Returning 0.0"
            )
            return 0.0

        if metric == "silhouette":
            return silhouette_score(X, labels)
        elif metric == "calinski_harabasz":
            return calinski_harabasz_score(X, labels)
        elif metric == "davies_bouldin":
            return davies_bouldin_score(X, labels)
        else:
            raise ValueError(f"Metric '{metric}' is not supported.")

    def plot_results(
        self,
        X: pd.DataFrame or np.ndarray,
        labels: np.ndarray,
        model_name: str = "Clustering Model",
    ):
        """
        Plots the clustering results using PCA to reduce dimensions to 2D for visualization.

        Parameters:
        X (pd.DataFrame or np.ndarray): Features.
        labels (np.ndarray): Cluster labels.
        model_name (str): Name of the model for the plot title.
        """
        import os
        from sklearn.decomposition import PCA

        print("Reducing dimensions to 2D using PCA for visualization...")
        # Reduce dimensions to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(10, 6))
        # Handle cases where labels are continuous vs categorical gracefully
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=labels,
            palette="viridis",
            legend="full",
            alpha=0.6,
        )

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(f"Clustering Results (PCA 2D Projection) - {model_name}")
        plt.grid(True)

        # Save plot
        if not os.path.exists("plots"):
            os.makedirs("plots")

        plot_path = f'plots/{model_name.replace(" ", "_")}_clustering_results.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Attempt non-blocking show to keep pipeline moving or short pause
        plt.show(block=False)
        plt.pause(3)
        plt.close()
