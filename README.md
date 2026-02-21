# Diamonds Predictor Application

This repository contains an end-to-end Machine Learning Framework designed to predict the price and cut quality of diamonds based on their physical attributes. 

## What problem is this solving?
Evaluating diamonds is traditionally subjective and complex, relying on the "4 Cs" (Carat, Cut, Color, Clarity) alongside physical dimensions (length, width, depth). This project solves that problem by providing a robust, automated framework that digests raw diamond features and employs multiple machine learning pipelines to:
1. **Classify** a diamond's `cut_quality` (Fair, Good, Very Good, Premium, Ideal).
2. **Cluster** diamonds into unsupervised groups based strictly on their physical similarities.
3. **Regress** (predict) the exact market `price_usd` of a diamond based on all engineered features.

## Prerequisites & Installation

This project is built using Python (3.11 - 3.12) and uses **UV** for fast dependency management.

### Required Libraries
The core dependencies include:
- `pandas` (Data manipulation)
- `numpy` (Numerical operations)
- `scikit-learn` (Machine learning models and metrics)
- `matplotlib` & `seaborn` (Data visualization)

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/ml-framework-project-new.git
   cd ml-framework-project-new
   ```

2. **Install dependencies using UV (or pip):**
   ```bash
   uv sync
   
   # Or alternatively, using pip with the pyproject.toml:
   pip install .
   ```

3. **Run the Application:**
   ```bash
   python -m ml_framework_project.main
   ```

## Project Structure & Documentation

The framework is strictly divided into specialized modules.

### 1. `data_analyzer`
Responsible for data ingestion, cleaning, and feature engineering.
* **`data_reader.py`**: Reads the raw CSV input.
* **`data_preprocessing.py`**: Handles NaN values, renames columns, and engineers new features (`volume`, `density`).
* **`encoder.py` & `scaler.py`**: Transforms categorical strings into integers (Label Encoding) and scales continuous prices (Z-Score & MinMax).

### 2. `models`
Contains modular, reusable Object-Oriented implementations of Scikit-Learn algorithms. Each class exposes a generic `.fit()`, `.predict()`, and `.score()` standard interface.

#### `classifier.py`
* **Input**: Training data (`X_train`, `y_train`) and a dynamic string `model_name` (e.g., 'KNN', 'Decision Tree', 'Random Forest', 'SVM', 'ANN').
* **Behavior**: Uses `GridSearchCV` to tune hyperparameters automatically.
* **Output**: A fully trained classification model, with scoring mapped to "Accuracy" and optional confusion matrix plotting.

#### `regressor.py`
* **Input**: Training data (`X_train`, `y_train`) and a `model_name` (e.g., 'Linear Regression', 'KNN', 'ANN').
* **Behavior**: Trains against continuous targets (like `price_usd`).
* **Output**: A trained regression model scored by default using $R^2$. Includes a method to generate a 2D scatter plot comparing `Actual vs Predicted` values.

#### `clustering.py`
* **Input**: Just features (`X`) without targets, aiming to discover patterns. Supports 'k-Means', 'Agglomerative Hierarchical Clustering', and 'Mean Shift Clustering'.
* **Behavior**: Trains the model and assigns cluster labels to the dataset. Automatically limits intense $O(N^2)$ calculations on large datasets for stability.
* **Output**: The clustered model, evaluated using `Silhouette Score`, and capable of rendering PCA-reduced 2D visualizations natively.

### 3. `main.py`
The orchestrator script. It sequentially groups the modules to automatically fetch data, preprocess it, and run the Classification, Clustering, and Regression pipelines back-to-back, dynamically reporting the "winning" algorithm for each discipline.

## How Others Can Improve This Code
If you're looking to contribute to or fork this project, here are some excellent areas for improvement:
1. **Data Expansion:** Update `data_reader.py` to pull live data from an external Diamond Pricing API rather than a static CSV.
2. **Deep Learning:** Enhance the `ANN` configurations within the model files to use TensorFlow or PyTorch for deeper, more reliable layer tuning.
3. **Web Interface:** Wrap `main.py` in a FastAPI or Streamlit frontend so users can input a new diamond's stats manually and receive an instant price/cut prediction UI.
