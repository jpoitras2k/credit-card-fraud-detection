# Credit Card Fraud Detection Project

This repository contains an end-to-end Machine Learning Framework designed to detect fraudulent credit card transactions. 

It was specifically built to fulfill the **RoboGarden / UpWork Machine Learning Bootcamp** requirements through a structured 6-phase approach ranging from Exploratory Data Analysis (EDA) to Deep Learning evaluation.

## Problem Description
Credit card fraud detection is a critical classification problem. The challenge is the extreme class imbalance: out of hundreds of thousands of transactions, only a tiny fraction are fraudulent.

**Dataset:** The dataset consists of 284,807 credit card transactions made by European cardholders in September 2013.
- **Link:** [Kaggle Source](https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv)
- **Features:** `Time` (seconds since first transaction), `V1-V28` (PCA transformed features due to confidentiality), and `Amount` (transaction amount).
- **Target:** `Class` (1 for Fraud, 0 for Genuine).

## Technologies Mastered
- **Languages:** Python (Object-Oriented Programming)
- **Data Science:** Numpy, Pandas, Matplotlib, Seaborn
- **Machine Learning (Scikit-Learn):** Linear & Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, K-Means Clustering.
- **Deep Learning (Keras/TensorFlow):** Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN).

## Bootcamp Implementation Phases

### Phase 1: Data Analysis and Preparation
- Conducted Exploratory Data Analysis (`exploration.ipynb`) to visualize the heavy class imbalance, transaction distributions, and correlation matrix.
- Preprocessed the raw data by applying statistical scaling to the `Time` and `Amount` fields prior to model ingestion.

### Phase 2 & 3: Model Selection, Design, and Parameter Tuning
- Programmed a modular `Classifier` class that supports dynamic routing between Classical ML (Scikit-Learn) and Deep Learning (Keras/TensorFlow) networks.
- Integrated `GridSearchCV` for automated hyperparameter tuning on classical models.
- Constructed robust Keras Neural Networks utilizing `Dropout` layers to prevent overfitting and `EarlyStopping` callbacks.

### Phase 4 & 5: Model Training, Evaluation, and Comparison
- Trained the suite of models against the imbalanced data. 
- Due to the nature of the dataset, evaluation aggressively prioritized **Precision-Recall AUC (PR-AUC)** and **F1-Scores** over standard accuracy.
- Generated an automated Seaborn Bar Chart plotting the comparative performance metrics across all models upon pipeline completion.

### Phase 6: Interactive UX
- Wrapped the execution pipeline in an interactive Command Line Interface (CLI) menu, allowing users to select subsets of models (e.g., exclusively Deep Learning or exclusively Classical models) to save training time during review.

## Setup Instructions

This project is built using Python 3.11+ and uses **UV** for fast dependency management.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jpoitras2k/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies using UV:**
   ```bash
   uv sync
   ```

3. **Run the Interactive Application:**
   ```bash
   uv run python -m ml_framework_project.main
   ```
   *Follow the on-screen terminal prompts to select which models to train!*

## Running in Docker (Production)
To ensure complete reproducibility and isolation, this project is fully Dockerized.
1. Make sure [Docker Desktop](https://www.docker.com/) is installed and running.
2. Build and run the interactive container:
   ```bash
   docker compose run fraud-detection-app
   ```
This will open the CLI menu directly in your terminal. Because the volumes are mapped in `docker-compose.yml`, any `.png` graphs or `.pkl`/`.keras` saved models generated inside the container will automatically appear in your local Windows folder!
