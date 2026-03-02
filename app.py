import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from ml_framework_project.datasets.credit_card import get_creditcard_data
from ml_framework_project.data_analyzer.data_preprocessing import preprocess_creditcard_data

# --- Config ---
st.set_page_config(page_title="Credit Fraud Detection AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- Styling Aesthetics ---
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .big-font {
        font-family: 'Inter', sans-serif;
        font-size: 3rem !important;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #1E2127;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4ECDC4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45B7AF;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_and_prep_data():
    df = get_creditcard_data()
    if df.empty:
        return None
    return preprocess_creditcard_data(df)

@st.cache_resource
def load_trained_models():
    models_dir = os.path.join(os.getcwd(), "saved_models")
    loaded_models = {}
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('.pkl'):
                if f.startswith('best_model_'): continue
                name = f.replace("model_", "").replace(".pkl", "").replace("_", " ").title()
                loaded_models[name] = {"path": os.path.join(models_dir, f), "type": "sklearn"}
            elif f.endswith('.keras'):
                if f.startswith('best_model_'): continue
                name = f.replace("model_", "").replace(".keras", "").replace("_", " ").title()
                loaded_models[name] = {"path": os.path.join(models_dir, f), "type": "keras"}
    return loaded_models

# --- Main App ---
st.markdown('<p class="big-font">🛡️ Credit Fraud Detection Engine</p>', unsafe_allow_html=True)
st.write("A comprehensive machine learning pipeline deployed to intercept and analyze anomalous financial transactions.")

# --- Sidebar ---
st.sidebar.title("Navigation 🧭")
app_mode = st.sidebar.radio("Select Mode", [
    "Data Overview & EDA", 
    "Model Training Workbench",
    "Model Performance & Comparison",
    "Model Evaluation & Inference"
])

# Load Data
with st.spinner("Loading Dataset..."):
    df = load_and_prep_data()

if df is None:
    st.error("Failed to load dataset. Please ensure `creditcard.csv` exists in the datasets directory.")
    st.stop()

if app_mode == "Data Overview & EDA":
    st.header("📊 Dataset Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Total Transactions</h3><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with col2:
        fraud_count = len(df[df['Class'] == 1])
        pct_fraud = (fraud_count / len(df)) * 100
        st.markdown(f'<div class="metric-card" style="border-left-color: #FF6B6B;"><h3>Fraudulent</h3><h2>{fraud_count:,} <span style="font-size: 1rem; color: #FF6B6B;">({pct_fraud:.2f}%)</span></h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #FFD166;"><h3>Features</h3><h2>{df.shape[1] - 1}</h2></div>', unsafe_allow_html=True)

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Class Distribution (Highly Imbalanced)")
    fig = px.pie(
        names=["Legitimate", "Fraud"],
        values=[len(df[df['Class'] == 0]), len(df[df['Class'] == 1])],
        hole=0.4,
        color_discrete_sequence=["#4ECDC4", "#FF6B6B"]
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)


elif app_mode == "Model Training Workbench":
    st.header("⚙️ Model Training Workbench")
    st.write("Select a model suite to train directly from the dashboard. Progress and logs will output to your local IDE terminal.")
    
    suite = st.selectbox("Select Training Suite", [
        "1. Run ALL Models (Classical + Keras Deep Learning + Clustering)",
        "2. Run ONLY Fast Classical Models",
        "3. Run ONLY Keras Deep Learning Models",
        "4. Run a Specific Custom Model",
        "5. Run ONLY Unsupervised Clustering (K-Means)"
    ])
    
    all_models = [
        "Linear Regression", "Logistic Regression", "KNN", "Decision Tree", 
        "Random Forest", "Keras ANN", "Keras CNN", "Keras RNN"
    ]
    classical_models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    keras_models = ["Keras ANN", "Keras CNN", "Keras RNN"]
    
    custom_model = None
    if "4." in suite:
        custom_model = st.selectbox("Select Custom Model", all_models)
        
    if st.button("Start Training Pipeline", type="primary"):
        st.cache_resource.clear() # Clear cache so new models load
        
        with st.spinner("Training models in the background... Please check your terminal console for live training logs and progress bars."):
            from ml_framework_project.main import run_project_pipeline
            
            try:
                if "1." in suite:
                    run_project_pipeline(df, all_models, run_clustering=True)
                elif "2." in suite:
                    run_project_pipeline(df, classical_models, run_clustering=False)
                elif "3." in suite:
                    run_project_pipeline(df, keras_models, run_clustering=False)
                elif "4." in suite:
                    run_project_pipeline(df, [custom_model], run_clustering=False)
                elif "5." in suite:
                    run_project_pipeline(df, [], run_clustering=True)
                
                st.success("Training Pipeline Completed! Check the 'Model Performance & Comparison' and 'Model Evaluation & Inference' tabs to interact with your new models.")
            except Exception as e:
                st.error(f"Error during training pipeline: {e}")

elif app_mode == "Model Performance & Comparison":
    st.header("📈 Model Performance Comparison")
    st.write("This section displays the evaluation metrics (Accuracy, F1-Score, PR-AUC) for all trained models. Run the Training Pipeline to generate the latest results.")
    
    # Path where main.py saves the plots
    plots_dir = os.path.join(os.getcwd(), "plots")
    comparison_plot_path = os.path.join(plots_dir, "model_comparison.png")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if os.path.exists(comparison_plot_path):
            st.image(comparison_plot_path, caption="Comparative Performance of Trained Models", use_container_width=True)
            st.success("Performance metrics successfully loaded from the latest pipeline run.")
        else:
            st.info("No comparison results found! Head over to the 'Model Training Workbench' to run the pipeline and generate performance reports.")
            
    st.divider()
    
    st.subheader("Unsupervised Learning: K-Means Clustering")
    kmeans_plot_path = os.path.join(plots_dir, "k-Means_clustering_results.png")
    if os.path.exists(kmeans_plot_path):
        from PIL import Image
        img = Image.open(kmeans_plot_path)
        st.image(img, caption="PCA 2D Projection of K-Means Clusters", use_container_width=False)
        st.write("The plot above shows the application of Unsupervised Learning (K-Means) to identify clusters of transactions without relying on 'Class' labels.")
        
        with st.expander("💡 How to read this graph (Presentation Info)"):
            st.markdown("""
            *   **What are the dots?** Each dot represents a single credit card transaction.
            *   **Why 'PCA Component'?** The raw dataset has 30 different mathematical features. It is impossible to draw a 30-dimensional graph. **P**rincipal **C**omponent **A**nalysis (PCA) was used to mathematically compress those 30 dimensions down to just 2 dimensions (the X and Y axes) so they can be visualized on a screen.
            *   **What do the colors mean?** The colors represent the mathematically distinct groups (clusters) that the Unsupervised K-Means algorithm discovered on its own. The algorithm grouped these transactions strictly based on their behavior, **without** ever being told whether they were fraud or genuine.
            """)
    else:
        st.write("Run the clustering pipeline from the Workbench to generate this visualization.")


elif app_mode == "Model Evaluation & Inference":
    st.header("🤖 AI Threat Detection")
    models = load_trained_models()
    
    if not models:
        st.warning("No trained models found! Please run the `main.py` pipeline locally to generate and save models into the `saved_models` directory before using the visualizer.")
    else:
        st.write("Select an active model from the repository to perform inference on a randomized sample of data.")
        
        selected_model_name = st.selectbox("Select Active Model", list(models.keys()))
        
        if st.button("Run Inference Simulator"):
            with st.spinner(f"Waking up {selected_model_name}..."):
                # Load chosen model
                model_info = models[selected_model_name]
                try:
                    loaded_clf = None
                    if model_info["type"] == "sklearn":
                         loaded_clf = joblib.load(model_info["path"])
                    else:
                         loaded_clf = tf.keras.models.load_model(model_info["path"])
                         
                    # To prevent 100% accuracy due to data leakage, we must simulate testing on *unseen* data
                    # We mimic the 80/20 train/test split from the main pipeline and only sample from the test set
                    from sklearn.model_selection import train_test_split
                    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    _, unseen_df = train_test_split(df_shuffled, test_size=0.2, random_state=42, stratify=df_shuffled['Class'])
                    
                    sample_size = min(500, len(unseen_df))
                    fraud_sample = unseen_df[unseen_df['Class'] == 1].sample(min(10, len(unseen_df[unseen_df['Class'] == 1])), replace=True)
                    legit_sample = unseen_df[unseen_df['Class'] == 0].sample(sample_size - len(fraud_sample))
                    test_df = pd.concat([fraud_sample, legit_sample]).sample(frac=1).reset_index(drop=True)
                    
                    X_test = test_df.drop(columns=["Class"])
                    y_test = test_df["Class"]
                    
                    st.success("Model loaded successfully!")
                    
                    # Predict
                    if model_info["type"] == "sklearn":
                         if hasattr(loaded_clf, "predict_proba"):
                             preds = loaded_clf.predict(X_test)
                             probs = loaded_clf.predict_proba(X_test)[:, 1]
                         else:
                             # For Linear Regression, predict() returns continuous values
                             probs = loaded_clf.predict(X_test)
                             preds = (probs > 0.5).astype(int) 
                    else:
                         # Keras
                         probs = loaded_clf.predict(X_test.values, verbose=0).flatten()
                         preds = (probs > 0.5).astype(int)
                         
                    # Calculate live basic metrics on sample
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, zero_division=0)
                    rec = recall_score(y_test, preds, zero_division=0)
                    f1 = f1_score(y_test, preds, zero_division=0)
                    
                    st.markdown("### Live Sample Simulation Results")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{acc:.2%}")
                    c2.metric("Precision", f"{prec:.2%}")
                    c3.metric("Recall", f"{rec:.2%}")
                    c4.metric("F1 Score", f"{f1:.2%}")
                    
                    # Simulated alerts
                    st.markdown("### Simulated Transaction Alerts (Top 10 High Risk)")
                    test_df['Risk Probability'] = probs
                    test_df['Model Prediction'] = preds
                    test_df['Actual Truth'] = y_test
                    
                    risk_df = test_df.sort_values(by="Risk Probability", ascending=False).head(10)
                    
                    def highlight_fraud(row):
                        if row['Actual Truth'] == 1 and row['Model Prediction'] == 1:
                            return ['background-color: rgba(78, 205, 196, 0.2)'] * len(row) # True Positive (Green)
                        elif row['Actual Truth'] == 0 and row['Model Prediction'] == 1:
                            return ['background-color: rgba(255, 209, 102, 0.2)'] * len(row) # False Positive (Yellow)
                        elif row['Actual Truth'] == 1 and row['Model Prediction'] == 0:
                            return ['background-color: rgba(255, 107, 107, 0.2)'] * len(row) # False Negative (Red)
                        return [''] * len(row)
                    
                    st.dataframe(risk_df[['Risk Probability', 'Model Prediction', 'Actual Truth']].style.apply(highlight_fraud, axis=1), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error executing inference: {e}")
