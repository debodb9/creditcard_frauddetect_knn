import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import load_and_preprocess_data
from src.knn_model import train_knn, train_logistic_regression, evaluate_model
from src.visualize import plot_class_distribution, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.visualize import plot_class_distribution_before_after_smote
from imblearn.over_sampling import SMOTE

# Page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Sidebar: Theme and Model Settings
with st.sidebar:
    st.title("🧰 Settings")

    # Theme toggle
    theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)

    # K value for KNN
    k = st.slider("🔢 Select K (Neighbors)", min_value=1, max_value=20, value=5)

# Apply Dark Theme Styling (for basic contrast)
if theme == "Dark":
    st.markdown("""<style>
        .main { background-color: #1e1e1e; color: white; }
        .sidebar .sidebar-content { background-color: #262730; }
        .widget-label { color: white; }
        </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        .main { background-color: white; color: black; }
        .sidebar .sidebar-content { background-color: #f0f2f6; }
        .widget-label { color: black; }
        </style>""", unsafe_allow_html=True)

# Title Section
st.title("💳 Credit Card Fraud Detection")
st.markdown("Compare **KNN** and **Logistic Regression** for detecting fraudulent transactions.")

# Load dataset and preprocess data
df = pd.read_csv("data/creditcard.csv")

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Dataset", "🤖 Model Training", "📈 Evaluation"])

# Dataset Tab
with tab1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📉 Class Distribution")
    fig = plot_class_distribution(df)
    st.pyplot(fig)
    
# Model Training Tab
with tab2:
    st.subheader("🔄 Preprocessing Data")
    X_train, X_test, y_train, y_test, y_train_orig = load_and_preprocess_data("data/creditcard.csv")
    st.success(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Display side-by-side class distribution before and after SMOTE
    st.subheader("📊 Class Distribution (Before vs After SMOTE)")
    fig_smote_comparison = plot_class_distribution_before_after_smote(y_train_orig, y_train_resampled)
    st.pyplot(fig_smote_comparison)
    
    # Train models
    st.subheader("🤖 Training Models")

    with st.spinner("Training KNN..."):
        model_knn = train_knn(X_train_resampled, y_train_resampled, k=k)
    st.success(f"KNN trained (k={k})")

    with st.spinner("Training Logistic Regression..."):
        model_logreg = train_logistic_regression(X_train_resampled, y_train_resampled)
    st.success("Logistic Regression trained")

# Evaluation Tab
with tab3:
    st.subheader("📋 Evaluation Metrics")

    y_pred_knn = evaluate_model(model_knn, X_test, y_test)
    y_pred_logreg = evaluate_model(model_logreg, X_test, y_test)

    def get_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred)
        }

    metrics_knn = get_metrics(y_test, y_pred_knn)
    metrics_logreg = get_metrics(y_test, y_pred_logreg)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("KNN Metrics")
        for metric, value in metrics_knn.items():
            st.metric(label=metric, value=f"{value:.4f}")

    with col2:
        st.subheader("Logistic Regression Metrics")
        for metric, value in metrics_logreg.items():
            st.metric(label=metric, value=f"{value:.4f}")

    # Metric Comparison Chart
    st.subheader("📊 Metrics Comparison")
    comparison_df = pd.DataFrame({
        "KNN": metrics_knn,
        "Logistic Regression": metrics_logreg
    })
    st.bar_chart(comparison_df)

    # Confusion Matrices
    st.subheader("📌 Confusion Matrix - KNN")
    fig_knn = plot_confusion_matrix(y_test, y_pred_knn)
    st.pyplot(fig_knn)

    st.subheader("📌 Confusion Matrix - Logistic Regression")
    fig_logreg = plot_confusion_matrix(y_test, y_pred_logreg)
    st.pyplot(fig_logreg)

st.success("✅ All Done!")
