import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.knn_model import train_knn, evaluate_model
from src.visualize import plot_class_distribution, plot_confusion_matrix, plot_metrics

def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
    
    # Title and description
    st.title("💳 Credit Card Fraud Detection")
    st.write("This app lets you compare **KNN** and **Logistic Regression** to detect fraudulent credit card transactions.")

    # Sidebar Theme Toggle
    theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

    if theme == "Dark":
        st.markdown("""
            <style>
            .main { background-color: #1e1e1e; color: white; }
            .sidebar .sidebar-content { background-color: #262730; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .main { background-color: white; color: black; }
            .sidebar .sidebar-content { background-color: #f0f2f6; }
            </style>
        """, unsafe_allow_html=True)

    # Sidebar Inputs
    with st.sidebar:
        st.header("⚙️ Model Settings")
        k = st.slider("K for KNN", 1, 20, 5)

    # Load data
    st.subheader("📊 Dataset Preview")
    df = pd.read_csv("data/creditcard.csv")
    st.write(df.head())

    # Plot Class Distribution
    st.subheader("📉 Class Distribution")
    fig1 = plot_class_distribution(df)
    st.pyplot(fig1)

    # Preprocessing Data
    st.subheader("🔄 Preprocessing Data")
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")
    st.success(f"Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # Train KNN Model
    st.subheader("🤖 Training KNN Model")
    with st.spinner("Training KNN model..."):
        model_knn = train_knn(X_train, y_train, k=k)
    st.success(f"KNN (k={k}) model trained!")

    # Evaluate Model
    st.subheader("📈 Model Evaluation")
    y_pred_knn = evaluate_model(model_knn, X_test, y_test)

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    fig_knn = plot_confusion_matrix(y_test, y_pred_knn)
    st.pyplot(fig_knn)

    # Performance Metrics
    st.subheader("📋 Performance Metrics")
    fig_metrics = plot_metrics(y_test, y_pred_knn)
    st.pyplot(fig_metrics)

    st.success("✅ All Done!")

if __name__ == "__main__":
    main()

