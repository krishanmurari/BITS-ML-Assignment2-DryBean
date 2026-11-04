import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Configure the Streamlit page metadata and overall app layout.
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide"
)

# App heading and course context for the assignment.
st.title("🫘 Dry Bean Classification")
st.write("BITS WILP M.Tech AIML - Assignment 2")

# Mapping of model names to the saved joblib artifacts in the models directory.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Load the expected feature order used during training so uploaded data matches the model inputs.
with open("models/feature_names.json") as f:
    features = json.load(f)

st.sidebar.header("Model Configuration")

# Allow the user to choose which trained dry-bean classifier to evaluate.
selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

# Upload a CSV file containing the same feature columns used during training.
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload test_data.csv to evaluate the model.")
    st.stop()

data = pd.read_csv(uploaded_file)

# Validate the uploaded dataset before inference so the app fails gracefully on schema issues.
missing = [col for col in features if col not in data.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Restrict input to the feature columns expected by the selected trained classifier.
X_input = data[features]

# Load the chosen trained model from disk.
model_path = os.path.join(
    "models",
    MODEL_FILES[selected_model]
)

model = joblib.load(model_path)

# Generate predictions for the uploaded dataset.
predictions = model.predict(X_input)

st.subheader("Predictions")

# Keep a copy of the original data and append model predictions for download and review.
output = data.copy()
output["Predicted_Class"] = predictions

st.dataframe(output.head(20), use_container_width=True)

st.download_button(
    "Download Predictions",
    output.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv"
)

if "Class" in data.columns:

    # Real labels are required to compute supervised evaluation metrics.
    y_true = data["Class"]
    probabilities = model.predict_proba(X_input)

    # Aggregate the key classification metrics for comparison and reporting.
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(
            y_true,
            probabilities,
            labels=model.classes_,
            multi_class="ovr",
            average="weighted"
        ),
        "Precision": precision_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, predictions)
    }

    st.subheader("Evaluation Metrics")

    cols = st.columns(3)

    for i, (name, value) in enumerate(metrics.items()):
        cols[i % 3].metric(name, f"{value:.4f}")

    st.subheader("Confusion Matrix")

    # Confusion matrix shows how often each class is confused with other bean types.
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    st.pyplot(fig)

    st.subheader("Classification Report")

    # Print a class-wise summary with precision, recall, F1-score, and support.
    st.text(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

else:
    st.warning(
        "Target column Class is missing. "
        "Predictions are available, but evaluation metrics "
        "cannot be calculated."
    )