import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification Dashboard")

st.write(
    "Machine Learning Assignment 2 | M.Tech AIML"
)

st.markdown("""
This application evaluates five machine learning
classification models using the UCI Dry Bean dataset.
Upload your test CSV file to evaluate model performance.
""")

# ------------------------------------------------
# LOAD TRAINED MODELS
# ------------------------------------------------

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, filename in model_files.items():

        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing model file: {filepath}"
            )

        loaded_models[name] = joblib.load(filepath)

    return loaded_models

try:
    trained_models = load_models()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ------------------------------------------------
# DATASET UPLOAD
# ------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload test_data.csv to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Unable to read CSV: {error}")
    st.stop()

st.subheader("Uploaded Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.write("Dataset shape:", df.shape)

# ------------------------------------------------
# VALIDATE DATASET
# ------------------------------------------------

target_column = "Class"

if target_column not in df.columns:
    st.error("CSV must contain the target column 'Class'.")
    st.stop()

X_test = df.drop(columns=[target_column])
y_test = df[target_column]

if X_test.isnull().values.any() or y_test.isnull().any():
    st.error("Dataset contains missing values. Please clean it.")
    st.stop()

if len(df) == 0:
    st.error("Uploaded dataset is empty.")
    st.stop()

# Obtain expected feature names

first_model = next(iter(trained_models.values()))

expected_features = list(first_model.feature_names_in_)

missing_features = [
    col for col in expected_features
    if col not in X_test.columns
]

extra_features = [
    col for col in X_test.columns
    if col not in expected_features
]

if missing_features or extra_features:

    st.error("Uploaded CSV feature columns do not match training data.")

    if missing_features:
        st.write("Missing features:", missing_features)

    if extra_features:
        st.write("Unexpected features:", extra_features)

    st.stop()

# Ensure same feature order as training

X_test = X_test[expected_features]

if not all(
    pd.api.types.is_numeric_dtype(X_test[col])
    for col in X_test.columns
):
    st.error("All 16 input features must be numeric.")
    st.stop()

# ------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------

st.header("2. Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a machine learning model",
    list(trained_models.keys())
)

selected_model = trained_models[selected_model_name]

# ------------------------------------------------
# MODEL EVALUATION FUNCTION
# ------------------------------------------------

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    metrics = {}

    metrics["Accuracy"] = accuracy_score(
        y, predictions
    )

    metrics["AUC"] = roc_auc_score(
        y,
        probabilities,
        multi_class="ovr",
        average="macro",
        labels=model.classes_
    )

    metrics["Precision"] = precision_score(
        y, predictions,
        average="macro",
        zero_division=0
    )

    metrics["Recall"] = recall_score(
        y, predictions,
        average="macro",
        zero_division=0
    )

    metrics["F1 Score"] = f1_score(
        y, predictions,
        average="macro",
        zero_division=0
    )

    metrics["MCC"] = matthews_corrcoef(
        y, predictions
    )

    return predictions, metrics

# ------------------------------------------------
# EVALUATE SELECTED MODEL
# ------------------------------------------------

st.header("3. Model Evaluation")

try:

    y_pred, selected_metrics = evaluate_model(
        selected_model,
        X_test,
        y_test
    )

except Exception as error:
    st.error(f"Evaluation failed: {error}")
    st.stop()

# Display metrics

metric_columns = st.columns(3)

for index, (metric_name, metric_value) in enumerate(
    selected_metrics.items()
):

    with metric_columns[index % 3]:

        st.metric(
            metric_name,
            f"{metric_value:.4f}"
        )

st.subheader("Selected Model")

st.success(f"Selected model: {selected_model_name}")

# ------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------

st.header("4. Confusion Matrix")

class_labels = selected_model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(selected_model_name + " - Confusion Matrix")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------

st.header("5. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# ------------------------------------------------
# COMPARE ALL FIVE MODELS
# ------------------------------------------------

st.header("6. Comparison of All Five Models")

comparison_results = []

for name, model in trained_models.items():

    try:

        predictions, metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        comparison_results.append({
            "Model": name,
            **metrics
        })

    except Exception as error:

        st.warning(
            f"Could not evaluate {name}: {error}"
        )

comparison_df = pd.DataFrame(comparison_results)

st.dataframe(
    comparison_df.round(4),
    use_container_width=True
)

# ------------------------------------------------
# DOWNLOAD RESULTS
# ------------------------------------------------

st.header("7. Download Evaluation Results")

csv_data = comparison_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Model Comparison CSV",
    data=csv_data,
    file_name="model_comparison.csv",
    mime="text/csv"
)

st.success("Evaluation completed successfully!")

st.caption(
    "M.Tech AIML | Machine Learning Assignment 2 | UCI Dry Bean Dataset"
)
