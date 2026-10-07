from preprocess import load_dataset, preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import mlflow
import mlflow.sklearn


# -----------------------------
# MLflow experiment
# -----------------------------

mlflow.set_experiment("Telco Customer Churn Prediction")


# -----------------------------
# Enable MLflow autologging
# -----------------------------

mlflow.sklearn.autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True
)


# -----------------------------
# Load dataset
# -----------------------------

df = load_dataset("data/raw/telco_customer_churn.csv")


# -----------------------------
# Model parameters
# -----------------------------

params = {
    "solver": "lbfgs",
    "max_iter": 500
}


# -----------------------------
# Start MLflow run
# -----------------------------

with mlflow.start_run():

    # -------------------------
    # Preprocess dataset
    # -------------------------

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # -------------------------
    # Train model
    # -------------------------

    model = LogisticRegression(**params)

    model.fit(X_train, y_train)

    # -------------------------
    # Predictions
    # -------------------------

    y_pred = model.predict(X_test)

    # -------------------------
    # Evaluation metrics
    # -------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")