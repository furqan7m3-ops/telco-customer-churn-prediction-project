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
# Load dataset
# -----------------------------

df = load_dataset("data/raw/telco_customer_churn.csv")
raw_data = mlflow.data.from_pandas(df,source='data/raw/telco_customer_churn.csv', name='telco-customer-churn')
# -----------------------------
# Model parameters
# -----------------------------

params = {
    "solver": "lbfgs",
    "max_iter": 250
}


# -----------------------------
# Start MLflow run
# -----------------------------

with mlflow.start_run():
    # -------------------------
    # Log input raw data
    # -------------------------
    mlflow.log_input(raw_data)


    # -------------------------
    #log model parameters
    # -------------------------
    mlflow.log_params(params)

    # -------------------------
    # Preprocess dataset
    # -------------------------

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # -------------------------
    # Train model
    # -------------------------

    model = LogisticRegression(**params)

    model.fit(X_train, y_train)
    #-------------------------
    # Log model
    #-------------------------
    mlflow.sklearn.log_model(model, "model")


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

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    #--------------------------
    #Log evaluation metrics
    #--------------------------
    mlflow.log_metrics({
        "training_accuracy": accuracy,
        "training_precision": precision,
        "training_recall": recall,
        "training_f1_score": f1
    })