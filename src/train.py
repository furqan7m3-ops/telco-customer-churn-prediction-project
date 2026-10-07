from src.preprocess import load_dataset, preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt


# -----------------------------
# MLflow experiment
# -----------------------------

mlflow.set_experiment("Telco Customer Churn Prediction Experiment")


# -----------------------------
# Load dataset
# -----------------------------

df = load_dataset("data/raw/telco_customer_churn.csv")


# -----------------------------
# Model parameters
# -----------------------------

params = {
    "solver": "lbfgs",
    "max_iter": 300
}


# -----------------------------
# Start MLflow run
# -----------------------------

with mlflow.start_run():

    # -------------------------
    # Log parameters
    # -------------------------

    mlflow.log_params(params)

    # -------------------------
    # Log dataset
    # -------------------------

    dataset = mlflow.data.from_pandas(
        df,
        source="data/raw/telco_customer_churn.csv",
        name="telco-customer-churn"
    )

    mlflow.log_input(
        dataset,
        context="training"
    )

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
    # Calculate metrics
    # -------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # -------------------------
    # Log metrics
    # -------------------------

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # -------------------------
    # Confusion matrix
    # -------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # -------------------------
    # Log trained model
    # -------------------------

    mlflow.sklearn.log_model(
        model,
        name="model"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")