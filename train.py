import os
import traceback
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

try:
    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("iris-classifier")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv("data/iris.csv")

    # ✅ Use ALL features this time
    X = df.drop(["species"], axis=1)
    y = df["species"]

    # ✅ Use full training split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200, C=1.0, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 200)
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"Run ID  : {run_id}")
        print(f"Accuracy: {accuracy:.4f}")

    # ── Export Run ID ──────────────────────────────────────────────────────────
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    print("model_info.txt written.")

except Exception:
    with open("error_logs.txt", "w") as f:
        f.write("Training failed with the following error:\n\n")
        f.write(traceback.format_exc())
    print("error_logs.txt written.")
    raise
