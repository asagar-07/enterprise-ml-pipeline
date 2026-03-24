from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os
import mlflow
import mlflow.pyfunc


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "xgboost": XGBClassifier
}


def load_champion_model(model_name: str = "creditcard_fraud_model"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Registry URI:", mlflow.get_registry_uri())

    model_uri = f"models:/{model_name}@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    return model