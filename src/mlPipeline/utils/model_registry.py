from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "xgboost": XGBClassifier
}