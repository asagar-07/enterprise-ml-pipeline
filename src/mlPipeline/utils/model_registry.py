from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier
   # "xgboost": XGBClassifier
}