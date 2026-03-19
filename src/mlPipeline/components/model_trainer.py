from pathlib import Path
import joblib
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

from mlPipeline.constants import PARAMS_FILE_PATH
from mlPipeline.utils.common import read_yaml, save_json
from mlPipeline.utils.model_registry import MODEL_REGISTRY
from mlPipeline.entity.config_entity import ModelTrainingConfig
from mlPipeline import logger



class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.target_col_name = self.params.target_column

        self.mlflow_tracking_uri = self._safe_get(self.params, ["mlflow", "tracking_uri"], default="sqlite:///mlflow.db")
        self.mlflow_experiment_name = self._safe_get(self.params, ["mlflow", "experiment_name"], default="Creditcard Fraud Detection")
        self.registered_model_name = self._safe_get(self.params, ["model_registry", "registered_model_name"], default="creditcard_fraud_model",)

        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        self.config.trained_model_dir.mkdir(parents=True, exist_ok=True)

    def _safe_get(self, obj: Any, keys: list[str], default: Any = None) -> Any:
        current = obj
        for key in keys:
            if current is None:
                return default

            if isinstance(current, dict):
                current = current.get(key, None)
            else:
                current = getattr(current, key, None)

        return default if current is None else current

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_parquet(self.config.train_data_path)
        val_df = pd.read_parquet(self.config.val_data_path)
        return train_df, val_df


    def _prepare_features_and_target(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = dataset.drop(columns=[self.target_col_name])
        y = dataset[self.target_col_name]
        return X, y


    def _build_models(self) -> dict:
        models = self.params.model_training.models
        random_state = self.params.model_training.random_state
        initialized_models = {}

        for model_name, hyperparameters in models.items():
            if model_name not in MODEL_REGISTRY:
                raise ValueError(f"Unsupported model: {model_name}")
            model_class = MODEL_REGISTRY[model_name]
            model_instance = model_class(random_state=random_state, **hyperparameters) 
            initialized_models[model_name] = model_instance 
        return initialized_models
    
    def _evaluate_on_validation(self, model, X_val, y_val, model_name: str) -> dict:
        y_pred = model.predict(X_val)
        
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None,
        }

        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        cm_path = self.config.root_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(cm_path))

        return metrics


    def train_and_track_models(self) -> dict[str, dict]:
        training_results = {}

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        with mlflow.start_run(run_name = "model_training_parent") as parent_run:
            mlflow.set_tag("stage", "model_training")
            mlflow.set_tag("run_type", "training")
            mlflow.log_param("target_column", self.target_col_name)
            mlflow.log_param("train_data_path", str(self.config.train_data_path))
            mlflow.log_param("val_data_path", str(self.config.val_data_path))

            logger.info(f"Parent Run ID: {parent_run.info.run_id}")

            train_df, val_df = self._load_data()
            X_train, y_train = self._prepare_features_and_target(train_df)
            X_val, y_val = self._prepare_features_and_target(val_df)
            models = self._build_models()
            
            for model_name, model in models.items():
                with mlflow.start_run(run_name = model_name, nested=True) as child_run:
                    logger.info(f"Child Run ID: {child_run.info.run_id}")

                    model.fit(X_train, y_train)

                    metrics = self._evaluate_on_validation(model=model, X_val=X_val, y_val=y_val, model_name=model_name)
                    mlflow.log_param("model_name", model_name)

                    for param_name, param_value in model.get_params().items():
                        mlflow.log_param(param_name, param_value)
                    
                    for metric_name, metric_value in metrics.items():
                        if metric_value is not None:
                            mlflow.log_metric(metric_name, metric_value)

                    model_file_path = self.config.trained_model_dir/ f"{model_name}.joblib"
                    joblib.dump(model, model_file_path)
                    mlflow.sklearn.log_model(sk_model=model, artifact_path = model_name) #name=model_name)
                    
                    training_results[model_name] = {
                        "model": model,
                        "metrics": metrics,
                        "model_file_path": model_file_path,
                        "mlflow_run_id": child_run.info.run_id,
                        "mlflow_model_artifact_path": model_name,
                    }

        return training_results
   

    def select_best_model(self, training_results:dict) -> dict:
        best_model_criteria_rank = ["f1_score", "recall", "roc_auc"]

        best_model_name = max (training_results, key= lambda name: (training_results[name]["metrics"][best_model_criteria_rank[0]],
                                                                    training_results[name]["metrics"][best_model_criteria_rank[1]],
                                                                    training_results[name]["metrics"][best_model_criteria_rank[2]]
                                                                    if training_results[name]["metrics"][best_model_criteria_rank[2]] is not None
                                                                    else float("-inf"),
                                                                    ),
                                                                )
        
        return {
            "best_model_name": best_model_name,
            "best_model": training_results[best_model_name]["model"],
            "best_model_metrics": training_results[best_model_name]["metrics"],
            "best_model_file_path": training_results[best_model_name]["model_file_path"],
            "best_model_mlflow_run_id": training_results[best_model_name]["mlflow_run_id"],
            "best_model_mlflow_artifact_path": training_results[best_model_name]["mlflow_model_artifact_path"],
        }
    
    
    def save_best_model(self, best_model_details:dict) -> Path:
        best_model_path = self.config.best_model_path
        best_model = best_model_details["best_model"]
        best_model_name = best_model_details["best_model_name"]

        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model '{best_model_name}' saved at {best_model_path}")
        return best_model_path


    def save_metrics(self, best_model_details: dict, training_results: dict) -> Path:
        metrics_path = self.config.metric_file_name

        metrics_payload = {
            "best_model_name": best_model_details["best_model_name"],
            "best_model_metrics": best_model_details["best_model_metrics"],
            "all_model_metrics": {
                model_name: details["metrics"]
                for model_name, details in training_results.items()
            },
        }
        save_json(metrics_path, metrics_payload)
        return metrics_path


    def save_mlflow_model_info(self, run_id: str, model_artifact_path: str = "model"):
        mlflow_model_info_path = self.config.root_dir/"mlflow_model_info.json"

        mlflow_model_info_payload = {
            "run_id": run_id,
            "model_artifact_path": model_artifact_path,
            "registered_model_name": self.registered_model_name,
            "local_model_path": str(self.config.best_model_path),
            }
        
        save_json(mlflow_model_info_path, mlflow_model_info_payload)
        return mlflow_model_info_path

    def run(self) ->dict[str, Any]:
        training_results = self.train_and_track_models()

        best_model_details = self.select_best_model(training_results)

        best_model_path = self.save_best_model(best_model_details)

        metrics_path = self.save_metrics(best_model_details=best_model_details, training_results=training_results)

        mlflow_model_info_path = self.save_mlflow_model_info(run_id=best_model_details["best_model_mlflow_run_id"], model_artifact_path=best_model_details["best_model_mlflow_artifact_path"],)

        logger.info(f"Best model saved at: {best_model_path}")
        logger.info(f"Metrics saved at: {metrics_path}")
        logger.info(f"MLflow model info saved at: {mlflow_model_info_path}")

        return {
            "best_model_path": str(best_model_path),
            "metrics_path": str(metrics_path),
            "mlflow_model_info_path": str(mlflow_model_info_path),
            "best_model_name": best_model_details["best_model_name"],
            "best_model_mlflow_run_id": best_model_details["best_model_mlflow_run_id"],
            "best_model_mlflow_artifact_path": best_model_details["best_model_mlflow_artifact_path"],
        }