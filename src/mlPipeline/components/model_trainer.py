from pathlib import Path
import joblib
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
        self.target_col_name = "Class"


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
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_pred_proba)
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
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Creditcard Fraud Detection")
        with mlflow.start_run(run_name = "model_training_parent") as parent_run:
            mlflow.set_tag("stage", "model_training")
            mlflow.log_param("target_column", self.target_col_name)
            logger.info(f"Parent run: {parent_run.info.run_id}")

            train_df, val_df = self._load_data()
            X_train, y_train = self._prepare_features_and_target(train_df)
            X_val, y_val = self._prepare_features_and_target(val_df)
            models = self._build_models()
            
            for model_name, model in models.items():
                with mlflow.start_run(run_name = model_name, nested=True) as child_run_1:
                    logger.info(f"Child run: {child_run_1.info.run_id}")

                    model.fit(X_train, y_train)

                    metrics = self._evaluate_on_validation(model=model, X_val=X_val, y_val=y_val, model_name=model_name)
                    mlflow.log_param("model_name", model_name)

                    for param_name, param_value in model.get_params().items():
                        mlflow.log_param(param_name, param_value)
                    
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    model_file_path = self.config.trained_model_dir/ f"{model_name}.joblib"
                    joblib.dump(model, model_file_path)
                    mlflow.sklearn.log_model(sk_model=model, name=model_name)
                    
                    training_results[model_name] = {
                        "model": model,
                        "metrics": metrics,
                        "model_file_path": model_file_path
                    }

        return training_results
   

    def select_best_model(self, training_results:dict) -> dict:
        best_model_criteria_rank = ["f1_score", "recall", "roc_auc"]

        best_model_name = max (training_results, key= lambda name: (training_results[name]["metrics"][best_model_criteria_rank[0]],
                                                                    training_results[name]["metrics"][best_model_criteria_rank[1]],
                                                                    training_results[name]["metrics"][best_model_criteria_rank[2]])
                               )
        
        return {
            "best_model_name": best_model_name,
            "best_model": training_results[best_model_name]["model"],
            "best_model_metrics": training_results[best_model_name]["metrics"],
            "best_model_file_path": training_results[best_model_name]["model_file_path"],
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
            }
        }

        save_json(metrics_path, metrics_payload)

        return metrics_path
    

    def run(self):
        training_results = self.train_and_track_models()

        best_model_details = self.select_best_model(training_results)

        best_model_path = self.save_best_model(best_model_details)

        metrics_path = self.save_metrics(
            best_model_details=best_model_details,
            training_results=training_results
        )

        logger.info(f"Best model saved at: {best_model_path}")
        logger.info(f"Metrics saved at: {metrics_path}")