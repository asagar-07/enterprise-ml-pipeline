from pathlib import Path
import joblib
from typing import Any
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)
import matplotlib.pyplot as plt

from mlPipeline.constants import PARAMS_FILE_PATH
from mlPipeline.utils.common import read_yaml, save_json, load_json, save_text
from mlPipeline.utils.model_registry import configure_mlflow
from mlPipeline.entity.config_entity import ModelEvaluationConfig
from mlPipeline import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.target_col_name = self.params.target_column

        # MLflow / registry defaults
        self.mlflow_tracking_uri = self._safe_get(self.params, ["mlflow", "tracking_uri"], default="sqlite:///mlflow.db")
        self.mlflow_experiment_name = self._safe_get(self.params, ["mlflow", "experiment_name"], default="Creditcard Fraud Detection")
        self.registered_model_name = self._safe_get(self.params, ["model_registry", "registered_model_name"], default="creditcard_fraud_model",)

        # Evaluation gate defaults
        self.enable_model_registration = self._safe_get(self.params, ["model_evaluation", "enable_model_registration"], default=True,)
        self.min_f1_score = self._safe_get(self.params, ["model_evaluation", "min_f1_score"],default=0.0,)
        self.min_roc_auc = self._safe_get(self.params, ["model_evaluation", "min_roc_auc"], default=0.0,)
        self.min_recall = self._safe_get(self.params, ["model_evaluation", "min_recall"], default=0.0,)

        configure_mlflow()
        logger.info(f"MLflow tracking URI in evaluation: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow registry URI in evaluation: {mlflow.get_registry_uri()}")

    
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

    def _load_trained_model(self):
        return joblib.load(self.config.best_model_path)
    
    
    def _load_test_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.test_data_path)


    def _prepare_features_and_target(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = dataset.drop(columns=[self.target_col_name])
        y = dataset[self.target_col_name]
        return X, y


    def _build_metrics(self, y_test, y_pred, y_pred_proba) -> dict[str, float | None]:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        }
        return metrics
    

    def _save_confusion_matrix(self, y_test, y_pred) -> Path:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        cm_path = self.config.root_dir / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        return cm_path


    def _save_classification_report(self, y_test, y_pred) -> Path:
        report_text = classification_report(y_test, y_pred, zero_division=0)
        report_path = self.config.root_dir / "classification_report.txt"
        save_text(report_path, report_text)
        return report_path


    def _get_training_mlflow_metadata_path(self) -> Path:
        return self.config.best_model_path.parent / "mlflow_model_info.json"


    def _load_training_mlflow_metadata(self) -> dict | None:
        metadata_path = self._get_training_mlflow_metadata_path()

        if not metadata_path.exists():
            logger.warning(
                f"Training MLflow metadata file not found at: {metadata_path}. "
                "Model registration will be skipped unless this file is created by training stage."
            )
            return None

        return load_json(metadata_path)


    def _should_register_model(self, metrics: dict[str, float | None]) -> tuple[bool, str]:
        if not self.enable_model_registration:
            return False, "model_registration_disabled"

        f1_ok = metrics["f1_score"] is not None and metrics["f1_score"] >= self.min_f1_score
        recall_ok = metrics["recall"] is not None and metrics["recall"] >= self.min_recall
        roc_ok = (True if metrics["roc_auc"] is None else metrics["roc_auc"] >= self.min_roc_auc)

        passed = f1_ok and recall_ok and roc_ok

        reason = (
            f"passed_gate(f1>={self.min_f1_score}, "
            f"recall>={self.min_recall}, roc_auc>={self.min_roc_auc})"
            if passed
            else
            f"failed_gate(f1>={self.min_f1_score}, "
            f"recall>={self.min_recall}, roc_auc>={self.min_roc_auc})"
        )

        return passed, reason
    

    def _set_candidate_alias(self, registered_model_name: str, model_version: str | int,) -> None:
            client = MlflowClient()
            try:
                client.set_registered_model_alias(name=registered_model_name, alias="candidate", version = str(model_version),)
                logger.info(f" Alias 'candidate'set to version {model_version} "
                            f"for model {registered_model_name}")
            except Exception as e:
                logger.error(
                    f"Failed to set 'candidate' alias for model"
                    f" {registered_model_name}, version {model_version}"
                )
                raise e


    def _get_current_champion(self, registered_model_name: str, ) -> dict | None:
        client = MlflowClient()

        try:
            model_version = client.get_model_version_by_alias(name=registered_model_name, alias="champion",)
            if not model_version:
                logger.info("No champion model found.")
                return None
            
            tags = model_version.tags or {}

            champion_info = {
                "version": model_version.version,
                "run_id": model_version.run_id,
                "metrics": {
                    "f1_score": float(tags.get("f1_score", 0.0)),
                    "recall": float(tags.get("recall", 0.0)),
                    "roc_auc": float(tags.get("roc_auc", 0.0)),
                },
            }
            logger.info(
                f"Current champion found: version= {champion_info['version']}, "
                f"metrics={champion_info['metrics']}"
            )
            return champion_info
        
        except Exception:
            logger.info("Champion alias does not exist yet. Treating as first model.")
            return None


    def _promote_candidate_to_champion_if_better(self, registered_model_name: str, candidate_version: int | str, candidate_metrics: dict[str, float | None], ) -> dict[str, Any]:
        """ Promote the given candidate version to 'champion' if:
            - no current champion exists, or
            - candidate is better by priority:  f1_score -> recall -> roc_auc
            - Returns a summary dict of the promotion decision. """
        client = MlflowClient()

        promotion_result = {
            "candidate_version": str(candidate_version),
            "previous_champion_version": None,
            "promoted_to_champion": False,
            "reason": None,
        }

        current_champion = self._get_current_champion(registered_model_name)

        if current_champion is None:
            client.set_registered_model_alias(name=registered_model_name, alias="champion", version=str(candidate_version),)

            promotion_result["promoted_to_champion"] = True
            promotion_result["reason"] = "no_existing_champion"

            logger.info(
                f"No existing champion. Promoted candidate version {candidate_version} "
                f"to champion for model '{registered_model_name}'."
            )
            return promotion_result

        champion_metrics = current_champion["metrics"]
        promotion_result["previous_champion_version"] = str(current_champion["version"])

        candidate_f1 = candidate_metrics.get("f1_score")
        candidate_recall = candidate_metrics.get("recall")
        candidate_roc_auc = candidate_metrics.get("roc_auc")

        champion_f1 = champion_metrics.get("f1_score")
        champion_recall = champion_metrics.get("recall")
        champion_roc_auc = champion_metrics.get("roc_auc")

        def safe_metric(value, default=float("-inf")):
            return default if value is None else value

        candidate_tuple = (
            safe_metric(candidate_f1),
            safe_metric(candidate_recall),
            safe_metric(candidate_roc_auc),
        )
        champion_tuple = (
            safe_metric(champion_f1),
            safe_metric(champion_recall),
            safe_metric(champion_roc_auc),
        )

        if candidate_tuple > champion_tuple:
            client.set_registered_model_alias(name=registered_model_name, alias="champion", version=str(candidate_version),)

            promotion_result["promoted_to_champion"] = True
            promotion_result["reason"] = (
                f"candidate_better_than_champion: "
                f"candidate={candidate_tuple} > champion={champion_tuple}"
            )

            logger.info(
                f"Promoted candidate version {candidate_version} to champion for "
                f"model '{registered_model_name}'. "
                f"Candidate metrics={candidate_metrics}, "
                f"Champion metrics={champion_metrics}"
            )
        else:
            promotion_result["reason"] = (
                f"candidate_not_better_than_champion: "
                f"candidate={candidate_tuple} <= champion={champion_tuple}"
            )

            logger.info(
                f"Candidate version {candidate_version} not promoted. "
                f"Candidate metrics={candidate_metrics}, "
                f"Champion metrics={champion_metrics}"
            )

        return promotion_result


    def _register_model_if_applicable(self, metrics: dict[str, float | None], evaluation_run_id: str,) -> dict[str, Any]:
        registration_result = {
            "attempted": False,
            "registered": False,
            "registered_model_name": None,
            "registered_model_version": None,
            "training_run_id": None,
            "model_uri": None,
            "reason": None,
            "promotion": None,
        }

        should_register, gate_reason = self._should_register_model(metrics)
        registration_result["reason"] = gate_reason

        if not should_register:
            logger.info(f"Model registration skipped: {gate_reason}")
            return registration_result

        training_metadata = self._load_training_mlflow_metadata()
        if not training_metadata:
            registration_result["attempted"] = True
            registration_result["reason"] = ("evaluation_passed_but_training_mlflow_metadata_missing")
            return registration_result

        logged_model_uri = training_metadata.get("logged_model_uri")
        registered_model_name = training_metadata.get("registered_model_name", self.registered_model_name,)

        if not logged_model_uri:
            registration_result["attempted"] = True
            registration_result["reason"] = ("evaluation_passed_but_training_run_id_missing_in_metadata")
            return registration_result

        model_uri = logged_model_uri
        training_run_id = training_metadata.get("run_id")

        logger.info(
            f"Evaluation passed. Registering model from URI: {model_uri} "
            f"as '{registered_model_name}'"
        )

        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name,)
        self._set_candidate_alias(registered_model_name=registered_model_name, model_version=result.version,)
        promotion_result = self._promote_candidate_to_champion_if_better(registered_model_name=registered_model_name, candidate_version= result.version, candidate_metrics= metrics,)

        registration_result.update(
            {
                "attempted": True,
                "registered": True,
                "registered_model_name": registered_model_name,
                "registered_model_version": result.version,
                "training_run_id": training_run_id,
                "model_uri": model_uri,
                "reason": gate_reason,
                "promotion": promotion_result,
            }
        )

        # Tagging the registered version with evaluation metadata
        client = MlflowClient()
        client.set_model_version_tag(name=registered_model_name, version=result.version, key="source_training_run_id", value=str(training_run_id),)
        client.set_model_version_tag(name=registered_model_name, version=result.version, key="evaluation_run_id", value=str(evaluation_run_id),)
        client.set_model_version_tag(name=registered_model_name, version=result.version, key="stage", value="post_evaluation_registration",)
        client.set_model_version_tag(name=registered_model_name, version=result.version, key="f1_score", value=str(metrics["f1_score"]),)
        client.set_model_version_tag(name=registered_model_name, version=result.version, key="recall", value=str(metrics["recall"]),)
        client.set_model_version_tag(name=registered_model_name, version=result.version, key="roc_auc", value=str(metrics["roc_auc"]),)
    
        return registration_result


    def evaluate_and_track_model(self) -> dict[str, Any]:
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_experiment(self.mlflow_experiment_name)

        with mlflow.start_run(run_name = "model_evaluation") as evaluation_run:
            evaluation_run_id = evaluation_run.info.run_id

            mlflow.set_tag("stage", "model_evaluation")
            mlflow.set_tag("run_type", "evaluation")
            mlflow.log_param("target_column", self.target_col_name)
            mlflow.log_param("test_data_path", str(self.config.test_data_path))
            mlflow.log_param("best_model_path", str(self.config.best_model_path))

            logger.info(f"Evaluation Run ID: {evaluation_run_id}")

            test_df = self._load_test_data()
            X_test, y_test = self._prepare_features_and_target(test_df)
            model = self._load_trained_model()

            y_pred = model.predict(X_test)
            
            y_pred_proba = None
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                logger.warning("Loaded model does not support predict_proba(), roc_auc will be logged as null")

            metrics = self._build_metrics(y_test, y_pred, y_pred_proba)
            
            # Logging only non-null numeric metrics to MLflow
            mlflow_metrics = {k: v for k, v in metrics.items() if v is not None}
            mlflow.log_metrics(mlflow_metrics)

            metrics_path = self.config.metric_file_name
            save_json(metrics_path, metrics)

            report_path = self._save_classification_report(y_test, y_pred)
            cm_path = self._save_confusion_matrix(y_test, y_pred)

            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(cm_path))

            registration_result = self._register_model_if_applicable(metrics=metrics, evaluation_run_id=evaluation_run_id,)

            mlflow.set_tag("model_registration_attempted", str(registration_result["attempted"]))
            mlflow.set_tag("model_registered", str(registration_result["registered"]))
            mlflow.set_tag("registration_reason", str(registration_result["reason"]))

            if registration_result["training_run_id"]:
                mlflow.set_tag("source_training_run_id", str(registration_result["training_run_id"]),)

            if registration_result["registered"]:
                mlflow.set_tag("registered_model_name", str(registration_result["registered_model_name"]),)
                mlflow.set_tag("registered_model_version", str(registration_result["registered_model_version"]),)
                mlflow.set_tag("registered_model_uri", str(registration_result["model_uri"]),)

            result = {
                "evaluation_run_id": evaluation_run_id,
                "metrics": metrics,
                "metric_file_path": str(metrics_path),
                "classification_report_path": str(report_path),
                "confusion_matrix_path": str(cm_path),
                "registration": registration_result,
            }

            logger.info(f"Evaluation completed successfully: {result}")
            return result

            