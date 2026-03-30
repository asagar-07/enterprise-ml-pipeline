import mlflow
from mlflow.tracking import MlflowClient
from collections import Counter
from typing import Any

class PromptAnalytics:
    def __init__(self, experiment_name: str = "prompt_tracking", tracking_uri: str = "http://mlflow:5000", ):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment = self.client.get_experiment_by_name(experiment_name)


    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


    def _get_common_param_values(self, runs, param_names: list[str], ) -> dict[str, Any]:
        result = {}

        for param_name in param_names:
            values = [
                run.data.params.get(param_name)
                for run in runs
                if run.data.params.get(param_name) is not None
            ]

            if values:
                most_common_value, count = Counter(values).most_common(1)[0]
                result[param_name] = {
                    "value": most_common_value,
                    "count": count,
                }
            else:
                result[param_name] = None

        return result


    def get_recent_summary(self, max_results: int = 10) -> dict[str, Any]:
        if not self.experiment:
            return {
                "experiment_found": False,
                "message": "Experiment 'prompt_tracking' not found.",
            }

        runs = self.client.search_runs(experiment_ids=[self.experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=max_results, )

        if not runs:
            return {
                "experiment_found": True,
                "total_runs_analyzed": 0,
                "fraud_runs": 0,
                "non_fraud_runs": 0,
                "avg_latency_ms": 0.0,
                "avg_fraud_latency_ms": 0.0,
                "avg_non_fraud_latency_ms": 0.0,
                "common_fraud_patterns": {},
                "recent_runs": [],
            }

        fraud_runs = []
        non_fraud_runs = []
        all_latency = []

        recent_runs = []

        for run in runs:
            metrics = run.data.metrics
            params = run.data.params

            latency_ms = self._safe_float(metrics.get("latency_ms"), 0.0)
            is_fraud_response = self._safe_int(metrics.get("is_fraud_response"), 0)

            all_latency.append(latency_ms)

            run_summary = {
                "run_id": run.info.run_id,
                "model_name": params.get("model_name"),
                "provider": params.get("provider"),
                "prompt_category": params.get("prompt_category"),
                "mentions_fraud": params.get("mentions_fraud"),
                "mentions_transaction": params.get("mentions_transaction"),
                "mentions_payment": params.get("mentions_payment"),
                "latency_ms": latency_ms,
                "is_fraud_response": is_fraud_response,
            }
            recent_runs.append(run_summary)

            if is_fraud_response == 1:
                fraud_runs.append(run)
            else:
                non_fraud_runs.append(run)

        fraud_latency = [
            self._safe_float(run.data.metrics.get("latency_ms"), 0.0)
            for run in fraud_runs
        ]
        non_fraud_latency = [
            self._safe_float(run.data.metrics.get("latency_ms"), 0.0)
            for run in non_fraud_runs
        ]

        common_fraud_patterns = self._get_common_param_values(
            fraud_runs,
            [
                "prompt_category",
                "mentions_fraud",
                "mentions_transaction",
                "mentions_payment",
                "has_question_mark",
                "provider",
                "model_name",
            ],
        ) if fraud_runs else {}

        return {
            "experiment_found": True,
            "total_runs_analyzed": len(runs),
            "fraud_runs": len(fraud_runs),
            "non_fraud_runs": len(non_fraud_runs),
            "avg_latency_ms": round(sum(all_latency) / len(all_latency), 2) if all_latency else 0.0,
            "avg_fraud_latency_ms": round(sum(fraud_latency) / len(fraud_latency), 2) if fraud_latency else 0.0,
            "avg_non_fraud_latency_ms": round(sum(non_fraud_latency) / len(non_fraud_latency), 2) if non_fraud_latency else 0.0,
            "common_fraud_patterns": common_fraud_patterns,
            "recent_runs": recent_runs,
        }