import mlflow
import uuid

class PromptTracker:
    def __init__(self, experiment_name="prompt_tracking", tracking_uri: str = "http://mlflow:5000",):
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(tracking_uri)

    def track(self, prompt: str, response: str, model_name: str, latency: float, extra_params: dict = None, metrics: dict = None,):
        run_name = f"prompt_run_{uuid.uuid4().hex[:8]}"

        with mlflow.start_run(run_name=run_name):
            # Logging inputs
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("prompt_length", len(prompt))

            if extra_params:
                for k, v in extra_params.items():
                    mlflow.log_param(k, v)

            # Logging outputs
            mlflow.log_text(prompt, "prompt.txt")
            mlflow.log_text(response, "response.txt")

            # Log metrics
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("latency_ms", latency * 1000)

            # Log Model versions
            mlflow.log_param("model_version", "champion")
            mlflow.log_param("model_uri", "models:/creditcard_fraud_model/Production")

            if metrics:
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)