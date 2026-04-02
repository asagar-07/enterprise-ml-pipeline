from __future__ import annotations

from typing import Any, Dict
import mlflow

class PromptLogger:
    """ Write prompt-run logs to a JSONL file. One JSON object per line. """

    def __init__(self, experiment_name: str = "prompt_versioning_final") -> None:
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value)
    

    def log_run(self, log_data: Dict[str, Any]) -> None:
        prompt_name = self._safe_str(log_data.get("prompt_name"))
        prompt_version = self._safe_str(log_data.get("prompt_version"))
        rendered_prompt = self._safe_str(log_data.get("rendered_prompt"))
        llm_response = self._safe_str(log_data.get("llm_response"))
        input_data = self._safe_str(log_data.get("input_data"))

        with mlflow.start_run(run_name=f"{prompt_name}_{prompt_version}"):
            mlflow.log_params(
                {
                    "prompt_name": prompt_name,
                    "prompt_version": prompt_version,
                }
            )

            mlflow.log_metrics(
                {
                    "prompt_length": float(len(rendered_prompt)),
                    "response_length": float(len(llm_response)),
                }
            )

            mlflow.log_text(rendered_prompt, artifact_file="rendered_prompt.txt")
            mlflow.log_text(llm_response, artifact_file="llm_response.txt")
            mlflow.log_text(input_data, artifact_file="input_data.txt")