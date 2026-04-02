from fastapi import APIRouter, Query
from mlflow import MlflowClient

from collections import Counter
from mlPipeline.analytics.prompt_analytics import PromptAnalytics

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/prompt-generate-summary")
def get_prompt_summary(max_results: int = Query(default=10, ge=1, le=100)):
    analytics = PromptAnalytics()
    return analytics.get_recent_summary(max_results=max_results)


@router.get("/prompt-run-summary")
def prompt_summary(experiment_name: str = "prompt_versioning_final"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {
            "experiment_name": experiment_name,
            "total_runs": 0,
            "by_version": {},
        }

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=500,)

    version_counter = Counter()

    for run in runs:
        version = run.data.params.get("prompt_version", "unknown")
        version_counter[version] += 1

    return {
        "experiment_name": experiment_name,
        "total_runs": len(runs),
        "by_version": dict(version_counter),
    }