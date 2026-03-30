from fastapi import APIRouter, Query
from mlPipeline.analytics.prompt_analytics import PromptAnalytics

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/prompt-summary")
def get_prompt_summary(max_results: int = Query(default=10, ge=1, le=100)):
    analytics = PromptAnalytics()
    return analytics.get_recent_summary(max_results=max_results)