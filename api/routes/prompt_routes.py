from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.mlPipeline.components.llm_service import generate_response
from mlPipeline.promptops.prompt_service import PromptService

router = APIRouter(prefix="/prompt", tags=["PromptOps"])

prompt_service = PromptService()


class PromptRequest(BaseModel):
    prompt: str


class PromptRunRequest(BaseModel):
    prompt_name: str = "fraud_detection"
    data: dict | str


@router.post("/generate")
def generate(req: PromptRequest):
    """ Send a raw prompt to the LLM service. """
    response = generate_response(req.prompt, enable_tracking=True)
    return {"response": response}


@router.post("/run")
def run_prompt(request: PromptRunRequest):
    """ Select prompt version via A/B routing, render the prompt, call the LLM service, and log the run. """
    try:
        prompt_payload = prompt_service.render_prompt(prompt_name=request.prompt_name, input_data=request.data)
        llm_response = generate_response(prompt_payload["rendered_prompt"], enable_tracking=False)
        result = {
            **prompt_payload,
            "input_data": request.data,
            "llm_response": llm_response,
        }

        prompt_service.log_prompt_run(result)

        return {
            "status": "success",
            "prompt_name": result["prompt_name"],
            "prompt_version": result["prompt_version"],
            "rendered_prompt": result["rendered_prompt"],
            "llm_response": result["llm_response"],
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

