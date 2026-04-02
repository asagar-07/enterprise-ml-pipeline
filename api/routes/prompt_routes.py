from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.mlPipeline.components.llm_service import generate_response
from mlPipeline.promptops.prompt_service import PromptService
from api.schemas import ExplainRequest
from mlPipeline.components.llm_service import call_ollama

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


PROMPT_TEMPLATES = {
    "v1": """
You are a fraud detection assistant.

Transaction details:
- Time: {time}
- Amount: {amount}
- Features: {features}

Model prediction: {prediction}

Explain why this transaction may have been classified this way.
""".strip(),

    "v2": """
You are assisting with fraud prediction interpretation.

Important context:
- V1-V28 are anonymized transformed features.
- Do not pretend to know their exact business meaning.

Transaction input:
- Time: {time}
- Amount: {amount}
- Features: {features}

Model output:
- Predicted class: {prediction}

Explain the result clearly and cautiously.
End by stating this is an LLM-generated explanation, not the model's internal reasoning.
""".strip(),
}


@router.get("/templates")
def get_prompt_templates():
    return {
        "available_versions": list(PROMPT_TEMPLATES.keys()),
        "templates": PROMPT_TEMPLATES,
    }


@router.post("/explain")
def explain(request: ExplainRequest):

    template = PROMPT_TEMPLATES.get(request.prompt_version, PROMPT_TEMPLATES["v1"])

    # build features string (if not already done)
    features = "\n".join([f"{k}: {v}" for k, v in request.model_dump().items() if k.startswith("V")])

    prompt = template.format(time=request.Time, amount=request.Amount, features=features, prediction=request.prediction)

    llm_response = call_ollama(prompt)

    return {
        "prompt_version": request.prompt_version,
        "prompt": prompt,
        "explanation": llm_response,
    }