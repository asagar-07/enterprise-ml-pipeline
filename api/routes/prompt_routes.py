from fastapi import APIRouter
from pydantic import BaseModel
from src.mlPipeline.components.llm_service import generate_response

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/generate")
def generate(req: PromptRequest):
    response = generate_response(req.prompt)
    return {"response": response}