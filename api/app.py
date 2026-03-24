
from __future__ import annotations

import gc
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, status, HTTPException, Depends

from api.deps import get_model, teardown_model
from api.errors import value_error_handler, unhandled_exception_handler, request_validation_error_hanlder
from fastapi.exceptions import RequestValidationError
from api.schemas import SinglePredictRequest, SinglePredictResponse, BatchPredictRequest, BatchPredictResponse, ItemResult, ErrorDetail

from mlPipeline.constants import PARAMS_FILE_PATH
from mlPipeline.utils.common import read_yaml
from mlPipeline.utils.model_registry import load_champion_model

params = read_yaml(PARAMS_FILE_PATH)
model_name = params.model_registry.registered_model_name

@asynccontextmanager
async def lifespan(app: FastAPI):
    #---Startup---
    try:
        model = load_champion_model(model_name) # construct the Fraud Model 
        app.state.model = model # store model in app.state.model
        app.state.ready = True # set ready state = True
        app.state.startup_error = None
    except Exception as e:
        # Handle initialization failure
        app.state.model = None
        app.state.ready = False
        app.state.startup_error = str(e)
    yield
    #--shutdown--
    app.state.model = None
    teardown_model()
    # Forcing Python garbage collection
    gc.collect()

app = FastAPI(title="Credit Card Fraud Detection API", lifespan=lifespan)
app.add_exception_handler(RequestValidationError, request_validation_error_hanlder)
app.add_exception_handler(ValueError, value_error_handler) # rather than @app.exception_handler() as my error hanldlers are centralized and defined in api/errors.py
app.add_exception_handler(Exception, unhandled_exception_handler) # rather than @app.exception_handler() as my error hanldlers are centralized and defined in api/errors.py

@app.get("/")
async def home():
    return {"message": "Fraud Detection API is running"}

@app.get("/health") 
async def health_check():
    """Readiness + Liveness check: 200 only when artifacts loaded successfully, else 503"""
    if getattr(app.state, "ready", False):
        return {"ready": True, "startup_error": None}
    
    err = getattr(app.state, "startup_error", "Unknown startup error")
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Fraud-Scoring Model Application failed to load. Error: {err}")


@app.post("/predict", response_model=SinglePredictResponse) # Sinlge request: unified response for singular named feature dict
async def single_predict(input: SinglePredictRequest, model = Depends(get_model)) -> SinglePredictResponse:
    # Input features are validated internally and raised exceptions bubbles up and FASTAPI catches and runs value_error_handler
    df = pd.DataFrame([input.features.model_dump()])
    prediction = model.predict(df)
    
    return SinglePredictResponse(
            request_id=input.request_id,
            result={
                "prediction": int(prediction[0]),
                "label": "Fraud" if int(prediction[0]) == 1 else "Not Fraud",
            },
        )


@app.post("/predict/batch", response_model=BatchPredictResponse) 
# Batch request: list of unified responses (per row), with per-item errors if any captured for list of featured dicts
# Batch guardrails, max_batch_size = 256, reject with 422 ERROR if larger than 256, each item should have features dict
# Missing, extra, non-numeric feature keys are already rejected by my serivce ( ValueError)
async def batch_predict(input: BatchPredictRequest, model = Depends(get_model)) -> BatchPredictResponse:
    if len(input.items) > 256:
        raise ValueError(f"Batch size cannot exceed 256, found {len(input.items)} instead!")

    succeeded = 0
    failed = 0
    item_results = []

    for item in input.items:
        try:
            df = pd.DataFrame([item.features.model_dump()])
            prediction = model.predict(df)

            item_results.append(
                ItemResult(
                    item_id = item.item_id,
                    result = { "prediction": int(prediction[0]),
                              "label": "Fraud" if prediction[0] == 1 else "Not Fraud"
                              }, 
                    error = None,
                    )
                )
            succeeded += 1
        except ValueError as e:
            item_results.append(
                ItemResult(
                    item_id = item.item_id,
                    result = None, 
                    error=ErrorDetail(type="ValidationError", message=str(e)),
                )
            )
            failed += 1
            #errors[key] = {"error": { "type": "ValidationError", "message": str(e)}}
    
    return BatchPredictResponse(
        request_id=input.request_id,
        results=item_results,
        summary={
            "total": len(input.items), 
            "succeeded": succeeded, 
            "failed": failed,
            },
        )