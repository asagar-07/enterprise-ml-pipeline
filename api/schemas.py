# holds pydantic request and response models
# Single Predict
# Request
#   request_id : Optional[str]
#   features: TransactionFeatures
# 
# Response
#   request_id : (echo)
#   result: PredicitonResult
# 
# Batch Predict
# Request
#   request_id : Optional[str]
#   items: List[{item_id?:str, features: TransactionFeatures}]
# 
# Batch Response
#   request_id: (echo)
#   results: List[{ item_id?: str, results?: PredictionResult, error?:{type, message}}]
#   summary: {total, succeeded, failed}

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Literal

class TransactionFeatures(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    model_config = ConfigDict(extra="forbid")

class PredictionResult(BaseModel):
    prediction: int
    label: str

class SinglePredictRequest(BaseModel):
    request_id: Optional[str] = None
    features: TransactionFeatures
    model_config = ConfigDict(extra='forbid')


class SinglePredictResponse(BaseModel):
    request_id: Optional[str] = None
    result: PredictionResult
#-----------------------------------------------

class BatchItem(BaseModel):
    item_id: Optional[str] = None
    features: TransactionFeatures
    model_config = ConfigDict(extra='forbid')


class ErrorDetail(BaseModel):
    type: str
    message: str

class ItemResult(BaseModel):
    item_id: Optional[str] = None
    result: Optional[PredictionResult] = None
    error: Optional[ErrorDetail]= None

class Summary(BaseModel):
    total: int
    succeeded: int
    failed: int

class BatchPredictRequest(BaseModel):
    request_id: Optional[str] = None
    items: List[BatchItem]
    model_config = ConfigDict(extra='forbid')

class BatchPredictResponse(BaseModel):
    request_id: Optional[str] = None
    results: List[ItemResult]
    summary: Summary


class ExplainRequest(BaseModel):
    Time: float = Field(..., example=12345)
    V1: float = Field(..., example=-1.23)
    V2: float = Field(..., example=0.45)
    V3: float = Field(..., example=-2.11)
    V4: float = Field(..., example=1.02)
    V5: float = Field(..., example=-0.76)
    V6: float = Field(..., example=0.14)
    V7: float = Field(..., example=-1.55)
    V8: float = Field(..., example=0.22)
    V9: float = Field(..., example=-0.91)
    V10: float = Field(..., example=1.34)
    V11: float = Field(..., example=-0.18)
    V12: float = Field(..., example=0.67)
    V13: float = Field(..., example=-1.41)
    V14: float = Field(..., example=2.05)
    V15: float = Field(..., example=-0.33)
    V16: float = Field(..., example=0.58)
    V17: float = Field(..., example=-1.76)
    V18: float = Field(..., example=0.94)
    V19: float = Field(..., example=-0.27)
    V20: float = Field(..., example=0.11)
    V21: float = Field(..., example=-0.44)
    V22: float = Field(..., example=0.72)
    V23: float = Field(..., example=-0.09)
    V24: float = Field(..., example=0.36)
    V25: float = Field(..., example=-0.62)
    V26: float = Field(..., example=0.49)
    V27: float = Field(..., example=-0.21)
    V28: float = Field(..., example=0.08)
    Amount: float = Field(..., example=249.99)
    prediction: str = Field(..., example="Fraud")
    prompt_version: Literal["v1", "v2"] = Field(
        default="v1",
        description="Prompt template version to use for explanation.",
        example="v1",
    )