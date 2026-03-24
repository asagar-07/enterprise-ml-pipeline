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

from pydantic import BaseModel, ConfigDict
from typing import List, Optional

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
