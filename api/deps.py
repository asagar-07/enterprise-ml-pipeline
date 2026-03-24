from functools import lru_cache
from fastapi import Request

from mlPipeline.constants import PARAMS_FILE_PATH
from mlPipeline.utils.common import read_yaml
from mlPipeline.utils.model_registry import load_champion_model

params = read_yaml(PARAMS_FILE_PATH)
model_name = params.model_registry.registered_model_name

@lru_cache(maxsize=1)
def _build_model():
    return load_champion_model(model_name) 

def get_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is not None:
        return model
    return _build_model()

def teardown_model():
    _build_model.cache_clear()