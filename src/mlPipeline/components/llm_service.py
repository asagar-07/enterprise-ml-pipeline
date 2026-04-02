import time
import os
import requests

from mlPipeline.utils.prompt_tracking import PromptTracker

tracker = PromptTracker()

def simple_evaluation(prompt: str, response: str) -> dict:
    prompt_lower = prompt.lower()
    response_lower = response.lower()

    return {
        "response_length": len(response),
        "prompt_length_metric": len(prompt),
        "contains_fraud_keyword": int("fraud" in response_lower),
        "prompt_mentions_fraud": int("fraud" in prompt_lower),
        "is_fraud_response": int("fraud" in response_lower),
    }


def build_prompt_features(prompt: str) -> dict:
    prompt_lower = prompt.lower()

    return {
        "provider": "ollama",
        "temperature": 0.7,
        "has_question_mark": int("?" in prompt),
        "mentions_fraud": int("fraud" in prompt_lower),
        "mentions_transaction": int("transaction" in prompt_lower),
        "mentions_payment": int("payment" in prompt_lower),
        "prompt_category": "fraud_check" if "fraud" in prompt_lower else "general",
    }


def call_ollama(prompt:str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    response = requests.post(f"{base_url}/api/generate", json={"model": model, "prompt": prompt, "stream": False }, timeout=120, )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def generate_response(prompt: str, enable_tracking: bool = True) -> str:
    start_time = time.time()

    # Replacing later with OpenAI/ Grok/ Ollama
    # response = f"Mock response for: {prompt}"

    #Integrating Ollama response
    try:
        response = call_ollama(prompt)
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
    
    except Exception as e:
        print("OLLAMA FAILED:", repr(e))
        response = f"[FALLBACK MOCK] Response for: {prompt}"
        model_name = "Fallback-mock"

    latency = time.time() - start_time
    metrics = simple_evaluation(prompt, response)
    extra_params = build_prompt_features(prompt)

    #For Mock response tracking
    #tracker.track(prompt=prompt, response=response, model_name="mock-llm-v1", latency=latency, extra_params=extra_params, metrics=metrics)

    # For Ollama response tracking
    if enable_tracking:
        tracker.track(prompt=prompt, response=response, model_name=model_name, latency=latency, extra_params=extra_params, metrics=metrics, )
    return response
