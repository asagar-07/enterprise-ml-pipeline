from mlPipeline.promptops.prompt_service import PromptService


def test_prompt_service_runs_successfully():
    service = PromptService()
    
    result = service.run_prompt(
        prompt_name="fraud_detection",
        input_data={"amount": 1200, 
                    "country": "US"
                    }
    )

    assert result["prompt_name"] == "fraud_detection"
    assert result["prompt_version"] in ["v1", "v2"]
    assert "rendered_prompt" in result
    assert "llm_response" in result