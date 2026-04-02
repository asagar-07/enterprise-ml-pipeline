from mlPipeline.promptops.prompt_registry import PromptRegistry


def test_prompt_registry_loads_versions():
    registry = PromptRegistry("configs/prompt_config.yaml")
    versions = registry.get_versions("fraud_detection")

    assert "v1" in versions
    assert "v2" in versions