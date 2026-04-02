from mlPipeline.promptops.prompt_router import PromptRouter


def test_prompt_router_returns_valid_version():
    traffic_config = {
        "v1": 0.7,
        "v2": 0.3
    }

    selected_version = PromptRouter.choose_version(traffic_config)

    assert selected_version in traffic_config