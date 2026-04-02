from typing import Any, Dict
from mlPipeline.promptops.prompt_logger import PromptLogger
from mlPipeline.promptops.prompt_registry import PromptRegistry
from mlPipeline.promptops.prompt_router import PromptRouter


class PromptService:
    """
    Orchestrates:
    - 1. loading prompt config
    - 2. selecting version via A/B routing
    - 3. template retrieval
    - 4. rendering the prompt
    - 5. logging the prompt run

    API layer will call LLM and generate response.
    """

    def __init__(self, registry: PromptRegistry | None = None, router: PromptRouter | None = None, logger: PromptLogger | None = None) -> None:
        self.registry = registry or PromptRegistry()
        self.router = router or PromptRouter()
        self.logger = logger or PromptLogger()

    def render_prompt(self, prompt_name: str, input_data: Any) -> Dict[str, Any]:
        """ Return promt metadata and rendered prompt."""

        # Get traffic config (A/B weghting) for the prompt
        traffic_config = self.registry.get_traffic_config(prompt_name)
        # Choose version based on weights
        selected_version = self.router.choose_version(traffic_config)
        # Fetch the template for the selected version
        template = self.registry.get_template(prompt_name, selected_version)

        try:
            rendered_prompt = template.format(data=input_data)
        except Exception as e:
            raise ValueError(f"Error rendering prompt '{prompt_name}' with version '{selected_version}': {str(e)}") from e

        return {
            "prompt_name": prompt_name,
            "prompt_version": selected_version,
            "prompt_template": template,
            "rendered_prompt": rendered_prompt,
        }

    def log_prompt_run(self, log_data: Dict[str, Any]) -> None:
        """ Log the prompt run details. """
        self.logger.log_run(log_data)