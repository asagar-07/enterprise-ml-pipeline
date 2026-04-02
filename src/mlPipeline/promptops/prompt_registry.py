from pathlib import Path
from typing import Any, Dict
from mlPipeline.utils.common import logger, read_yaml
from mlPipeline.constants import PROMPT_CONFIG_FILE_PATH


class PromptRegistry:
    """Load and serve prompt definitions from YAML config."""

    def __init__(self, config_path: Path = PROMPT_CONFIG_FILE_PATH) -> None:
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config file not found at: {self.config_path}")

        logger.info(f"Loading prompt config from: {self.config_path}")
        config = read_yaml(self.config_path) or {}

        if "prompts" not in config:
            raise ValueError("Invalid prompt config: missing 'prompts' section")
        return config

    def get_prompt_config(self, prompt_name: str) -> Dict[str, Any]:
        prompt_config = self._config["prompts"].get(prompt_name)

        if prompt_config is None:
            raise ValueError(f"Prompt '{prompt_name}' not found in config")

        if not prompt_config.get("active", False):
            raise ValueError(f"Prompt '{prompt_name}' is not active")
        return prompt_config

    def get_versions(self, prompt_name: str) -> Dict[str, str]:
        versions = self.get_prompt_config(prompt_name).get("versions", {})

        if not versions:
            raise ValueError(f"No versions found for prompt '{prompt_name}'")
        return versions

    def get_template(self, prompt_name: str, version: str) -> str:
        template = self.get_versions(prompt_name).get(version)

        if template is None:
            raise ValueError(f"Version '{version}' not found for prompt '{prompt_name}'")
        return template

    def get_traffic_config(self, prompt_name: str) -> Dict[str, float]:
        traffic = self.get_prompt_config(prompt_name).get("traffic", {})

        if not traffic:
            raise ValueError(f"No traffic config found for prompt '{prompt_name}'")
        return traffic