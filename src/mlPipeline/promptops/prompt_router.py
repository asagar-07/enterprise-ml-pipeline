import random
from typing import Dict


class PromptRouter:
    """ Select prompt version based on traffic weights.
        Example: {"v1": 0.7, "v2": 0.3}
    """

    @staticmethod
    def choose_version(traffic_config: Dict[str, float]) -> str:
        if not traffic_config:
            raise ValueError("Traffic config cannot be empty")

        versions = list(traffic_config.keys())
        weights = list(traffic_config.values())

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Traffic weights must sum to a value greater than 0")

        normalized_weights = [weight / total_weight for weight in weights]
        selected_version = random.choices(population=versions, weights=normalized_weights, k=1 )[0]

        return selected_version