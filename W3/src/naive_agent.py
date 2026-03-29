import numpy as np


class NaiveAgent:
    """Naive baseline agent with fixed strategies for Blackjack."""

    VALID_STRATEGIES = ("random", "always_hit", "always_stick")

    def __init__(self, strategy: str = "random"):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of {self.VALID_STRATEGIES}"
            )
        self.strategy = strategy

    def select_action(self, state: tuple) -> int:
        if self.strategy == "random":
            return int(np.random.choice([0, 1]))
        elif self.strategy == "always_hit":
            return 1
        else:
            return 0

    def update(self, episode: list) -> None:
        """No-op. NaiveAgent does not learn."""
        pass
