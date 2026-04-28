import numpy as np


class TabularModel:
    def __init__(self):
        self._transitions: dict[tuple[int, int], tuple[float, int]] = {}

    def update(self, s: int, a: int, r: float, s_next: int) -> None:
        self._transitions[(s, a)] = (r, s_next)

    def sample(self, rng: np.random.Generator) -> tuple[int, int, float, int]:
        if not self._transitions:
            raise ValueError("model is empty")
        keys = list(self._transitions)
        s, a = keys[rng.integers(len(keys))]
        r, s_next = self._transitions[(s, a)]
        return s, a, r, s_next
