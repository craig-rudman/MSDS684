"""Multi-armed bandit environment compatible with Gymnasium."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional


class MultiArmedBanditEnv(gym.Env):
    """
    A k-armed bandit environment following the Gymnasium API.

    Each arm has a Gaussian reward distribution. When randomize=True,
    the true arm means are re-sampled from N(0, 1) on each reset(),
    matching the 10-armed testbed from Sutton & Barto Chapter 2.

    Observations are always 0 (stateless problem).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        k: int = 10,
        arms: Optional[list[dict]] = None,
        randomize: bool = True,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.k = k
        self.randomize = randomize
        self.max_steps = max_steps
        self.np_random = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Discrete(1)

        # Store explicit arm configs if provided
        self._arm_configs = arms
        self._true_means = np.zeros(self.k)
        self._step_count = 0

    def _sample_arm_means(self):
        """Sample or set the true means for each arm."""
        if self._arm_configs is not None:
            for i, cfg in enumerate(self._arm_configs):
                self._true_means[i] = cfg.get("mu", 0.0)
        elif self.randomize:
            self._true_means = self.np_random.standard_normal(self.k)
        else:
            self._true_means = np.zeros(self.k)

    def _sample_reward(self, arm: int) -> float:
        """Sample a reward from the given arm's distribution."""
        if self._arm_configs is not None:
            sigma = self._arm_configs[arm].get("sigma", 1.0)
        else:
            sigma = 1.0
        return float(self.np_random.normal(self._true_means[arm], sigma))

    def true_means(self) -> np.ndarray:
        """Return the true expected reward for each arm."""
        return self._true_means.copy()

    def optimal_arm(self) -> int:
        """Return the index of the arm with the highest true mean."""
        return int(np.argmax(self._true_means))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._sample_arm_means()
        self._step_count = 0
        return 0, {"true_means": self.true_means()}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        reward = self._sample_reward(action)
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        optimal = action == self.optimal_arm()
        return 0, reward, terminated, False, {"optimal_action": optimal}
