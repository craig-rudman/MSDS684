from __future__ import annotations

import gymnasium as gym


class MountainCarEnvironment:
    """Facade over the Gymnasium MountainCar-v0 environment. Only one instance exists."""

    _instance: MountainCarEnvironment | None = None

    def __new__(cls) -> MountainCarEnvironment:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_env'):
            self._env = gym.make('MountainCar-v0')

    def reset(self, seed=42):
        self._env.action_space.seed(seed)
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        """Release environment resources."""
        return self._env.close()

    @property
    def observation_space(self):
        """Return the Gymnasium observation space (Box)."""
        return self._env.observation_space

    @property
    def action_space(self):
        """Return the Gymnasium action space (Discrete)."""
        return self._env.action_space
