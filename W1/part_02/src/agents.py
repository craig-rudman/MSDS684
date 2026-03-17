"""MDP agents: abstract base class and a random-policy agent."""

from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym
import numpy as np


class Agent(ABC):
    """
    Abstract base class for agents that interact with Gymnasium environments.

    Subclasses must implement select_action(). Override update() and
    reset() if the agent maintains internal state across steps.
    """

    def __init__(self, action_space: gym.Space, seed: Optional[int] = None):
        self.action_space = action_space
        self.np_random = np.random.default_rng(seed)
        self.agent_name: str = type(self).__name__

    @abstractmethod
    def select_action(self, observation) -> int:
        """Choose an action given the current observation."""
        ...

    def update(self, observation, action: int, reward: float,
               next_observation, terminated: bool, truncated: bool):
        """Update internal state after a transition (no-op by default)."""

    def reset(self):
        """Reset agent state for a new episode (no-op by default)."""


class RandomAgent(Agent):
    """
    Agent that selects actions uniformly at random.

    Serves as a performance baseline for any environment.
    """

    def __init__(self, action_space: gym.Space, seed: Optional[int] = None):
        super().__init__(action_space, seed=seed)
        self.agent_name = "Random"

    def select_action(self, observation) -> int:
        return int(self.action_space.sample())
