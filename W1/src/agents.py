"""Bandit agents: abstract base class and concrete strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BanditAgent(ABC):
    """
    Abstract base class for bandit agents.

    Tracks per-arm pull counts and incremental value estimates.
    Subclasses must implement select_action().
    """

    def __init__(self, k: int, seed: Optional[int] = None):
        self.k = k
        self.np_random = np.random.default_rng(seed)
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0

    @abstractmethod
    def select_action(self) -> int:
        """Choose an arm to pull."""
        ...

    def update(self, action: int, reward: float):
        """Update value estimate for the given arm using incremental mean."""
        self.t += 1
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n

    def reset(self):
        """Reset all estimates for a new run."""
        self.counts = np.zeros(self.k)
        self.values = np.zeros(self.k)
        self.t = 0


class EpsilonGreedyAgent(BanditAgent):
    """
    Epsilon-greedy agent.

    With probability epsilon, selects a random arm (explore).
    Otherwise, selects the arm with the highest estimated value (exploit).
    """

    def __init__(self, k: int, epsilon: float = 0.1, **kwargs):
        super().__init__(k, **kwargs)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if self.np_random.random() < self.epsilon:
            return int(self.np_random.integers(self.k))
        return int(np.argmax(self.values))


class UCBAgent(BanditAgent):
    """
    Upper Confidence Bound (UCB1) agent.

    Selects the arm maximizing: Q(a) + c * sqrt(ln(t) / N(a)).
    Plays each arm once before applying the UCB formula.
    """

    def __init__(self, k: int, c: float = 2.0, **kwargs):
        super().__init__(k, **kwargs)
        self.c = c

    def select_action(self) -> int:
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.t) / self.counts
        )
        return int(np.argmax(ucb_values))
