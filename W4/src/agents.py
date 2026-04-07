from abc import ABC, abstractmethod
import numpy as np
from schedules import EpsilonSchedule


class TDAgent(ABC):
    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float, schedule: EpsilonSchedule):
        self.n_states = n_states                                # number of states in the environment
        self.n_actions = n_actions                              # number of actions in the environment
        self.alpha = alpha                                      # learning step factor
        self.gamma = gamma                                      # discount factor
        self.schedule = schedule                                # epsilon decay schedule
        self.Q = np.zeros((n_states, n_actions), dtype=float)   # Q-table: estimated value for each (s,a)

    def select_action(self, state: int) -> int:
        ''' Implements epsilon-greedy policy for managing the exploration/exploitation decision.'''
        if np.random.random() < self.schedule.value:
            # explore
            return np.random.choice(self.n_actions)
        # exploit
        return np.argmax(self.Q[state])

    @abstractmethod
    def _bootstrap(self, s_next: int, a_next: int, terminated: bool) -> float:
        pass

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, terminated: bool) -> None:
        target = r + self._bootstrap(s_next, a_next, terminated)
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def reset_qtable(self) -> None:
        self.Q[:] = 0.0

    def decay_epsilon(self) -> None:
        self.schedule.step()


class SARSAAgent(TDAgent):
    def _bootstrap(self, s_next: int, a_next: int, terminated: bool) -> float:
        # on-policy: uses the actual next action in this trajectory
        return 0.0 if terminated else self.gamma * self.Q[s_next, a_next]


class QLearningAgent(TDAgent):
    def _bootstrap(self, s_next: int, a_next: int, terminated: bool) -> float:
        # off-policy: uses the optimal next action across all possible trajectories
        return 0.0 if terminated else self.gamma * np.max(self.Q[s_next])
