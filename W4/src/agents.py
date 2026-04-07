from abc import ABC, abstractmethod
import numpy as np
from schedules import EpsilonSchedule


class TDAgent(ABC):
    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float, schedule: EpsilonSchedule):
        pass

    def select_action(self, state: int) -> int:
        pass

    @abstractmethod
    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, terminated: bool) -> None:
        pass

    def reset_qtable(self) -> None:
        pass

    def decay_epsilon(self) -> None:
        pass


class SARSAAgent(TDAgent):
    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, terminated: bool) -> None:
        pass


class QLearningAgent(TDAgent):
    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, terminated: bool) -> None:
        pass
