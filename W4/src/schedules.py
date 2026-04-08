from abc import ABC, abstractmethod


class EpsilonSchedule(ABC):
    @property
    @abstractmethod
    def value(self) -> float:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class ConstantSchedule(EpsilonSchedule):
    def __init__(self, epsilon: float):
        self._epsilon = epsilon

    @property
    def value(self) -> float:
        return self._epsilon

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass


class LinearDecaySchedule(EpsilonSchedule):
    def __init__(self, epsilon_start: float, epsilon_end: float, n_episodes: int):
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._n_episodes = n_episodes
        self._episode = 0

    @property
    def value(self) -> float:
        fraction = min(self._episode / self._n_episodes, 1.0)
        return self._epsilon_start + fraction * (self._epsilon_end - self._epsilon_start)

    def step(self) -> None:
        self._episode += 1

    def reset(self) -> None:
        self._episode = 0


class ExponentialDecaySchedule(EpsilonSchedule):
    def __init__(self, epsilon_start: float, epsilon_end: float, n_episodes: int):
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._n_episodes = n_episodes
        self._decay_rate = (epsilon_end / epsilon_start) ** (1.0 / n_episodes)
        self._episode = 0

    @property
    def value(self) -> float:
        return max(self._epsilon_start * (self._decay_rate ** self._episode), self._epsilon_end)

    def step(self) -> None:
        self._episode += 1

    def reset(self) -> None:
        self._episode = 0
