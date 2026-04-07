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
        pass

    @property
    def value(self) -> float:
        pass

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass


class ExponentialDecaySchedule(EpsilonSchedule):
    def __init__(self, epsilon_start: float, decay_rate: float):
        pass

    @property
    def value(self) -> float:
        pass

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass
