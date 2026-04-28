from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, state: int) -> int:
        ...

    @abstractmethod
    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ) -> dict:
        ...
