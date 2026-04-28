import numpy as np

from agent import Agent


class QLearningAgent(Agent):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        seed: int = 0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))
        self.rng = np.random.default_rng(seed)

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        max_value = self.Q[state].max()
        tied_actions = np.flatnonzero(self.Q[state] == max_value)
        return int(self.rng.choice(tied_actions))

    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ) -> dict:
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state].max()
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        return {}
