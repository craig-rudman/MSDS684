import time

import numpy as np

from q_learning_agent import QLearningAgent
from tabular_model import TabularModel


class DynaQAgent(QLearningAgent):
    def __init__(self, num_states, num_actions, alpha, epsilon, gamma, n_planning, seed=0):
        if n_planning < 1:
            raise ValueError(f"n_planning must be >= 1, got {n_planning}")
        super().__init__(num_states, num_actions, alpha, epsilon, gamma, seed)
        self.n_planning = n_planning
        self.model = TabularModel()

    def learn(self, state, action, reward, next_state, terminated, truncated):
        super().learn(state, action, reward, next_state, terminated, truncated)
        self.model.update(state, action, reward, next_state)

        start = time.perf_counter()
        for _ in range(self.n_planning):
            s, a, r, s_next = self.model.sample(self.rng)
            target = r + self.gamma * self.Q[s_next].max()
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        wall_time_planning = time.perf_counter() - start

        return {"n_planning_updates": self.n_planning, "wall_time_planning": wall_time_planning}
