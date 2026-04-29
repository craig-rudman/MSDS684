import math
import time

from dyna_q_agent import DynaQAgent


class DynaQPlusAgent(DynaQAgent):
    def __init__(self, num_states, num_actions, alpha, epsilon, gamma,
                 n_planning, kappa=0.001, seed=0):
        if n_planning < 0:
            raise ValueError(f"n_planning must be >= 0, got {n_planning}")
        from q_learning_agent import QLearningAgent
        QLearningAgent.__init__(self, num_states, num_actions, alpha, epsilon, gamma, seed)
        self.n_planning = n_planning
        from tabular_model import TabularModel
        self.model = TabularModel()
        self.kappa = kappa
        self.time_since: dict[tuple[int, int], int] = {}
        self._step_count = 0

    def learn(self, state, action, reward, next_state, terminated, truncated):
        self._step_count += 1

        # Age all tracked pairs, then reset the real (s,a) to 0
        for key in self.time_since:
            self.time_since[key] += 1
        self.time_since[(state, action)] = 0

        # Real Q-learning update (via grandparent to skip DynaQ's planning)
        from q_learning_agent import QLearningAgent
        QLearningAgent.learn(self, state, action, reward, next_state, terminated, truncated)
        self.model.update(state, action, reward, next_state)

        # Planning with exploration bonus
        start = time.perf_counter()
        for _ in range(self.n_planning):
            s, a, _r, _s_next = self.model.sample(self.rng)
            self._planning_step(s, a)
        wall_time_planning = time.perf_counter() - start

        return {"n_planning_updates": self.n_planning, "wall_time_planning": wall_time_planning}

    def _planning_step(self, s, a):
        r, s_next = self.model._transitions[(s, a)]
        tau = self.time_since.get((s, a), 0)
        bonus = self.kappa * math.sqrt(tau)
        target = (r + bonus) + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
