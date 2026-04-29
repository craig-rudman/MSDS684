import heapq
import time
from collections import defaultdict

from dyna_q_agent import DynaQAgent
from q_learning_agent import QLearningAgent
from tabular_model import TabularModel


class PrioritizedSweepingAgent(DynaQAgent):
    def __init__(self, num_states, num_actions, alpha, epsilon, gamma,
                 n_planning, theta=0.01, seed=0):
        if n_planning < 0:
            raise ValueError(f"n_planning must be >= 0, got {n_planning}")
        QLearningAgent.__init__(self, num_states, num_actions, alpha, epsilon, gamma, seed)
        self.n_planning = n_planning
        self.model = TabularModel()
        self.theta = theta
        self.priority_queue: list = []
        self.reverse_model: dict[int, set] = defaultdict(set)
        self._in_queue: set = set()

    def learn(self, state, action, reward, next_state, terminated, truncated):
        td_error = abs(
            reward + self.gamma * self.Q[next_state].max() - self.Q[state, action]
        )
        QLearningAgent.learn(self, state, action, reward, next_state, terminated, truncated)
        self.model.update(state, action, reward, next_state)
        self.reverse_model[next_state].add((state, action))

        if td_error > self.theta:
            self._push(td_error, state, action)

        start = time.perf_counter()
        n_updates = self._run_planning()
        wall_time_planning = time.perf_counter() - start

        return {"n_planning_updates": n_updates, "wall_time_planning": wall_time_planning}

    def _push(self, priority, s, a):
        heapq.heappush(self.priority_queue, (-priority, s, a))
        self._in_queue.add((s, a))


    def _run_planning(self) -> int:
        n_updates = 0
        for _ in range(self.n_planning):
            if not self.priority_queue:
                break
            _, s, a = heapq.heappop(self.priority_queue)
            self._in_queue.discard((s, a))
            r, s_next = self.model._transitions[(s, a)]
            target = r + self.gamma * self.Q[s_next].max()
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
            n_updates += 1

            for s_pred, a_pred in self.reverse_model.get(s, set()):
                if (s_pred, a_pred) not in self.model._transitions:
                    continue
                r_pred, _ = self.model._transitions[(s_pred, a_pred)]
                td_pred = abs(
                    r_pred + self.gamma * self.Q[s].max() - self.Q[s_pred, a_pred]
                )
                if td_pred > self.theta and (s_pred, a_pred) not in self._in_queue:
                    self._push(td_pred, s_pred, a_pred)

        return n_updates
