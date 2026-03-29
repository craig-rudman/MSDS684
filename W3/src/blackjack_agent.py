import numpy as np
from collections import defaultdict


class BlackjackAgent:
    """First-visit Monte Carlo control agent with epsilon-soft policy."""

    def __init__(self, epsilon: float = 0.1, discount_factor: float = 1.0,
                 decay_schedule=None):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay_schedule = decay_schedule
        self.q_values = defaultdict(lambda: np.zeros(2))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

    def select_action(self, state: tuple) -> int:
        n_actions = 2
        greedy_action = int(np.argmax(self.q_values[state]))
        probabilities = np.ones(n_actions) * (self.epsilon / n_actions)
        probabilities[greedy_action] += 1.0 - self.epsilon
        return int(np.random.choice(n_actions, p=probabilities))

    def update(self, episode: list) -> None:
        """First-visit MC update from a complete episode trajectory.

        Args:
            episode: list of (state, action, reward) tuples.
        """
        visited = set()
        G = 0.0
        for state, action, reward in reversed(episode):
            G = self.discount_factor * G + reward
            sa_pair = (state, action)
            if sa_pair not in visited:
                visited.add(sa_pair)
                self.returns_sum[sa_pair] += G
                self.returns_count[sa_pair] += 1.0
                self.q_values[state][action] = (
                    self.returns_sum[sa_pair] / self.returns_count[sa_pair]
                )

    def decay_epsilon(self) -> None:
        """Decay epsilon using the injected schedule, if provided."""
        if self.decay_schedule is not None:
            self.epsilon = self.decay_schedule(self.epsilon)

    def get_policy(self) -> dict:
        """Return greedy policy derived from Q-values."""
        return {state: int(np.argmax(actions)) for state, actions in self.q_values.items()}

    def get_value_function(self) -> dict:
        """Return state-value function (max Q over actions)."""
        return {state: float(np.max(actions)) for state, actions in self.q_values.items()}
