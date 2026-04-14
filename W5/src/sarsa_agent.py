from __future__ import annotations

import numpy as np

from src.tile_coder import TileCoder


class SarsaAgent:
    """Semi-gradient SARSA agent with tile-coded linear function approximation."""

    def __init__(
        self,
        tile_coder: TileCoder,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon: float = 0.1,
    ) -> None:
        self.tile_coder = tile_coder
        self.n_actions = n_actions
        self.alpha = alpha
        self.alpha_eff = alpha / tile_coder.n_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = np.zeros((n_actions, tile_coder.num_features))

    def q_value(self, state: list[float], action: int) -> float:
        """Return the estimated action-value Q(state, action)."""
        tiles = self.tile_coder.get_tiles(state)
        return float(np.sum(self.weights[action][tiles]))

    def greedy_action(self, state: list[float]) -> int:
        """Return the action with the highest estimated value for the given state."""
        return int(np.argmax([self.q_value(state, a) for a in range(self.n_actions)]))

    def select_action(self, state: list[float]) -> int:
        """Return an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return self.greedy_action(state)

    def update(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        next_action: int,
        done: bool,
    ) -> None:
        """Perform a semi-gradient SARSA weight update."""
        next_q = 0.0 if done else self.q_value(next_state, next_action)
        target = reward + self.gamma * next_q
        td_error = target - self.q_value(state, action)
        tiles = self.tile_coder.get_tiles(state)
        self.weights[action][tiles] += self.alpha_eff * td_error

    def reset_weights(self) -> None:
        """Reset all weights to zero."""
        self.weights = np.zeros((self.n_actions, self.tile_coder.num_features))

    def save(self, path: str) -> None:
        """Persist weights and configuration to a .npz file."""
        np.savez(
            path,
            weights=self.weights,
            n_tilings=self.tile_coder.n_tilings,
            tiles_per_dim=np.array(self.tile_coder.tiles_per_dim),
            state_bounds=np.array(self.tile_coder.state_bounds),
            n_actions=self.n_actions,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )

    @classmethod
    def load(cls, path: str) -> SarsaAgent:
        """Reconstruct a SarsaAgent from a .npz file produced by save()."""
        data = np.load(path)
        tc = TileCoder(
            n_tilings=int(data['n_tilings']),
            tiles_per_dim=data['tiles_per_dim'].tolist(),
            state_bounds=[tuple(row) for row in data['state_bounds']],
        )
        agent = cls(
            tc,
            n_actions=int(data['n_actions']),
            alpha=float(data['alpha']),
            gamma=float(data['gamma']),
            epsilon=float(data['epsilon']),
        )
        agent.weights = data['weights']
        return agent
