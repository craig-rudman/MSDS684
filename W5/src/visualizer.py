from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Produces plots of the learned value function, policy, and trajectories."""

    GOAL_POSITION = 0.5

    def __init__(self, agent, n_grid: int = 100) -> None:
        self.agent = agent
        self.n_grid = n_grid
        self._pos_grid = None
        self._vel_grid = None
        self._value_grid = None
        self._policy_grid = None

    def _build_grids(self) -> None:
        """Compute value and policy grids over the state space. Results are cached."""
        if self._value_grid is not None:
            return
        positions = np.linspace(-1.2, 0.6, self.n_grid)
        velocities = np.linspace(-0.07, 0.07, self.n_grid)
        self._pos_grid, self._vel_grid = np.meshgrid(positions, velocities)
        self._value_grid = np.zeros((self.n_grid, self.n_grid))
        self._policy_grid = np.zeros((self.n_grid, self.n_grid), dtype=int)
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                state = [self._pos_grid[i, j], self._vel_grid[i, j]]
                qs = [self.agent.q_value(state, a) for a in range(self.agent.n_actions)]
                self._value_grid[i, j] = max(qs)
                self._policy_grid[i, j] = int(np.argmax(qs))

    def plot_value_function(self, save_path=None) -> plt.Figure:
        """Show the learned value function as a contour plot over the 2D state space."""
        self._build_grids()
        fig, ax = plt.subplots(figsize=(8, 5))
        cf = ax.contourf(self._pos_grid, self._vel_grid, self._value_grid, levels=30, cmap='RdYlGn')
        fig.colorbar(cf, ax=ax, label='Max Q(s,a)')
        ax.axvline(self.GOAL_POSITION, color='black', lw=2, ls='--', label='Goal (pos=0.5)')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title('Learned Value Function')
        ax.legend()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig

    def plot_policy(self, save_path=None) -> plt.Figure:
        """Show the greedy action chosen by the agent across the 2D state space."""
        self._build_grids()
        fig, ax = plt.subplots(figsize=(8, 5))
        action_symbols = {0: '<', 1: '.', 2: '>'}
        action_colors = {0: '#AA7733', 1: '#777777', 2: '#3377AA'}
        for action in range(3):
            mask = self._policy_grid == action
            ax.scatter( self._pos_grid[mask], 
                       self._vel_grid[mask], 
                       c=action_colors[action],
                       marker=action_symbols[action],
                       s=8,
                       alpha=0.7,
                       label=f"Action {action}: {['push left','no push','push right'][action]}")
        ax.axvline(0.5, color='black', lw=2, ls='--', label='Goal')
        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Velocity', fontsize=10)
        ax.set_title('Learned Policy', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, markerscale=2)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig

    def plot_trajectories(self, trajectories: list[list[list[float]]], save_path=None) -> plt.Figure:
        """Show trajectories overlaid on the value function heatmap."""
        raise NotImplementedError
