from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

    def plot_trajectories(self, trajectories: list[list[list[float]]], labels=None, colors=None, ax=None, title=None, save_path=None) -> plt.Figure:
        """Show trajectories overlaid on the value function heatmap.

        Each trajectory is a list of [position, velocity] steps. An open circle
        marks the start of each trajectory; a star marks the end.

        If ax is provided, the plot is drawn into that axes (caller manages layout).
        If ax is None, a new figure is created and returned.
        """
        self._build_grids()
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        cf = ax.contourf(self._pos_grid, self._vel_grid, self._value_grid, levels=30, cmap='RdYlGn', alpha=0.8)
        fig.colorbar(cf, ax=ax, label='Max Q(s,a)')
        ax.axvline(self.GOAL_POSITION, color='black', lw=2, ls='--', label='Goal')

        if colors is None:
            colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(trajectories)))
        for i, traj in enumerate(trajectories):
            positions = [s[0] for s in traj]
            velocities = [s[1] for s in traj]
            label = labels[i] if labels else f'Trajectory {i + 1}'
            ax.plot(positions, velocities, color=colors[i], lw=1.5, alpha=0.85, label=label)
            ax.scatter(positions[0], velocities[0], color=colors[i], marker='o', s=60, zorder=5)
            ax.scatter(positions[-1], velocities[-1], color=colors[i], marker='*', s=120, zorder=5)

        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_title(title or 'Sample Trajectories on Value Function')
        ax.legend(fontsize=7)

        if own_fig:
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path)
        return fig

    @staticmethod
    def _hill_height(xs: np.ndarray) -> np.ndarray:
        """MountainCar hill profile: matches Gymnasium's internal _height()."""
        return np.sin(3 * xs) * 0.45 + 0.55

    def animate_episode(self, trajectory: list[list[float]], interval: int = 40) -> animation.FuncAnimation:
        """Animate a single episode as two synchronized panels.

        Left panel: physical view -- the car (dot) moving along the hill profile.
        Right panel: state space view -- a point tracing (position, velocity) on
                     the value function heatmap.

        Parameters
        ----------
        trajectory:
            List of [position, velocity] steps recorded during the episode.
        interval:
            Milliseconds between frames.
        """
        self._build_grids()
        positions = [s[0] for s in trajectory]
        velocities = [s[1] for s in trajectory]

        # Hill curve for the physical panel
        hill_x = np.linspace(-1.2, 0.6, 300)
        hill_y = self._hill_height(hill_x)

        fig, (ax_hill, ax_state) = plt.subplots(1, 2, figsize=(12, 4))

        # --- Left: hill profile ---
        ax_hill.plot(hill_x, hill_y, 'k-', lw=2)
        ax_hill.axvline(self.GOAL_POSITION, color='green', lw=1.5, ls='--', label='Goal')
        car_dot, = ax_hill.plot([], [], 'o', color='crimson', ms=10, zorder=5)
        ax_hill.set_xlim(-1.2, 0.6)
        ax_hill.set_ylim(hill_y.min() - 0.05, hill_y.max() + 0.1)
        ax_hill.set_xlabel('Position')
        ax_hill.set_ylabel('Height')
        ax_hill.set_title('Physical View')
        ax_hill.legend(fontsize=8)
        step_text = ax_hill.text(0.02, 0.95, '', transform=ax_hill.transAxes,
                                 fontsize=9, va='top')

        # --- Right: state space ---
        ax_state.contourf(self._pos_grid, self._vel_grid, self._value_grid,
                          levels=30, cmap='RdYlGn', alpha=0.85)
        ax_state.axvline(self.GOAL_POSITION, color='black', lw=1.5, ls='--', label='Goal')
        trail_line, = ax_state.plot([], [], '-', color='white', lw=1, alpha=0.5)
        state_dot, = ax_state.plot([], [], 'o', color='white', ms=8, zorder=5)
        ax_state.set_xlim(-1.2, 0.6)
        ax_state.set_ylim(-0.07, 0.07)
        ax_state.set_xlabel('Position')
        ax_state.set_ylabel('Velocity')
        ax_state.set_title('State Space View')
        ax_state.legend(fontsize=8)

        plt.tight_layout()

        def init():
            car_dot.set_data([], [])
            trail_line.set_data([], [])
            state_dot.set_data([], [])
            step_text.set_text('')
            return car_dot, trail_line, state_dot, step_text

        def update(frame):
            pos, vel = positions[frame], velocities[frame]
            car_dot.set_data([pos], [self._hill_height(np.array([pos]))[0]])
            trail_line.set_data(positions[:frame + 1], velocities[:frame + 1])
            state_dot.set_data([pos], [vel])
            step_text.set_text(f'step {frame + 1} / {len(trajectory)}')
            return car_dot, trail_line, state_dot, step_text

        return animation.FuncAnimation(
            fig, update, frames=len(trajectory),
            init_func=init, interval=interval, blit=True
        )
