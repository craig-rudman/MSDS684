"""Visualization utilities for GridWorld experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .gridworld import GridWorld


class GridWorldVisualizer:
    """Plotting helpers for value functions, policies, and convergence curves.

    Parameters
    ----------
    env : GridWorld
        The environment whose geometry is used for plotting.
    """

    def __init__(self, env: GridWorld) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # Value function heatmap
    # ------------------------------------------------------------------

    def plot_value_function(
        self,
        V: np.ndarray,
        title: str = "Value Function",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Render V(s) as a heatmap over the grid.

        Parameters
        ----------
        V : np.ndarray
            Value function of shape ``(n_states,)``.
        title : str
            Plot title.
        ax : plt.Axes or None
            Axes to draw on; creates a new figure if None.

        Returns
        -------
        plt.Axes
        """
        size = self.env.size
        V_grid = V.reshape(size, size)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(V_grid, cmap="viridis", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate each cell with its value
        for r in range(size):
            for c in range(size):
                ax.text(c, r, f"{V_grid[r, c]:.2f}",
                        ha="center", va="center", color="white", fontsize=9)

        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_title(title)
        return ax

    # ------------------------------------------------------------------
    # Policy arrow (quiver) plot
    # ------------------------------------------------------------------

    def plot_policy(
        self,
        policy: np.ndarray,
        title: str = "Policy",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Render the policy as arrows (quiver plot) over the grid.

        Parameters
        ----------
        policy : np.ndarray
            Policy of shape ``(n_states,)`` with action indices.
        title : str
            Plot title.
        ax : plt.Axes or None
            Axes to draw on; creates a new figure if None.

        Returns
        -------
        plt.Axes
        """
        size = self.env.size

        # Action-to-direction mapping: (dx, dy) in plot coordinates
        # Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
        # In imshow coordinates: row increases downward, col increases rightward
        # quiver uses (U, V) where U is horizontal (col) and V is vertical (row-up)
        action_to_dxdy = {
            0: (0, 1),    # UP: arrow points up (positive V)
            1: (1, 0),    # RIGHT: arrow points right (positive U)
            2: (0, -1),   # DOWN: arrow points down (negative V)
            3: (-1, 0),   # LEFT: arrow points left (negative U)
        }

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        # Build arrow arrays
        X, Y, U, V = [], [], [], []
        for s in range(self.env.n_states):
            r, c = self.env.state_to_rc(s)
            # Skip obstacles and terminal states
            if (r, c) in self.env.obstacles or (r, c) in self.env.terminal_states:
                continue
            dx, dy = action_to_dxdy[int(policy[s])]
            X.append(c)
            Y.append(r)
            U.append(dx)
            V.append(dy)

        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)  # invert y so row 0 is at top
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_aspect("equal")
        ax.set_title(title)

        if X:
            ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=2.5,
                      pivot="middle", color="steelblue", width=0.02)

        return ax

    # ------------------------------------------------------------------
    # Iteration history heatmaps
    # ------------------------------------------------------------------

    def plot_value_history(
        self,
        value_history: list[np.ndarray],
        interval: int = 1,
    ) -> plt.Figure:
        """Show value function heatmaps at selected iterations.

        Parameters
        ----------
        value_history : list[np.ndarray]
            List of V arrays recorded at each iteration.
        interval : int
            Plot every ``interval``-th snapshot.

        Returns
        -------
        plt.Figure
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Policy history arrow plots
    # ------------------------------------------------------------------

    def plot_policy_history(
        self,
        policy_history: list[np.ndarray],
        interval: int = 1,
    ) -> plt.Figure:
        """Show policy arrow plots at selected iterations.

        Parameters
        ----------
        policy_history : list[np.ndarray]
            List of policy arrays recorded at each iteration.
        interval : int
            Plot every ``interval``-th snapshot.

        Returns
        -------
        plt.Figure
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convergence curves
    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        results: dict[str, dict],
    ) -> plt.Figure:
        """Compare algorithms by iteration count and wall-clock time.

        Parameters
        ----------
        results : dict[str, dict]
            Mapping of algorithm name to a dict with keys
            ``"value_history"`` and ``"wall_clock_times"``.

        Returns
        -------
        plt.Figure
        """
        raise NotImplementedError
