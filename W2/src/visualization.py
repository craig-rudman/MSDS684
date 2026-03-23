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
        raise NotImplementedError

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
        raise NotImplementedError

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
