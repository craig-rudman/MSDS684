"""CLI entry point for running GridWorld experiments."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from .gridworld import GridWorld
from .dp_solvers import PolicyIteration, ValueIteration
from .visualization import GridWorldVisualizer


class ExperimentRunner:
    """Orchestrates experiments across environments and solvers.

    Provides helpers for running deterministic / stochastic GridWorld
    experiments as well as Gymnasium's FrozenLake-v1, collecting results,
    and producing comparison visualizations.
    """

    def __init__(self, gamma: float = 0.99, theta: float = 1e-8) -> None:
        self.gamma = gamma
        self.theta = theta
        self.results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # GridWorld experiments
    # ------------------------------------------------------------------

    def run_gridworld(
        self,
        name: str,
        env: GridWorld,
        modes: list[str] | None = None,
    ) -> dict[str, dict]:
        """Run all four DP variants on the given GridWorld.

        Parameters
        ----------
        name : str
            Label for this experiment (e.g. "deterministic_4x4").
        env : GridWorld
            A configured GridWorld instance.
        modes : list[str] | None
            Subset of ``["pi_sync", "pi_inplace", "vi_sync", "vi_inplace"]``
            to run. Defaults to all four.

        Returns
        -------
        dict[str, dict]
            Per-variant results keyed by variant name.
        """
        all_modes = ["pi_sync", "pi_inplace", "vi_sync", "vi_inplace"]
        if modes is None:
            modes = all_modes

        results: dict[str, dict] = {}
        for mode in modes:
            algo, sweep = mode.split("_", 1)
            if algo == "pi":
                solver = PolicyIteration(env, gamma=self.gamma, theta=self.theta)
            else:
                solver = ValueIteration(env, gamma=self.gamma, theta=self.theta)

            V, policy = solver.solve(mode=sweep)

            results[mode] = {
                "V": V,
                "policy": policy,
                "value_history": solver.value_history,
                "wall_clock_times": solver.wall_clock_times,
            }

        self.results[name] = results
        return results

    # ------------------------------------------------------------------
    # FrozenLake experiment
    # ------------------------------------------------------------------

    def run_frozen_lake(self, map_name: str = "4x4", is_slippery: bool = True) -> dict[str, dict]:
        """Run DP solvers on Gymnasium's FrozenLake-v1.

        Parameters
        ----------
        map_name : str
            FrozenLake map size (default "4x4").
        is_slippery : bool
            Whether the lake is slippery (default True).

        Returns
        -------
        dict[str, dict]
            Per-variant results.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_results(self, experiment_name: str) -> None:
        """Produce all plots for a completed experiment.

        Parameters
        ----------
        experiment_name : str
            Key into ``self.results``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def cli() -> None:
        """Parse command-line arguments and run experiments."""
        raise NotImplementedError


if __name__ == "__main__":
    ExperimentRunner.cli()
