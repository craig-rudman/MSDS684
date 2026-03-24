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
        env = gym.make(
            "FrozenLake-v1", map_name=map_name, is_slippery=is_slippery
        )
        uw = env.unwrapped
        adapter = _FrozenLakeAdapter(uw)

        name = f"frozen_lake_{map_name}_{'slippery' if is_slippery else 'deterministic'}"
        results = self.run_gridworld(name, adapter)
        env.close()
        return results

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


class _FrozenLakeAdapter:
    """Thin adapter exposing Gymnasium FrozenLake with the same interface as GridWorld.

    The DP solvers expect ``n_states``, ``n_actions``, ``P``,
    ``terminal_states``, and ``rc_to_state``.  FrozenLake already stores
    its transition model in the compatible ``(prob, s', reward, done)``
    format, so this adapter simply bridges the attribute names.
    """

    def __init__(self, unwrapped_env: gym.Env) -> None:
        self.n_states = unwrapped_env.observation_space.n
        self.n_actions = unwrapped_env.action_space.n
        self.P = unwrapped_env.P

        desc = unwrapped_env.desc
        ncol = desc.shape[1]
        self.terminal_states = [
            (r, c)
            for r in range(desc.shape[0])
            for c in range(ncol)
            if desc[r][c] in (b"H", b"G")
        ]
        self._ncol = ncol

    def rc_to_state(self, r: int, c: int) -> int:
        return r * self._ncol + c


if __name__ == "__main__":
    ExperimentRunner.cli()
