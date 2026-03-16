"""Run bandit simulations and collect per-step metrics."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .bandit_env import MultiArmedBanditEnv
from .agents import BanditAgent


@dataclass
class SimulationResult:
    """
    Stores results from multiple independent runs.

    Raw arrays have shape (n_runs, n_steps). Accessor methods
    return per-step averages across runs.
    """

    agent_name: str
    _rewards: np.ndarray
    _optimal_actions: np.ndarray
    _cumulative_regret: np.ndarray

    def mean_reward(self) -> np.ndarray:
        """Per-step average reward across all runs."""
        return self._rewards.mean(axis=0)

    def optimal_action_pct(self) -> np.ndarray:
        """Per-step percentage of runs that chose the optimal arm."""
        return self._optimal_actions.mean(axis=0) * 100.0

    def cumulative_regret(self) -> np.ndarray:
        """Per-step average cumulative regret across all runs."""
        return self._cumulative_regret.mean(axis=0)

    def summary(self) -> dict:
        """Return a dict of final metrics."""
        return {
            "agent_name": self.agent_name,
            "final_mean_reward": float(self.mean_reward()[-1]),
            "final_optimal_pct": float(self.optimal_action_pct()[-1]),
            "final_cumulative_regret": float(self.cumulative_regret()[-1]),
        }


def run_simulation(
    env: MultiArmedBanditEnv,
    agent: BanditAgent,
    n_runs: int = 100,
    agent_name: Optional[str] = None,
) -> SimulationResult:
    """
    Run multiple independent episodes of a bandit simulation.

    On each run, env.reset() re-samples arm means (if randomize=True),
    the agent is reset, and max_steps actions are taken.

    Args:
        env: The bandit environment.
        agent: The bandit agent.
        n_runs: Number of independent runs to average over.
        agent_name: Label for this agent (defaults to class name).

    Returns:
        SimulationResult with per-step metrics across all runs.
    """
    n_steps = env.max_steps
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps))
    all_regret = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        env.reset()
        agent.reset()

        best_mean = env.true_means().max()
        cum_regret = 0.0

        for step in range(n_steps):
            action = agent.select_action()
            _, reward, terminated, _, info = env.step(action)
            agent.update(action, reward)

            all_rewards[run, step] = reward
            all_optimal[run, step] = float(info["optimal_action"])
            cum_regret += best_mean - reward
            all_regret[run, step] = cum_regret

            if terminated:
                break

    name = agent_name if agent_name is not None else type(agent).__name__
    return SimulationResult(
        agent_name=name,
        _rewards=all_rewards,
        _optimal_actions=all_optimal,
        _cumulative_regret=all_regret,
    )
