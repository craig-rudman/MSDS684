"""Run episodic simulations and collect per-episode metrics."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym

from .agents import Agent


@dataclass
class EpisodeResult:
    """
    Stores results from multiple independent episodes.

    Each array has length n_episodes.
    """

    agent_name: str
    env_id: str
    _returns: np.ndarray
    _lengths: np.ndarray
    _successes: np.ndarray

    def returns(self) -> np.ndarray:
        """Total return for each episode."""
        return self._returns

    def lengths(self) -> np.ndarray:
        """Number of steps in each episode."""
        return self._lengths

    def successes(self) -> np.ndarray:
        """Boolean array indicating whether each episode succeeded."""
        return self._successes

    def summary(self) -> dict:
        """Return aggregate metrics across all episodes."""
        return {
            "agent_name": self.agent_name,
            "env_id": self.env_id,
            "n_episodes": len(self._returns),
            "mean_return": float(self._returns.mean()),
            "std_return": float(self._returns.std()),
            "mean_length": float(self._lengths.mean()),
            "success_rate": float(self._successes.mean()) * 100.0,
        }


def run_episodes(
    env: gym.Env,
    agent: Agent,
    n_episodes: int = 1000,
    agent_name: Optional[str] = None,
    success_reward: float = 0.0,
) -> EpisodeResult:
    """
    Run multiple episodes of an agent in an environment.

    Args:
        env: A Gymnasium environment.
        agent: An Agent instance.
        n_episodes: Number of episodes to run.
        agent_name: Label for this agent (defaults to agent.agent_name).
        success_reward: Minimum final reward to count as a success.
                        For FrozenLake, goal gives reward 1.0 so
                        success_reward=0.0 works. Adjust per environment.

    Returns:
        EpisodeResult with per-episode metrics.
    """
    all_returns = np.zeros(n_episodes)
    all_lengths = np.zeros(n_episodes, dtype=int)
    all_successes = np.zeros(n_episodes, dtype=bool)

    env_id = env.spec.id if env.spec else type(env).__name__

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()

        episode_return = 0.0
        steps = 0
        last_reward = 0.0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated, truncated)

            episode_return += reward
            last_reward = reward
            steps += 1
            obs = next_obs

            if terminated or truncated:
                break

        all_returns[ep] = episode_return
        all_lengths[ep] = steps
        all_successes[ep] = last_reward > success_reward

    name = agent_name if agent_name is not None else agent.agent_name
    return EpisodeResult(
        agent_name=name,
        env_id=env_id,
        _returns=all_returns,
        _lengths=all_lengths,
        _successes=all_successes,
    )
