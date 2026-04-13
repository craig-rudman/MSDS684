from __future__ import annotations

from src.episode_runner import EpisodeRunner


class Trainer:
    """Runs a full training session across many episodes and collects step counts."""

    def __init__(self, env, agent, n_episodes: int, recorder=None, seed=None, start_episode: int = 0) -> None:
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.recorder = recorder
        self.seed = seed
        self.start_episode = start_episode
        self._runner = EpisodeRunner(env, agent, recorder, seed, start_episode)

    def train(self) -> list[int]:
        """Run n_episodes episodes and return the step count for each."""
        return [self._runner.run_episode() for _ in range(self.n_episodes)]
