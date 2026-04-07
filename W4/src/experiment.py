import os
from dataclasses import dataclass
from typing import Type
import numpy as np
import pandas as pd
from agents import TDAgent
from schedules import EpsilonSchedule
from environment import EnvironmentManager
from runner import EpisodeRunner


@dataclass
class ExperimentConfig:
    label: str
    agent_class: Type[TDAgent]
    alpha: float
    epsilon_schedule: EpsilonSchedule
    n_seeds: int
    n_episodes: int


class ExperimentResult:
    def __init__(self, config: ExperimentConfig, reward_matrix: np.ndarray):
        self.config = config
        self.reward_matrix = reward_matrix

    def final_performance(self) -> float:
        # Mean reward over the last 10% of episodes across all seeds
        n = max(1, self.reward_matrix.shape[1] // 10)
        return float(np.mean(self.reward_matrix[:, -n:]))

    def learning_speed(self) -> int:
        # First episode where smoothed mean reward crosses 90% of the start-to-finish improvement
        n_episodes = self.reward_matrix.shape[1]
        mean_per_episode = np.mean(self.reward_matrix, axis=0)
        start = mean_per_episode[0]
        threshold = start + 0.9 * (self.final_performance() - start)
        window = max(1, n_episodes // 20)
        smoothed = np.convolve(mean_per_episode, np.ones(window) / window, mode='valid')
        crossed = np.where(smoothed >= threshold)[0]
        return int(crossed[0]) if len(crossed) > 0 else n_episodes

    def sample_efficiency(self) -> float:
        # Mean reward across all episodes and seeds -- normalized AUC
        return float(np.mean(self.reward_matrix))

    def save(self, output_dir: str) -> None:
        path = os.path.join(output_dir, f"{self.config.label}.npy")
        np.save(path, self.reward_matrix)


class ExperimentSuite:
    def __init__(self, configs: list[ExperimentConfig], env_manager: EnvironmentManager):
        self.configs = configs
        self.env_manager = env_manager
        self.results = []

    def run(self) -> list[ExperimentResult]:
        self.results = []
        for config in self.configs:
            agent = config.agent_class(
                n_states=self.env_manager.n_states,
                n_actions=self.env_manager.n_actions,
                alpha=config.alpha,
                gamma=1.0,
                schedule=config.epsilon_schedule,
            )
            runner = EpisodeRunner(agent, self.env_manager)
            matrix = runner.run_experiment(config.n_seeds, config.n_episodes)
            self.results.append(ExperimentResult(config, matrix))
        return self.results

    def summarize(self, output_path: str = None) -> pd.DataFrame:
        rows = [
            {
                'label':              r.config.label,
                'final_performance':  r.final_performance(),
                'learning_speed':     r.learning_speed(),
                'sample_efficiency':  r.sample_efficiency(),
            }
            for r in self.results
        ]
        df = pd.DataFrame(rows)
        if output_path:
            df.to_csv(output_path, index=False)
        return df
