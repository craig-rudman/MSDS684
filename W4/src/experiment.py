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
    def __init__(self, config: ExperimentConfig, reward_matrix: np.ndarray, q_table: np.ndarray = None):
        self.config = config
        self.reward_matrix = reward_matrix
        self.q_table = q_table  # trained Q-table from final seed

    def final_performance(self) -> float:
        # Mean reward over the last 10% of episodes across all seeds
        n = max(1, self.reward_matrix.shape[1] // 10)
        return float(np.mean(self.reward_matrix[:, -n:]))

    def learning_speed(self, threshold: float = -75.0, smooth_window: int = 10) -> int:
        # First episode where smoothed mean reward crosses threshold.
        # smooth_window should match the window used in plot_learning_curves so that
        # the threshold crossing is measured on the same signal shown in the plot.
        # Meaningful as a comparison only when all results in a suite share the same
        # initial conditions (alpha, epsilon, environment); otherwise the threshold
        # may be trivial for some configs and unreachable for others.
        n_episodes = self.reward_matrix.shape[1]
        mean_per_episode = np.mean(self.reward_matrix, axis=0)
        window = max(1, smooth_window)
        smoothed = np.convolve(mean_per_episode, np.ones(window) / window, mode='valid')
        crossed = np.where(smoothed >= threshold)[0]
        return int(crossed[0]) if len(crossed) > 0 else n_episodes

    def sample_efficiency(self) -> float:
        # Mean reward across all episodes and seeds -- normalized AUC
        return float(np.mean(self.reward_matrix))

    def performance_index(self, w_final: float = 0.5, w_speed: float = 0.2, w_efficiency: float = 0.3,
                          ranges: dict = None) -> float:
        # Normalized weighted composite: higher is better
        # ranges: dict with keys 'final', 'speed', 'efficiency', each a (min, max) tuple
        fp = self.final_performance()
        ls = self.learning_speed()
        se = self.sample_efficiency()
        if ranges:
            def norm(val, lo, hi):
                return (val - lo) / (hi - lo) if hi != lo else 0.0
            fp_norm = norm(fp, *ranges['final'])
            ls_norm = 1.0 - norm(ls, *ranges['speed'])   # lower episode count is better
            se_norm = norm(se, *ranges['efficiency'])
        else:
            fp_norm, ls_norm, se_norm = fp, float(-ls), se
        return float(w_final * fp_norm + w_speed * ls_norm + w_efficiency * se_norm)

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
            self.results.append(ExperimentResult(config, matrix, q_table=agent.Q.copy()))
        return self.results

    def summarize(self, output_path: str = None, weights: dict = None,
                  sort_by: str = None, ascending: bool = True,
                  learning_speed_threshold: float = -75.0) -> pd.DataFrame:
        # weights: {'final_performance': float, 'learning_speed': float, 'sample_efficiency': float}
        rows = [
            {
                'label':              r.config.label,
                'final_performance':  r.final_performance(),
                'learning_speed':     r.learning_speed(threshold=learning_speed_threshold),
                'sample_efficiency':  r.sample_efficiency(),
            }
            for r in self.results
        ]
        df = pd.DataFrame(rows)
        if weights is not None:
            w_final      = weights.get('final_performance', 0.5)
            w_speed      = weights.get('learning_speed', 0.2)
            w_efficiency = weights.get('sample_efficiency', 0.3)
            ranges = {
                'final':      (df['final_performance'].min(), df['final_performance'].max()),
                'speed':      (df['learning_speed'].min(),    df['learning_speed'].max()),
                'efficiency': (df['sample_efficiency'].min(), df['sample_efficiency'].max()),
            }
            df['performance_index'] = [
                r.performance_index(w_final, w_speed, w_efficiency, ranges)
                for r in self.results
            ]
        if sort_by is not None:
            if sort_by not in df.columns:
                raise ValueError(f"sort_by column '{sort_by}' not in DataFrame; available: {list(df.columns)}")
            df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        if output_path:
            df.to_csv(output_path, index=False)
        return df
