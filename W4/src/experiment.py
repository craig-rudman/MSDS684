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
        pass

    def final_performance(self) -> float:
        pass

    def learning_speed(self) -> float:
        pass

    def sample_efficiency(self) -> float:
        pass

    def save(self, output_dir: str) -> None:
        pass


class ExperimentSuite:
    def __init__(self, configs: list[ExperimentConfig], env_manager: EnvironmentManager):
        pass

    def run(self) -> list[ExperimentResult]:
        pass

    def summarize(self, output_path: str = None) -> pd.DataFrame:
        pass
