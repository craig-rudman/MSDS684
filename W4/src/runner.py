import numpy as np
from agents import TDAgent
from environment import EnvironmentManager


class EpisodeRunner:
    def __init__(self, agent: TDAgent, env_manager: EnvironmentManager):
        pass

    def run_episode(self) -> float:
        pass

    def run_experiment(self, n_seeds: int, n_episodes: int, base_seed: int = 0) -> np.ndarray:
        pass
