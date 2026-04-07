import numpy as np
import matplotlib.pyplot as plt
from environment import EnvironmentManager


class Visualizer:
    def __init__(self, env_manager: EnvironmentManager):
        pass

    def plot_learning_curves(self, results: list, output_dir: str = None) -> None:
        pass

    def plot_policy_arrows(self, q_table: np.ndarray, label: str = '', output_dir: str = None) -> None:
        pass

    def plot_value_heatmap(self, q_table: np.ndarray, label: str = '', output_dir: str = None) -> None:
        pass

    def plot_trajectory(self, q_table: np.ndarray, label: str = '', output_dir: str = None) -> None:
        pass
