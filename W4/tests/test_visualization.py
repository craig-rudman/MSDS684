import os
import numpy as np
import pytest
from visualization import Visualizer
from experiment import ExperimentResult, ExperimentConfig
from agents import SARSAAgent
from schedules import ConstantSchedule


def make_dummy_result():
    config = ExperimentConfig(
        label='sarsa',
        agent_class=SARSAAgent,
        alpha=0.1,
        epsilon_schedule=ConstantSchedule(0.1),
        n_seeds=3,
        n_episodes=10,
    )
    reward_matrix = np.full((3, 10), -20.0)
    return ExperimentResult(config, reward_matrix)


@pytest.fixture
def visualizer(env_manager):
    return Visualizer(env_manager)


@pytest.fixture
def dummy_qtable():
    return np.zeros((48, 4))


class TestVisualizer:
    def test_plot_learning_curves_saves_file(self, visualizer, tmp_path):
        results = [make_dummy_result()]
        visualizer.plot_learning_curves(results, output_dir=str(tmp_path))
        files = list(tmp_path.iterdir())
        assert len(files) == 1

    def test_plot_policy_arrows_saves_file(self, visualizer, dummy_qtable, tmp_path):
        visualizer.plot_policy_arrows(dummy_qtable, label='sarsa', output_dir=str(tmp_path))
        files = list(tmp_path.iterdir())
        assert len(files) == 1

    def test_plot_value_heatmap_saves_file(self, visualizer, dummy_qtable, tmp_path):
        visualizer.plot_value_heatmap(dummy_qtable, label='sarsa', output_dir=str(tmp_path))
        files = list(tmp_path.iterdir())
        assert len(files) == 1

    def test_plot_trajectory_saves_file(self, visualizer, dummy_qtable, tmp_path):
        visualizer.plot_trajectory(dummy_qtable, label='sarsa', output_dir=str(tmp_path))
        files = list(tmp_path.iterdir())
        assert len(files) == 1
