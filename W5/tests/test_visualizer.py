import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing
import matplotlib.pyplot as plt

from src.sarsa_agent import SarsaAgent
from src.tile_coder import TileCoder
from src.visualizer import Visualizer


@pytest.fixture
def visualizer():
    agent = SarsaAgent(TileCoder(), n_actions=3)
    return Visualizer(agent)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')


class TestVisualizerConstruction:
    def test_stores_agent(self):
        agent = SarsaAgent(TileCoder(), n_actions=3)
        v = Visualizer(agent)
        assert v.agent is agent


class TestVisualizerPlotValueFunction:
    def test_runs_without_error(self, visualizer):
        visualizer.plot_value_function()

    def test_returns_figure(self, visualizer):
        fig = visualizer.plot_value_function()
        assert isinstance(fig, plt.Figure)


class TestVisualizerPlotPolicy:
    def test_runs_without_error(self, visualizer):
        visualizer.plot_policy()

    def test_returns_figure(self, visualizer):
        fig = visualizer.plot_policy()
        assert isinstance(fig, plt.Figure)


class TestVisualizerPlotTrajectories:
    def test_runs_without_error(self, visualizer):
        trajectories = [
            [[-0.5, 0.0], [-0.48, 0.01], [-0.45, 0.02]],
        ]
        visualizer.plot_trajectories(trajectories)

    def test_returns_figure(self, visualizer):
        trajectories = [
            [[-0.5, 0.0], [-0.48, 0.01], [-0.45, 0.02]],
        ]
        fig = visualizer.plot_trajectories(trajectories)
        assert isinstance(fig, plt.Figure)
