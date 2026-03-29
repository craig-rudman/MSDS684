import pytest
import os
import tempfile
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from visualizer import Visualizer


class TestVisualizerValueSurface:
    """Tests for 3D value surface plots."""

    def setup_method(self):
        self.viz = Visualizer()
        self.value_function = {}
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [True, False]:
                    self.value_function[(player_sum, dealer_card, usable_ace)] = 0.0

    def test_returns_figure(self):
        fig = self.viz.plot_value_surface(self.value_function, "Test")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_usable_ace_plot(self):
        fig = self.viz.plot_value_surface(self.value_function, "Usable Ace",
                                          usable_ace=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_no_usable_ace_plot(self):
        fig = self.viz.plot_value_surface(self.value_function, "No Usable Ace",
                                          usable_ace=False)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestVisualizerLearningCurve:
    """Tests for learning curve plots."""

    def setup_method(self):
        self.viz = Visualizer()

    def test_returns_figure(self):
        rewards = [1.0, -1.0, 0.0, 1.0, -1.0] * 100
        fig = self.viz.plot_learning_curve(rewards, window_size=50)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_episode(self):
        fig = self.viz.plot_learning_curve([1.0], window_size=1)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_window_larger_than_data(self):
        fig = self.viz.plot_learning_curve([1.0, -1.0], window_size=100)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestVisualizerBakeoff:
    """Tests for bake-off comparison chart."""

    def setup_method(self):
        self.viz = Visualizer()

    def test_returns_figure(self):
        results = {"NaiveRandom": -0.3, "BasicStrategy": -0.05, "MC Agent": -0.02}
        fig = self.viz.plot_bakeoff(results)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_handles_single_agent(self):
        results = {"MC Agent": -0.02}
        fig = self.viz.plot_bakeoff(results)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestVisualizerSavePlots:
    """Tests for saving plots to disk."""

    def setup_method(self):
        self.viz = Visualizer()

    def test_save_creates_file(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_plot.png")
            self.viz.save_plot(fig, path)
            assert os.path.exists(path)
        plt.close(fig)

    def test_save_creates_subdirectory(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "test_plot.png")
            self.viz.save_plot(fig, path)
            assert os.path.exists(path)
        plt.close(fig)
