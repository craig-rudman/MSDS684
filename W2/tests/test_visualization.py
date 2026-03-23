"""Tests for src/visualization.py -- GridWorldVisualizer."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/testing

import matplotlib.pyplot as plt
import numpy as np
import pytest


class TestPlotValueFunction:
    """Heatmap of V(s)."""

    def test_returns_axes(self, visualizer, env_4x4_deterministic):
        V = np.zeros(env_4x4_deterministic.n_states)
        ax = visualizer.plot_value_function(V)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_heatmap_shape(self, visualizer, env_4x4_deterministic):
        V = np.random.randn(env_4x4_deterministic.n_states)
        ax = visualizer.plot_value_function(V)
        # The heatmap image data should be (size, size)
        images = ax.get_images()
        assert len(images) > 0
        data = images[0].get_array()
        assert data.shape == (env_4x4_deterministic.size, env_4x4_deterministic.size)
        plt.close("all")


class TestPlotPolicy:
    """Quiver plot of the policy."""

    def test_returns_axes(self, visualizer, env_4x4_deterministic):
        policy = np.zeros(env_4x4_deterministic.n_states, dtype=int)
        ax = visualizer.plot_policy(policy)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_has_quiver(self, visualizer, env_4x4_deterministic):
        policy = np.zeros(env_4x4_deterministic.n_states, dtype=int)
        ax = visualizer.plot_policy(policy)
        quivers = [c for c in ax.get_children() if isinstance(c, matplotlib.quiver.Quiver)]
        assert len(quivers) > 0
        plt.close("all")


class TestPlotValueHistory:
    """Value function snapshots across iterations."""

    def test_returns_figure(self, visualizer, env_4x4_deterministic):
        history = [np.random.randn(env_4x4_deterministic.n_states) for _ in range(5)]
        fig = visualizer.plot_value_history(history)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotPolicyHistory:
    """Policy snapshots across iterations."""

    def test_returns_figure(self, visualizer, env_4x4_deterministic):
        history = [np.zeros(env_4x4_deterministic.n_states, dtype=int) for _ in range(5)]
        fig = visualizer.plot_policy_history(history)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotConvergence:
    """Convergence comparison curves."""

    def test_returns_figure(self, visualizer):
        results = {
            "algo_a": {
                "value_history": [np.zeros(16) for _ in range(3)],
                "wall_clock_times": [0.01, 0.02, 0.03],
            },
            "algo_b": {
                "value_history": [np.zeros(16) for _ in range(5)],
                "wall_clock_times": [0.01, 0.02, 0.03, 0.04, 0.05],
            },
        }
        fig = visualizer.plot_convergence(results)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
