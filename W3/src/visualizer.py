import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure


class Visualizer:
    """Visualization tools for Blackjack MC control results."""

    def plot_value_surface(self, value_function: dict, title: str,
                           usable_ace: bool = True) -> Figure:
        """3D surface plot of the state-value function.

        Args:
            value_function: dict mapping (player_sum, dealer_card, usable_ace) -> value.
            title: plot title.
            usable_ace: filter for usable ace states.

        Returns:
            matplotlib Figure.
        """
        player_range = np.arange(12, 22)
        dealer_range = np.arange(1, 11)
        X, Y = np.meshgrid(dealer_range, player_range)
        Z = np.zeros_like(X, dtype=float)

        for i, player_sum in enumerate(player_range):
            for j, dealer_card in enumerate(dealer_range):
                Z[i, j] = value_function.get((player_sum, dealer_card, usable_ace), 0.0)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel("Dealer Showing", labelpad=10)
        ax.set_ylabel("Player Sum", labelpad=10)
        ax.set_zlabel("")
        ax.text2D(-0.05, 0.5, "Value", transform=ax.transAxes,
                  rotation=90, va="center", fontsize=10)
        ax.set_title(title)
        ax.view_init(elev=25, azim=-135)
        return fig

    def _rolling_stats(self, rewards, window_size):
        """Compute rolling mean and 95% CI band for a reward series."""
        rewards_arr = np.array(rewards, dtype=float)
        effective_window = min(window_size, len(rewards_arr))
        if effective_window < 1:
            effective_window = 1
        cumsum = np.cumsum(np.insert(rewards_arr, 0, 0))
        mean = (cumsum[effective_window:] - cumsum[:-effective_window]) / effective_window
        cumsum_sq = np.cumsum(np.insert(rewards_arr ** 2, 0, 0))
        mean_sq = (cumsum_sq[effective_window:] - cumsum_sq[:-effective_window]) / effective_window
        rolling_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
        ci = 1.96 * rolling_std / np.sqrt(effective_window)
        return mean, ci

    def _plot_curve(self, ax, rewards, window_size, label=None, show_ci=False):
        """Plot a single rolling-average curve with optional CI band."""
        mean, ci = self._rolling_stats(rewards, window_size)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label)
        if show_ci:
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)

    def plot_learning_curve(self, rewards: list, window_size: int = 1000,
                            show_ci: bool = False) -> Figure:
        """Smoothed average returns over episodes.

        Args:
            rewards: list of per-episode rewards.
            window_size: rolling average window size.
            show_ci: if True, show 95% confidence interval shading.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        self._plot_curve(ax, rewards, window_size, show_ci=show_ci)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Average Return (window={window_size})")
        ax.set_title("Learning Curve")
        if show_ci:
            ax.annotate("Shading = 95% confidence interval",
                         xy=(0.99, 0.01), xycoords="axes fraction",
                         ha="right", va="bottom", fontsize=8, fontstyle="italic",
                         color="gray")
        return fig

    def plot_bakeoff(self, results: dict) -> Figure:
        """Bar chart comparing average returns across agents.

        Args:
            results: dict mapping agent name -> average return.

        Returns:
            matplotlib Figure.
        """
        names = list(results.keys())
        values = list(results.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values)
        for bar, val in zip(bars, values):
            color = "green" if val >= 0 else "red"
            bar.set_color(color)
        ax.set_ylabel("Average Return")
        ax.set_title("Agent Bake-Off")
        ax.axhline(y=0, color="black", linewidth=0.5)
        return fig

    def plot_bakeoff_curves(self, reward_series: dict,
                             window_size: int = 1000,
                             show_ci: bool = True,
                             title: str = "Agent Bake-Off: Learning Curves") -> Figure:
        """Overlay rolling-average learning curves for multiple agents.

        Args:
            reward_series: dict mapping agent name -> list of per-episode rewards.
            window_size: rolling average window size.
            show_ci: if True, show 95% confidence interval shading.
            title: plot title.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, rewards in reward_series.items():
            self._plot_curve(ax, rewards, window_size, label=name, show_ci=show_ci)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Average Return (window={window_size})")
        ax.set_title(title)
        ax.legend()
        if show_ci:
            ax.annotate("Shading = 95% confidence interval",
                         xy=(0.99, 0.01), xycoords="axes fraction",
                         ha="right", va="bottom", fontsize=8, fontstyle="italic",
                         color="gray")
        return fig

    def plot_outcome_breakdown(self, reward_series: dict) -> Figure:
        """Stacked bar chart showing win/loss/draw proportions per agent.

        Args:
            reward_series: dict mapping agent name -> list of per-episode rewards.

        Returns:
            matplotlib Figure.
        """
        names = list(reward_series.keys())
        wins, losses, draws = [], [], []
        for name in names:
            rewards = np.array(reward_series[name])
            n = len(rewards)
            wins.append(np.sum(rewards > 0) / n)
            losses.append(np.sum(rewards < 0) / n)
            draws.append(np.sum(rewards == 0) / n)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names))
        ax.bar(x, losses, label="Loss", color="red")
        ax.bar(x, draws, bottom=losses, label="Draw", color="gold")
        ax.bar(x, wins, bottom=np.array(losses) + np.array(draws),
               label="Win", color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Proportion")
        ax.set_title("Outcome Breakdown")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        return fig

    def save_plot(self, fig: Figure, path: str) -> None:
        """Save a figure to disk, creating directories if needed.

        Args:
            fig: matplotlib Figure.
            path: output file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
