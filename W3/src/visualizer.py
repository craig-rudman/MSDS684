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

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        ax.set_title(title)
        return fig

    def plot_learning_curve(self, rewards: list, window_size: int = 1000) -> Figure:
        """Smoothed average returns over episodes.

        Args:
            rewards: list of per-episode rewards.
            window_size: rolling average window size.

        Returns:
            matplotlib Figure.
        """
        rewards_arr = np.array(rewards)
        effective_window = min(window_size, len(rewards_arr))
        if effective_window < 1:
            effective_window = 1
        cumsum = np.cumsum(np.insert(rewards_arr, 0, 0))
        smoothed = (cumsum[effective_window:] - cumsum[:-effective_window]) / effective_window

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(smoothed)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Average Return (window={effective_window})")
        ax.set_title("Learning Curve")
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
                             window_size: int = 1000) -> Figure:
        """Overlay rolling-average learning curves for multiple agents.

        Args:
            reward_series: dict mapping agent name -> list of per-episode rewards.
            window_size: rolling average window size.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, rewards in reward_series.items():
            rewards_arr = np.array(rewards, dtype=float)
            effective_window = min(window_size, len(rewards_arr))
            if effective_window < 1:
                effective_window = 1
            # Rolling mean
            cumsum = np.cumsum(np.insert(rewards_arr, 0, 0))
            smoothed = (cumsum[effective_window:] - cumsum[:-effective_window]) / effective_window
            # Rolling standard deviation
            cumsum_sq = np.cumsum(np.insert(rewards_arr ** 2, 0, 0))
            mean_sq = (cumsum_sq[effective_window:] - cumsum_sq[:-effective_window]) / effective_window
            rolling_std = np.sqrt(np.maximum(mean_sq - smoothed ** 2, 0.0))
            # 95% CI
            band = 1.96 * rolling_std / np.sqrt(effective_window)
            x = np.arange(len(smoothed))
            ax.plot(x, smoothed, label=name)
            ax.fill_between(x, smoothed - band, smoothed + band, alpha=0.2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Average Return (window={window_size})")
        ax.set_title("Agent Bake-Off: Learning Curves")
        ax.legend()
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
