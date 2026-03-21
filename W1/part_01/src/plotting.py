"""Plotting utilities for comparing bandit agent performance."""

import matplotlib.pyplot as plt
import numpy as np

from .runner import SimulationResult


PANEL_REGISTRY = {
    "reward": {
        "accessor": "mean_reward",
        "ylabel": "Average reward",
        "title": "Average reward over time",
    },
    "optimal_pct": {
        "accessor": "optimal_action_pct",
        "ylabel": "% Optimal action",
        "title": "Optimal action selection over time",
        "ylim": (0, 100),
    },
    "regret": {
        "accessor": "cumulative_regret",
        "ylabel": "Cumulative regret",
        "title": "Cumulative regret over time",
    },
}


class BanditPlotter:
    """Plotting class for comparing bandit agent performance."""

    def __init__(
        self,
        results: list[SimulationResult] | None = None,
        smoothing: int = 1,
    ):
        self.results = results or []
        self.smoothing = smoothing

    @staticmethod
    def _smooth(arr, w):
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    def _get_colors(self):
        return plt.cm.tab10(np.linspace(0, 1, len(self.results)))

    def _plot_panel_on_ax(self, ax, panel_key):
        spec = PANEL_REGISTRY[panel_key]
        colors = self._get_colors()

        for res, color in zip(self.results, colors):
            data = getattr(res, spec["accessor"])()
            smoothed = self._smooth(data, self.smoothing)
            x = np.arange(len(smoothed))
            ax.plot(x, smoothed, label=res.agent_name, color=color, linewidth=1.5)

        ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"])
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

        if "ylim" in spec:
            ax.set_ylim(spec["ylim"])

    def plot_comparison(
        self,
        panels: list[str] | None = None,
        figsize: tuple | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot a multi-panel comparison of bandit agents.

        Args:
            panels: Which panels to show. Options: "reward", "optimal_pct", "regret".
                    Defaults to all three.
            figsize: Figure size. Defaults to (12, 4*n_panels).
            save_path: If provided, save the figure to this path.

        Returns:
            The matplotlib Figure.
        """
        if panels is None:
            panels = ["reward", "optimal_pct", "regret"]

        for p in panels:
            if p not in PANEL_REGISTRY:
                raise ValueError(f"Unknown panel '{p}'. Options: {list(PANEL_REGISTRY)}")

        n_panels = len(panels)
        if figsize is None:
            figsize = (12, 4 * n_panels)

        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
        if n_panels == 1:
            axes = [axes]

        for i, panel_key in enumerate(panels):
            self._plot_panel_on_ax(axes[i], panel_key)

        axes[-1].set_xlabel("Step")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_panel(
        self,
        panel: str,
        figsize: tuple = (12, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot a single metric as its own figure.

        Args:
            panel: Which metric to plot. Options: "reward", "optimal_pct", "regret".
            figsize: Figure size.
            save_path: If provided, save the figure to this path.

        Returns:
            The matplotlib Figure.
        """
        if panel not in PANEL_REGISTRY:
            raise ValueError(f"Unknown panel '{panel}'. Options: {list(PANEL_REGISTRY)}")

        fig, ax = plt.subplots(figsize=figsize)
        self._plot_panel_on_ax(ax, panel)
        ax.set_xlabel("Step")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def plot_reward_distributions(
        env,
        n_samples: int = 1000,
        figsize: tuple = (10, 6),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Violin plot of the reward distribution for each arm in a bandit environment.

        Args:
            env: A MultiArmedBanditEnv instance (must already be reset).
            n_samples: Number of reward samples to draw per arm.
            figsize: Figure size.
            save_path: If provided, save the figure to this path.

        Returns:
            The matplotlib Figure.
        """
        k = env.k
        true_means = env.true_means()
        samples = np.array([
            [env._sample_reward(arm) for _ in range(n_samples)]
            for arm in range(k)
        ])

        fig, ax = plt.subplots(figsize=figsize)
        parts = ax.violinplot(samples.T, positions=range(k), showmeans=True, showmedians=False)

        # Highlight the optimal arm
        optimal = int(np.argmax(true_means))
        for i, body in enumerate(parts["bodies"]):
            body.set_alpha(0.7)
            if i == optimal:
                body.set_facecolor("gold")
            else:
                body.set_facecolor("steelblue")

        ax.scatter(range(k), true_means, color="red", zorder=5, s=30, label="True mean (μ)")
        ax.set_xticks(range(k))
        ax.set_xticklabels([f"Arm {i}" for i in range(k)])
        ax.set_xlabel("Action (arm)")
        ax.set_ylabel("Reward distribution")
        ax.set_title("Reward Distributions by Arm")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


# Module-level convenience functions for backward compatibility
def plot_comparison(
    results: list[SimulationResult],
    panels: list[str] | None = None,
    smoothing: int = 1,
    figsize: tuple | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    return BanditPlotter(results, smoothing).plot_comparison(panels, figsize, save_path)


def plot_panel(
    results: list[SimulationResult],
    panel: str,
    smoothing: int = 1,
    figsize: tuple = (12, 5),
    save_path: str | None = None,
) -> plt.Figure:
    return BanditPlotter(results, smoothing).plot_panel(panel, figsize, save_path)


def plot_reward_distributions(
    env,
    n_samples: int = 1000,
    figsize: tuple = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    return BanditPlotter.plot_reward_distributions(env, n_samples, figsize, save_path)
