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


def plot_comparison(
    results: list[SimulationResult],
    panels: list[str] | None = None,
    smoothing: int = 1,
    figsize: tuple | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a multi-panel comparison of bandit agents.

    Args:
        results: List of SimulationResult objects to compare.
        panels: Which panels to show. Options: "reward", "optimal_pct", "regret".
                Defaults to all three.
        smoothing: Window size for moving average (1 = no smoothing).
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

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    def smooth(arr, w):
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    for i, panel_key in enumerate(panels):
        panel = PANEL_REGISTRY[panel_key]
        ax = axes[i]

        for res, color in zip(results, colors):
            data = getattr(res, panel["accessor"])()
            smoothed = smooth(data, smoothing)
            x = np.arange(len(smoothed))
            ax.plot(x, smoothed, label=res.agent_name, color=color, linewidth=1.5)

        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

        if "ylim" in panel:
            ax.set_ylim(panel["ylim"])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
