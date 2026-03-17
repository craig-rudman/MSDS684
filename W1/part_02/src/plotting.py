"""Plotting utilities for episodic agent performance."""

import matplotlib.pyplot as plt
import numpy as np

from .runner import EpisodeResult


def plot_episode_metrics(
    result: EpisodeResult,
    figsize: tuple | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot per-episode return distribution and rolling success rate.

    Creates a two-panel figure:
      1. Histogram of episode returns.
      2. Rolling success rate over episodes.

    Args:
        result: An EpisodeResult from run_episodes().
        figsize: Figure size. Defaults to (12, 5).
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    if figsize is None:
        figsize = (12, 5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{result.agent_name} on {result.env_id}", fontsize=14)

    # Panel 1: Return distribution
    returns = result.returns()
    ax1.hist(returns, bins=30, edgecolor="black", alpha=0.7)
    ax1.axvline(returns.mean(), color="red", linestyle="--",
                label=f"Mean: {returns.mean():.2f}")
    ax1.set_xlabel("Episode return")
    ax1.set_ylabel("Count")
    ax1.set_title("Return distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: Rolling success rate
    successes = result.successes().astype(float)
    window = max(1, len(successes) // 20)
    kernel = np.ones(window) / window
    rolling = np.convolve(successes, kernel, mode="valid")
    ax2.plot(rolling * 100, linewidth=1.5)
    ax2.axhline(successes.mean() * 100, color="red", linestyle="--",
                label=f"Overall: {successes.mean() * 100:.1f}%")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success rate (%)")
    ax2.set_title(f"Rolling success rate (window={window})")
    ax2.set_ylim(0, max(rolling.max() * 100 * 1.2, 10))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
