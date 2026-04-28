from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compute_cumulative_reward(traces: pd.DataFrame) -> pd.DataFrame:
    sorted_traces = traces.sort_values(["agent_id", "seed", "episode"])
    cumulative = (
        sorted_traces
        .groupby(["agent_id", "seed"])["total_reward"]
        .cumsum()
    )
    return pd.DataFrame({
        "agent_id": sorted_traces["agent_id"].values,
        "seed": sorted_traces["seed"].values,
        "cumulative_steps": sorted_traces["cumulative_steps"].values,
        "cumulative_reward": cumulative.values,
    })


def plot_cumulative_reward(
    traces: pd.DataFrame,
    output_path: Path | str,
    label_map: dict[str, str] | None = None,
) -> None:
    data = compute_cumulative_reward(traces)
    if label_map:
        data = data.assign(agent_id=data["agent_id"].map(lambda x: label_map.get(x, x)))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="cumulative_steps",
        y="cumulative_reward",
        hue="agent_id",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_title("Cumulative reward over real steps")
    ax.set_xlabel("Real step")
    ax.set_ylabel("Cumulative reward")
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150)
    plt.close(fig)


def compute_termination_rate(traces: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    sorted_traces = traces.sort_values(["agent_id", "seed", "episode"])
    rolling = (
        sorted_traces
        .groupby(["agent_id", "seed"])["terminated"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return pd.DataFrame({
        "agent_id": sorted_traces["agent_id"].values,
        "seed": sorted_traces["seed"].values,
        "episode": sorted_traces["episode"].values,
        "trailing_termination_rate": rolling.values,
    })


def plot_termination_rate(
    traces: pd.DataFrame,
    output_path: Path | str,
    threshold: float | None = None,
    window: int = 50,
    label_map: dict[str, str] | None = None,
) -> None:
    data = compute_termination_rate(traces, window=window)
    if label_map:
        data = data.assign(agent_id=data["agent_id"].map(lambda x: label_map.get(x, x)))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="episode",
        y="trailing_termination_rate",
        hue="agent_id",
        errorbar=("ci", 95),
        ax=ax,
    )
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="gray", alpha=0.7)
    ax.set_title(f"Trailing-{window} termination rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Termination rate")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150)
    plt.close(fig)
