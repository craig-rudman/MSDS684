from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compute_cumulative_reward(traces: pd.DataFrame) -> pd.DataFrame:
    sorted_traces = traces.sort_values(["agent_id", "seed", "step"])
    cumulative = (
        sorted_traces
        .groupby(["agent_id", "seed"])["reward"]
        .cumsum()
    )
    return pd.DataFrame({
        "agent_id": sorted_traces["agent_id"].values,
        "seed": sorted_traces["seed"].values,
        "step": sorted_traces["step"].values,
        "cumulative_reward": cumulative.values,
    })


def plot_cumulative_reward(traces: pd.DataFrame, output_path: Path | str) -> None:
    data = compute_cumulative_reward(traces)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x="step",
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
