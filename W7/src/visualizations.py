from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def compute_cumulative_reward_aggregated(
    traces: pd.DataFrame,
    num_bins: int = 200,
) -> pd.DataFrame:
    cumulative = compute_cumulative_reward(traces)
    rows = []
    for agent_id, group in cumulative.groupby("agent_id"):
        max_steps = group["cumulative_steps"].max()
        edges = pd.cut(
            group["cumulative_steps"],
            bins=num_bins,
            labels=False,
            include_lowest=True,
        )
        bin_centers = (
            group.assign(bin=edges)
            .groupby("bin")["cumulative_steps"]
            .mean()
        )
        agg = (
            group.assign(bin=edges)
            .groupby("bin")["cumulative_reward"]
            .agg(["mean", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)])
            .rename(columns={"<lambda_0>": "ci_lower", "<lambda_1>": "ci_upper"})
        )
        for bin_idx in agg.index:
            rows.append({
                "agent_id": agent_id,
                "cumulative_steps": float(bin_centers.loc[bin_idx]),
                "mean": float(agg.loc[bin_idx, "mean"]),
                "ci_lower": float(agg.loc[bin_idx, "ci_lower"]),
                "ci_upper": float(agg.loc[bin_idx, "ci_upper"]),
            })
    return pd.DataFrame(rows, columns=["agent_id", "cumulative_steps", "mean", "ci_lower", "ci_upper"])


def plot_cumulative_reward(
    traces: pd.DataFrame,
    output_path: Path | str,
    label_map: dict[str, str] | None = None,
    band_alpha: float | None = None,
    num_bins: int = 200,
    legend_order: list[str] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    data = compute_cumulative_reward_aggregated(traces, num_bins=num_bins)
    if label_map:
        data = data.assign(agent_id=data["agent_id"].map(lambda x: label_map.get(x, x)))
    if x_min is not None:
        data = data[data["cumulative_steps"] >= x_min]
    if x_max is not None:
        data = data[data["cumulative_steps"] <= x_max]
    fig, ax = plt.subplots(figsize=(10, 6))
    band_alpha_value = band_alpha if band_alpha is not None else 0.2
    for agent_id, group in data.groupby("agent_id"):
        sorted_group = group.sort_values("cumulative_steps")
        line = ax.plot(sorted_group["cumulative_steps"], sorted_group["mean"], label=agent_id)
        ax.fill_between(
            sorted_group["cumulative_steps"],
            sorted_group["ci_lower"],
            sorted_group["ci_upper"],
            alpha=band_alpha_value,
            color=line[0].get_color(),
        )
    title = "Cumulative reward over real steps"
    if x_min is not None or x_max is not None:
        lo = int(x_min) if x_min is not None else 0
        hi = int(x_max) if x_max is not None else "end"
        title += f" (steps {lo}–{hi})"
    ax.set_title(title)
    ax.set_xlabel("Real step")
    ax.set_ylabel("Cumulative reward")
    if legend_order:
        handles = {h.get_label(): h for h in ax.get_lines()}
        ordered = [handles[label] for label in legend_order if label in handles]
        ax.legend(handles=ordered, labels=[h.get_label() for h in ordered], title="agent_id")
    else:
        ax.legend(title="agent_id")
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150)
    plt.close(fig)


def steps_to_reliable_termination(
    traces: pd.DataFrame,
    target_rate: float = 0.8,
    window: int = 50,
) -> pd.DataFrame:
    sorted_traces = traces.sort_values(["agent_id", "seed", "episode"])
    rolling = (
        sorted_traces
        .groupby(["agent_id", "seed"])["terminated"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    sorted_traces = sorted_traces.assign(trailing_rate=rolling.values)

    rows = []
    for (agent_id, seed), group in sorted_traces.groupby(["agent_id", "seed"]):
        crossed = group[group["trailing_rate"] >= target_rate]
        steps = float(crossed["cumulative_steps"].iloc[0]) if len(crossed) else float("nan")
        rows.append({
            "agent_id": agent_id,
            "seed": seed,
            "steps_to_reliable_termination": steps,
        })
    return pd.DataFrame(rows, columns=["agent_id", "seed", "steps_to_reliable_termination"])


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
        "cumulative_steps": sorted_traces["cumulative_steps"].values,
        "trailing_termination_rate": rolling.values,
    })


def compute_termination_rate_aggregated(
    traces: pd.DataFrame,
    window: int = 50,
    num_bins: int = 200,
    x_axis: str = "episode",
) -> pd.DataFrame:
    if x_axis not in {"episode", "cumulative_steps"}:
        raise ValueError(f"x_axis must be 'episode' or 'cumulative_steps', got {x_axis!r}")
    rolling = compute_termination_rate(traces, window=window)
    rows = []
    for agent_id, group in rolling.groupby("agent_id"):
        edges = pd.cut(group[x_axis], bins=num_bins, labels=False, include_lowest=True)
        bin_centers = (
            group.assign(bin=edges)
            .groupby("bin")[x_axis]
            .mean()
        )
        agg = (
            group.assign(bin=edges)
            .groupby("bin")["trailing_termination_rate"]
            .agg(["mean", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)])
            .rename(columns={"<lambda_0>": "ci_lower", "<lambda_1>": "ci_upper"})
        )
        for bin_idx in agg.index:
            rows.append({
                "agent_id": agent_id,
                "x": float(bin_centers.loc[bin_idx]),
                "mean": float(agg.loc[bin_idx, "mean"]),
                "ci_lower": float(agg.loc[bin_idx, "ci_lower"]),
                "ci_upper": float(agg.loc[bin_idx, "ci_upper"]),
            })
    return pd.DataFrame(rows, columns=["agent_id", "x", "mean", "ci_lower", "ci_upper"])


def plot_termination_rate(
    traces: pd.DataFrame,
    output_path: Path | str,
    threshold: float | None = None,
    window: int = 50,
    x_axis: str = "episode",
    label_map: dict[str, str] | None = None,
    band_alpha: float | None = None,
    num_bins: int = 200,
    legend_order: list[str] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> None:
    if x_axis not in {"episode", "cumulative_steps"}:
        raise ValueError(f"x_axis must be 'episode' or 'cumulative_steps', got {x_axis!r}")
    data = compute_termination_rate_aggregated(
        traces, window=window, num_bins=num_bins, x_axis=x_axis
    )
    if label_map:
        data = data.assign(agent_id=data["agent_id"].map(lambda x: label_map.get(x, x)))
    if x_min is not None:
        data = data[data["x"] >= x_min]
    if x_max is not None:
        data = data[data["x"] <= x_max]
    fig, ax = plt.subplots(figsize=(10, 6))
    band_alpha_value = band_alpha if band_alpha is not None else 0.2
    for agent_id, group in data.groupby("agent_id"):
        sorted_group = group.sort_values("x")
        line = ax.plot(sorted_group["x"], sorted_group["mean"], label=agent_id)
        ax.fill_between(
            sorted_group["x"],
            sorted_group["ci_lower"],
            sorted_group["ci_upper"],
            alpha=band_alpha_value,
            color=line[0].get_color(),
        )
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="gray", alpha=0.7)
    if x_axis == "episode":
        xlabel = "Episode"
        base_title = "Episodes until optimal performance"
        range_label = "episodes"
    else:
        xlabel = "Real step"
        base_title = "Sample efficiency: real steps until optimal performance"
        range_label = "steps"
    if x_min is not None or x_max is not None:
        lo = int(x_min) if x_min is not None else 0
        hi = int(x_max) if x_max is not None else "end"
        title = f"{base_title} ({range_label} {lo}–{hi})"
    else:
        title = base_title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Trailing-{window} termination rate")
    ax.set_ylim(-0.05, 1.05)
    if legend_order:
        handles = {h.get_label(): h for h in ax.get_lines()}
        ordered = [handles[label] for label in legend_order if label in handles]
        ax.legend(handles=ordered, labels=[h.get_label() for h in ordered], title="agent_id")
    else:
        ax.legend(title="agent_id")
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150)
    plt.close(fig)
