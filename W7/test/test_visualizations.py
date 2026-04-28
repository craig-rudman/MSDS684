import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visualizations import (
    compute_cumulative_reward,
    compute_termination_rate,
    plot_cumulative_reward,
    plot_termination_rate,
)


def make_episode_trace(agent_id: str, seed: int, total_rewards: list[float]) -> pd.DataFrame:
    n = len(total_rewards)
    num_steps = [10] * n
    n_planning_updates = [0] * n
    return pd.DataFrame({
        "agent_id": [agent_id] * n,
        "seed": [seed] * n,
        "episode": list(range(n)),
        "total_reward": total_rewards,
        "num_steps": num_steps,
        "cumulative_steps": pd.Series(num_steps).cumsum().tolist(),
        "terminated": [True] * n,
        "truncated": [False] * n,
        "wall_time_episode": [0.01] * n,
        "wall_time_planning_episode": [0.0] * n,
        "n_planning_updates_episode": n_planning_updates,
        "cumulative_planning_updates": pd.Series(n_planning_updates).cumsum().tolist(),
    })


def test_cumulative_reward_single_agent_single_seed():
    trace = make_episode_trace("q_learning", seed=0, total_rewards=[1.0, 2.0, -1.0, 5.0])
    result = compute_cumulative_reward(trace)
    assert list(result["cumulative_reward"]) == [1.0, 3.0, 2.0, 7.0]


def test_cumulative_reward_single_agent_multiple_seeds():
    trace = pd.concat([
        make_episode_trace("q_learning", seed=0, total_rewards=[1.0, 1.0, 1.0]),
        make_episode_trace("q_learning", seed=1, total_rewards=[2.0, 2.0, 2.0]),
    ], ignore_index=True)
    result = compute_cumulative_reward(trace)
    seed0 = result[result["seed"] == 0]["cumulative_reward"].tolist()
    seed1 = result[result["seed"] == 1]["cumulative_reward"].tolist()
    assert seed0 == [1.0, 2.0, 3.0]
    assert seed1 == [2.0, 4.0, 6.0]


def test_cumulative_reward_multiple_agents():
    trace = pd.concat([
        make_episode_trace("q_learning", seed=0, total_rewards=[1.0, 1.0]),
        make_episode_trace("dyna_q_n10", seed=0, total_rewards=[3.0, 3.0]),
    ], ignore_index=True)
    result = compute_cumulative_reward(trace)
    q_learning = result[result["agent_id"] == "q_learning"]["cumulative_reward"].tolist()
    dyna = result[result["agent_id"] == "dyna_q_n10"]["cumulative_reward"].tolist()
    assert q_learning == [1.0, 2.0]
    assert dyna == [3.0, 6.0]


def test_cumulative_reward_output_schema():
    trace = make_episode_trace("q_learning", seed=0, total_rewards=[1.0, 2.0])
    result = compute_cumulative_reward(trace)
    assert set(result.columns) == {"agent_id", "seed", "cumulative_steps", "cumulative_reward"}


def test_plot_cumulative_reward_writes_file(tmp_path):
    trace = pd.concat([
        make_episode_trace("q_learning", seed=0, total_rewards=[1.0, 2.0, 3.0, 4.0, 5.0]),
        make_episode_trace("dyna_q_n10", seed=0, total_rewards=[2.0, 4.0, 6.0, 8.0, 10.0]),
    ], ignore_index=True)
    output_path = tmp_path / "cumulative_reward.png"
    plot_cumulative_reward(trace, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_cumulative_reward_with_label_map(tmp_path):
    trace = make_episode_trace("q_learning", seed=0, total_rewards=[1.0, 2.0, 3.0])
    output_path = tmp_path / "cumulative_reward_labeled.png"
    plot_cumulative_reward(trace, output_path, label_map={"q_learning": "Q-learning"})
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def make_episode_trace_with_terminations(
    agent_id: str, seed: int, terminated_pattern: list[bool]
) -> pd.DataFrame:
    n = len(terminated_pattern)
    num_steps = [10] * n
    n_planning_updates = [0] * n
    return pd.DataFrame({
        "agent_id": [agent_id] * n,
        "seed": [seed] * n,
        "episode": list(range(n)),
        "total_reward": [0.0] * n,
        "num_steps": num_steps,
        "cumulative_steps": pd.Series(num_steps).cumsum().tolist(),
        "terminated": terminated_pattern,
        "truncated": [not t for t in terminated_pattern],
        "wall_time_episode": [0.01] * n,
        "wall_time_planning_episode": [0.0] * n,
        "n_planning_updates_episode": n_planning_updates,
        "cumulative_planning_updates": pd.Series(n_planning_updates).cumsum().tolist(),
    })


def test_termination_rate_output_schema():
    trace = make_episode_trace_with_terminations(
        "q_learning", seed=0, terminated_pattern=[False, True, True]
    )
    result = compute_termination_rate(trace, window=2)
    assert set(result.columns) == {"agent_id", "seed", "episode", "trailing_termination_rate"}


def test_termination_rate_rolling_mean_values():
    trace = make_episode_trace_with_terminations(
        "q_learning", seed=0, terminated_pattern=[False, True, True, True, False]
    )
    result = compute_termination_rate(trace, window=2)
    rates = result["trailing_termination_rate"].tolist()
    assert rates == [0.0, 0.5, 1.0, 1.0, 0.5]


def test_termination_rate_per_agent_seed_independent():
    trace = pd.concat([
        make_episode_trace_with_terminations(
            "q_learning", seed=0, terminated_pattern=[False, False, False]
        ),
        make_episode_trace_with_terminations(
            "q_learning", seed=1, terminated_pattern=[True, True, True]
        ),
    ], ignore_index=True)
    result = compute_termination_rate(trace, window=2)
    seed0 = result[result["seed"] == 0]["trailing_termination_rate"].tolist()
    seed1 = result[result["seed"] == 1]["trailing_termination_rate"].tolist()
    assert seed0 == [0.0, 0.0, 0.0]
    assert seed1 == [1.0, 1.0, 1.0]


def test_plot_termination_rate_writes_file(tmp_path):
    trace = make_episode_trace_with_terminations(
        "q_learning", seed=0, terminated_pattern=[False, False, True, True, True]
    )
    output_path = tmp_path / "episodes_to_optimal.png"
    plot_termination_rate(trace, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_termination_rate_with_threshold_writes_file(tmp_path):
    trace = make_episode_trace_with_terminations(
        "q_learning", seed=0, terminated_pattern=[False, False, True, True, True]
    )
    output_path = tmp_path / "episodes_to_optimal_threshold.png"
    plot_termination_rate(trace, output_path, threshold=0.9)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_termination_rate_with_label_map(tmp_path):
    trace = make_episode_trace_with_terminations(
        "q_learning", seed=0, terminated_pattern=[False, True, True]
    )
    output_path = tmp_path / "episodes_to_optimal_labeled.png"
    plot_termination_rate(trace, output_path, label_map={"q_learning": "Q-learning"})
    assert output_path.exists()
    assert output_path.stat().st_size > 0
