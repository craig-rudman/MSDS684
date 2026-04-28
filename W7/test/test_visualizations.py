import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visualizations import compute_cumulative_reward, plot_cumulative_reward


def make_trace(agent_id: str, seed: int, rewards: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "agent_id": [agent_id] * len(rewards),
        "seed": [seed] * len(rewards),
        "step": list(range(len(rewards))),
        "reward": rewards,
    })


def test_cumulative_reward_single_agent_single_seed():
    trace = make_trace("q_learning", seed=0, rewards=[1.0, 2.0, -1.0, 5.0])
    result = compute_cumulative_reward(trace)
    assert list(result["cumulative_reward"]) == [1.0, 3.0, 2.0, 7.0]


def test_cumulative_reward_single_agent_multiple_seeds():
    trace = pd.concat([
        make_trace("q_learning", seed=0, rewards=[1.0, 1.0, 1.0]),
        make_trace("q_learning", seed=1, rewards=[2.0, 2.0, 2.0]),
    ], ignore_index=True)
    result = compute_cumulative_reward(trace)
    seed0 = result[result["seed"] == 0]["cumulative_reward"].tolist()
    seed1 = result[result["seed"] == 1]["cumulative_reward"].tolist()
    assert seed0 == [1.0, 2.0, 3.0]
    assert seed1 == [2.0, 4.0, 6.0]


def test_cumulative_reward_multiple_agents():
    trace = pd.concat([
        make_trace("q_learning", seed=0, rewards=[1.0, 1.0]),
        make_trace("dyna_q_n10", seed=0, rewards=[3.0, 3.0]),
    ], ignore_index=True)
    result = compute_cumulative_reward(trace)
    q_learning = result[result["agent_id"] == "q_learning"]["cumulative_reward"].tolist()
    dyna = result[result["agent_id"] == "dyna_q_n10"]["cumulative_reward"].tolist()
    assert q_learning == [1.0, 2.0]
    assert dyna == [3.0, 6.0]


def test_cumulative_reward_output_schema():
    trace = make_trace("q_learning", seed=0, rewards=[1.0, 2.0])
    result = compute_cumulative_reward(trace)
    assert set(result.columns) == {"agent_id", "seed", "step", "cumulative_reward"}


def test_plot_cumulative_reward_writes_file(tmp_path):
    trace = pd.concat([
        make_trace("q_learning", seed=0, rewards=[1.0, 2.0, 3.0, 4.0, 5.0]),
        make_trace("dyna_q_n10", seed=0, rewards=[2.0, 4.0, 6.0, 8.0, 10.0]),
    ], ignore_index=True)
    output_path = tmp_path / "cumulative_reward.png"
    plot_cumulative_reward(trace, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
