import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_traces import load_traces


TRACE_COLUMNS = [
    "agent_id",
    "seed",
    "step",
    "episode",
    "step_in_episode",
    "state",
    "action",
    "reward",
    "next_state",
    "terminated",
    "truncated",
    "wall_time_step",
    "wall_time_planning",
    "n_planning_updates",
]


def write_trace(directory: Path, agent_id: str, seed: int, n_rows: int = 5) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(n_rows):
        rows.append({
            "agent_id": agent_id,
            "seed": seed,
            "step": step,
            "episode": 0,
            "step_in_episode": step,
            "state": 0,
            "action": 0,
            "reward": -1.0,
            "next_state": 0,
            "terminated": False,
            "truncated": step == n_rows - 1,
            "wall_time_step": 0.001,
            "wall_time_planning": 0.0,
            "n_planning_updates": 0,
        })
    pd.DataFrame(rows, columns=TRACE_COLUMNS).to_csv(
        directory / f"trace_seed_{seed}.csv", index=False
    )


def test_single_agent_single_seed(tmp_path):
    agent_dir = tmp_path / "q_learning"
    write_trace(agent_dir, "q_learning", seed=0, n_rows=5)
    df = load_traces(agent_dir)
    assert len(df) == 5
    assert list(df.columns) == TRACE_COLUMNS
    assert (df["agent_id"] == "q_learning").all()
    assert (df["seed"] == 0).all()


def test_single_agent_multiple_seeds(tmp_path):
    agent_dir = tmp_path / "q_learning"
    for seed in (0, 1, 2):
        write_trace(agent_dir, "q_learning", seed=seed, n_rows=4)
    df = load_traces(agent_dir)
    assert len(df) == 12
    assert set(df["seed"].unique()) == {0, 1, 2}


def test_multiple_agent_dirs(tmp_path):
    write_trace(tmp_path / "q_learning", "q_learning", seed=0, n_rows=3)
    write_trace(tmp_path / "dyna_q_n10", "dyna_q_n10", seed=0, n_rows=3)
    df = load_traces(tmp_path / "q_learning", tmp_path / "dyna_q_n10")
    assert len(df) == 6
    assert set(df["agent_id"].unique()) == {"q_learning", "dyna_q_n10"}


def test_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty_agent"
    empty_dir.mkdir()
    with pytest.raises(Exception) as exc_info:
        load_traces(empty_dir)
    assert "empty_agent" in str(exc_info.value)


def test_nonexistent_directory_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(Exception) as exc_info:
        load_traces(missing)
    assert "does_not_exist" in str(exc_info.value)


def test_order_preservation(tmp_path):
    write_trace(tmp_path / "second", "second", seed=0, n_rows=2)
    write_trace(tmp_path / "first", "first", seed=0, n_rows=2)
    df = load_traces(tmp_path / "first", tmp_path / "second")
    agent_order = df["agent_id"].tolist()
    assert agent_order.index("first") < agent_order.index("second")
