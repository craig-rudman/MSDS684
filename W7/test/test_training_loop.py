import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Agent
from taxi_env import TaxiEnv
from training_loop import TrainingLoop


EXPECTED_COLUMNS = [
    "agent_id",
    "seed",
    "episode",
    "total_reward",
    "num_steps",
    "cumulative_steps",
    "terminated",
    "truncated",
    "wall_time_episode",
    "wall_time_planning_episode",
    "n_planning_updates_episode",
    "cumulative_planning_updates",
]


class StubAgent(Agent):
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.learn_call_count = 0

    def act(self, state):
        return state % self.num_actions

    def learn(self, state, action, reward, next_state, terminated, truncated):
        self.learn_call_count += 1
        return {"wall_time_planning": 0.123, "n_planning_updates": 7}


def make_loop(tmp_path, agent_id="stub_agent", hyperparams=None, record_per_step=False):
    env = TaxiEnv()
    agent = StubAgent(num_actions=env.num_actions)
    return TrainingLoop(
        env=env,
        agent=agent,
        agent_id=agent_id,
        output_dir=tmp_path,
        hyperparams=hyperparams or {"alpha": 0.1, "epsilon": 0.1, "gamma": 0.99},
        record_per_step=record_per_step,
    ), agent


def test_construction(tmp_path):
    loop, _ = make_loop(tmp_path)
    assert loop is not None


def test_run_produces_one_row_per_episode(tmp_path):
    loop, _ = make_loop(tmp_path)
    trace = loop.run(num_episodes=3, seed=0)
    assert len(trace) == 3
    assert list(trace["episode"]) == [0, 1, 2]


def test_trace_has_expected_columns(tmp_path):
    loop, _ = make_loop(tmp_path)
    trace = loop.run(num_episodes=1, seed=0)
    assert list(trace.columns) == EXPECTED_COLUMNS


def test_trace_csv_is_written(tmp_path):
    loop, _ = make_loop(tmp_path, agent_id="stub_agent")
    loop.run(num_episodes=1, seed=0)
    expected_path = tmp_path / "stub_agent" / "trace_seed_0.csv"
    assert expected_path.exists()
    on_disk = pd.read_csv(expected_path)
    assert list(on_disk.columns) == EXPECTED_COLUMNS


def test_config_sidecar_is_written(tmp_path):
    hyperparams = {"alpha": 0.5, "epsilon": 0.2, "gamma": 0.9}
    loop, _ = make_loop(tmp_path, agent_id="stub_agent", hyperparams=hyperparams)
    loop.run(num_episodes=2, seed=0)
    expected_path = tmp_path / "stub_agent" / "config_seed_0.json"
    assert expected_path.exists()
    config = json.loads(expected_path.read_text())
    assert config["agent_id"] == "stub_agent"
    assert config["seed"] == 0
    assert config["num_episodes"] == 2
    assert config["hyperparams"] == hyperparams


def test_reserved_keys_in_hyperparams_raise(tmp_path):
    env = TaxiEnv()
    agent = StubAgent(num_actions=env.num_actions)
    with pytest.raises(ValueError):
        TrainingLoop(
            env=env,
            agent=agent,
            agent_id="bad_agent",
            output_dir=tmp_path,
            hyperparams={"num_episodes": 100},
        )


def test_subdirectory_is_created(tmp_path):
    loop, _ = make_loop(tmp_path, agent_id="brand_new_agent")
    assert not (tmp_path / "brand_new_agent").exists()
    loop.run(num_episodes=1, seed=0)
    assert (tmp_path / "brand_new_agent").is_dir()


def test_overwrite_on_rerun_with_same_seed(tmp_path):
    loop, _ = make_loop(tmp_path, agent_id="stub_agent")
    loop.run(num_episodes=2, seed=0)
    trace_b = loop.run(num_episodes=3, seed=0)
    on_disk = pd.read_csv(tmp_path / "stub_agent" / "trace_seed_0.csv")
    assert len(on_disk) == len(trace_b)


def test_each_episode_ends_with_terminated_or_truncated(tmp_path):
    loop, _ = make_loop(tmp_path)
    trace = loop.run(num_episodes=3, seed=0)
    for _, row in trace.iterrows():
        assert bool(row["terminated"]) or bool(row["truncated"])


def test_learn_called_once_per_real_step(tmp_path):
    loop, agent = make_loop(tmp_path)
    trace = loop.run(num_episodes=3, seed=0)
    assert agent.learn_call_count == int(trace["num_steps"].sum())


def test_aggregated_stats_from_learn_propagate(tmp_path):
    loop, _ = make_loop(tmp_path)
    trace = loop.run(num_episodes=1, seed=0)
    expected_planning_time = 0.123 * trace["num_steps"].iloc[0]
    expected_planning_updates = 7 * trace["num_steps"].iloc[0]
    assert trace["wall_time_planning_episode"].iloc[0] == pytest.approx(expected_planning_time)
    assert trace["n_planning_updates_episode"].iloc[0] == expected_planning_updates


def test_wall_time_episode_positive(tmp_path):
    loop, _ = make_loop(tmp_path)
    trace = loop.run(num_episodes=1, seed=0)
    assert (trace["wall_time_episode"] > 0).all()


def test_cumulative_steps_is_running_sum(tmp_path):
    loop, _ = make_loop(tmp_path)
    trace = loop.run(num_episodes=3, seed=0)
    expected = trace["num_steps"].cumsum().tolist()
    assert trace["cumulative_steps"].tolist() == expected


def test_cumulative_planning_updates_is_running_sum(tmp_path):
    loop, _ = make_loop(tmp_path)
    trace = loop.run(num_episodes=3, seed=0)
    expected = trace["n_planning_updates_episode"].cumsum().tolist()
    assert trace["cumulative_planning_updates"].tolist() == expected


def test_record_per_step_true_raises_not_implemented(tmp_path):
    loop, _ = make_loop(tmp_path, record_per_step=True)
    with pytest.raises(NotImplementedError):
        loop.run(num_episodes=1, seed=0)
