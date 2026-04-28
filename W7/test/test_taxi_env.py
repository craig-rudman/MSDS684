import sys
from pathlib import Path

import gymnasium as gym
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taxi_env import TaxiEnv


def test_construction_default():
    env = TaxiEnv()
    assert env is not None


def test_construction_with_seed():
    env = TaxiEnv(seed=42)
    assert env is not None


def test_reset_returns_int_in_range():
    env = TaxiEnv(seed=42)
    state = env.reset()
    assert isinstance(state, int)
    assert 0 <= state < 500


def test_step_returns_four_tuple_with_correct_types():
    env = TaxiEnv(seed=42)
    env.reset()
    result = env.step(0)
    assert len(result) == 4
    state, reward, terminated, truncated = result
    assert isinstance(state, int)
    assert 0 <= state < 500
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_determinism_across_instances():
    actions = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3] * 5

    env_a = TaxiEnv(seed=42)
    env_b = TaxiEnv(seed=42)

    state_a = env_a.reset()
    state_b = env_b.reset()
    assert state_a == state_b

    for action in actions:
        result_a = env_a.step(action)
        result_b = env_b.step(action)
        assert result_a == result_b
        if result_a[2] or result_a[3]:
            state_a = env_a.reset()
            state_b = env_b.reset()
            assert state_a == state_b


def test_seed_once_semantics_initial_states_vary():
    env = TaxiEnv(seed=42)
    initial_states = {env.reset() for _ in range(5)}
    assert len(initial_states) > 1


def test_wrapper_injection_honored():
    inner = gym.make("Taxi-v4")
    env = TaxiEnv(gym_env=inner, seed=42)
    state = env.reset()
    assert isinstance(state, int)
    assert 0 <= state < 500
    result = env.step(0)
    assert len(result) == 4


def test_num_actions_is_six():
    env = TaxiEnv()
    assert env.num_actions == 6


def test_num_states_is_five_hundred():
    env = TaxiEnv()
    assert env.num_states == 500


def test_dimensions_available_before_reset():
    env = TaxiEnv(seed=42)
    assert env.num_actions == 6
    assert env.num_states == 500
