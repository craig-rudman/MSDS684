import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# DynamicTaxiWrapper
# ---------------------------------------------------------------------------

def make_wrapped_env(mutation_step=1000):
    from dynamic_taxi_wrapper import DynamicTaxiWrapper
    return DynamicTaxiWrapper(mutation_step=mutation_step)


def test_wrapper_locs_unchanged_before_mutation():
    env = make_wrapped_env(mutation_step=1000)
    env.reset(seed=0)
    original_locs = list(env.unwrapped.locs)
    for _ in range(999):
        obs, r, term, trunc, info = env.env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    assert list(env.unwrapped.locs) == original_locs


def test_wrapper_swaps_locs_at_mutation_step():
    env = make_wrapped_env(mutation_step=5)
    env.reset(seed=0)
    original_r = env.unwrapped.locs[0]
    original_b = env.unwrapped.locs[3]
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    assert env.unwrapped.locs[0] == original_b
    assert env.unwrapped.locs[3] == original_r


def test_wrapper_mutation_fires_only_once():
    env = make_wrapped_env(mutation_step=3)
    env.reset(seed=0)
    for _ in range(10):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    # After a second swap the locs would be back to original; they should not be
    locs_after = list(env.unwrapped.locs)
    original = [(0, 0), (0, 4), (4, 0), (4, 3)]
    assert locs_after != original


def test_wrapper_step_count_resets_on_reset():
    env = make_wrapped_env(mutation_step=1000)
    env.reset(seed=0)
    assert env.total_steps == 0
    env.step(env.action_space.sample())
    assert env.total_steps == 1


def test_wrapper_exposes_num_states_and_actions():
    env = make_wrapped_env()
    assert env.unwrapped.observation_space.n == 500
    assert env.unwrapped.action_space.n == 6


# ---------------------------------------------------------------------------
# DynaQPlusAgent
# ---------------------------------------------------------------------------

def make_plus_agent(n_planning=5, kappa=0.001, num_states=10, num_actions=6,
                    alpha=0.1, epsilon=0.0, gamma=0.99, seed=0):
    from dyna_q_plus_agent import DynaQPlusAgent
    return DynaQPlusAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        n_planning=n_planning,
        kappa=kappa,
        seed=seed,
    )


def test_dyna_q_plus_is_subclass_of_dyna_q_agent():
    from dyna_q_agent import DynaQAgent
    agent = make_plus_agent()
    assert isinstance(agent, DynaQAgent)


def test_dyna_q_plus_time_since_initializes_empty():
    agent = make_plus_agent()
    assert len(agent.time_since) == 0


def test_dyna_q_plus_time_since_increments_after_learn():
    agent = make_plus_agent(n_planning=1)
    agent.learn(state=0, action=0, reward=1.0, next_state=1,
                terminated=False, truncated=False)
    # Every visited (s,a) should appear in time_since
    assert (0, 0) in agent.time_since


def test_dyna_q_plus_stale_pair_gets_exploration_bonus():
    # With kappa>0, a (s,a) pair not visited for tau steps should produce a
    # higher effective Q-update than a freshly visited pair.
    agent = make_plus_agent(n_planning=0, kappa=1.0, alpha=1.0, gamma=0.0)
    # Manually age (s=0, a=0) by 100 steps without visiting it
    agent.model.update(s=0, a=0, r=0.0, s_next=1)
    agent.time_since[(0, 0)] = 100
    agent.time_since[(1, 1)] = 1
    agent.model.update(s=1, a=1, r=0.0, s_next=2)

    q_before_stale = float(agent.Q[0, 0])
    q_before_fresh = float(agent.Q[1, 1])

    # Trigger one planning step for each manually
    agent._planning_step(s=0, a=0)
    agent._planning_step(s=1, a=1)

    bonus_stale = agent.Q[0, 0] - q_before_stale
    bonus_fresh = agent.Q[1, 1] - q_before_fresh
    assert bonus_stale > bonus_fresh


def test_dyna_q_plus_learn_returns_planning_stats():
    agent = make_plus_agent(n_planning=5)
    stats = agent.learn(state=0, action=0, reward=1.0, next_state=1,
                        terminated=False, truncated=False)
    assert stats["n_planning_updates"] == 5
    assert "wall_time_planning" in stats
