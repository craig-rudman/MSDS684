import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tabular_model import TabularModel


# ---------------------------------------------------------------------------
# TabularModel
# ---------------------------------------------------------------------------

def test_tabular_model_sample_raises_when_empty():
    model = TabularModel()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        model.sample(rng)


def test_tabular_model_stores_and_retrieves_transition():
    model = TabularModel()
    rng = np.random.default_rng(0)
    model.update(s=0, a=1, r=2.0, s_next=3)
    s, a, r, s_next = model.sample(rng)
    assert (s, a, r, s_next) == (0, 1, 2.0, 3)


def test_tabular_model_overwrites_same_sa_pair():
    model = TabularModel()
    rng = np.random.default_rng(0)
    model.update(s=0, a=1, r=1.0, s_next=2)
    model.update(s=0, a=1, r=9.0, s_next=5)
    s, a, r, s_next = model.sample(rng)
    assert (s, a, r, s_next) == (0, 1, 9.0, 5)


def test_tabular_model_sample_covers_all_stored_pairs():
    model = TabularModel()
    rng = np.random.default_rng(0)
    transitions = {(0, 0): (1.0, 1), (1, 2): (-1.0, 3), (4, 5): (0.0, 0)}
    for (s, a), (r, s_next) in transitions.items():
        model.update(s=s, a=a, r=r, s_next=s_next)
    seen = set()
    for _ in range(200):
        s, a, r, s_next = model.sample(rng)
        seen.add((s, a))
    assert seen == set(transitions.keys())


# ---------------------------------------------------------------------------
# DynaQAgent
# ---------------------------------------------------------------------------

def make_agent(n_planning=5, num_states=10, num_actions=6,
               alpha=0.1, epsilon=0.0, gamma=0.99, seed=0):
    from dyna_q_agent import DynaQAgent
    return DynaQAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        n_planning=n_planning,
        seed=seed,
    )


def test_dyna_q_is_subclass_of_q_learning_agent():
    from q_learning_agent import QLearningAgent
    agent = make_agent()
    assert isinstance(agent, QLearningAgent)


def test_dyna_q_learn_returns_planning_stats():
    agent = make_agent(n_planning=5)
    stats = agent.learn(state=0, action=0, reward=1.0, next_state=1,
                        terminated=False, truncated=False)
    assert stats["n_planning_updates"] == 5
    assert "wall_time_planning" in stats
    assert stats["wall_time_planning"] >= 0.0


def test_dyna_q_planning_updates_q_table():
    agent = make_agent(n_planning=10, alpha=1.0, gamma=0.0)
    agent.Q[1, 0] = 5.0
    agent.learn(state=0, action=0, reward=1.0, next_state=1,
                terminated=False, truncated=False)
    # alpha=1, gamma=0: real step sets Q[0,0]=1.0; planning replays same
    # transition so Q[0,0] stays 1.0
    assert agent.Q[0, 0] == pytest.approx(1.0)


def test_dyna_q_zero_planning_not_allowed():
    from dyna_q_agent import DynaQAgent
    with pytest.raises((ValueError, AssertionError)):
        DynaQAgent(num_states=10, num_actions=6, alpha=0.1,
                   epsilon=0.0, gamma=0.99, n_planning=0, seed=0)


def test_dyna_q_model_grows_with_experience():
    agent = make_agent(n_planning=1)
    assert len(agent.model._transitions) == 0
    agent.learn(state=0, action=0, reward=0.0, next_state=1,
                terminated=False, truncated=False)
    assert len(agent.model._transitions) == 1
    agent.learn(state=1, action=2, reward=0.0, next_state=2,
                terminated=False, truncated=False)
    assert len(agent.model._transitions) == 2
