import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from q_learning_agent import QLearningAgent


def make_agent(num_states=10, num_actions=6, alpha=0.1, epsilon=0.1, gamma=0.99, seed=0):
    return QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        seed=seed,
    )


def test_construction_initializes_zero_q_table():
    agent = make_agent(num_states=10, num_actions=6)
    assert agent.Q.shape == (10, 6)
    assert np.all(agent.Q == 0)


def test_act_returns_valid_action():
    agent = make_agent()
    action = agent.act(state=0)
    assert 0 <= action < agent.num_actions


def test_act_with_epsilon_zero_picks_unique_argmax():
    agent = make_agent(epsilon=0.0)
    agent.Q[3] = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        assert agent.act(state=3) == 2


def test_act_with_epsilon_zero_breaks_ties_uniformly():
    agent = make_agent(epsilon=0.0)
    agent.Q[7] = np.array([0.0, 5.0, 0.0, 5.0, 0.0, 0.0])
    picks = {agent.act(state=7) for _ in range(50)}
    assert 1 in picks
    assert 3 in picks
    assert picks <= {1, 3}


def test_act_with_epsilon_one_picks_uniformly():
    agent = make_agent(epsilon=1.0, num_actions=6)
    agent.Q[2] = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    picks = {agent.act(state=2) for _ in range(200)}
    assert picks == {0, 1, 2, 3, 4, 5}


def test_learn_updates_q_toward_target_non_terminal_no_bootstrap():
    agent = make_agent(alpha=1.0, gamma=0.0)
    agent.learn(state=1, action=2, reward=5.0, next_state=3, terminated=False, truncated=False)
    assert agent.Q[1, 2] == 5.0


def test_learn_bootstraps_when_not_terminated():
    agent = make_agent(alpha=1.0, gamma=1.0)
    agent.Q[3] = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    agent.learn(state=1, action=2, reward=0.0, next_state=3, terminated=False, truncated=False)
    assert agent.Q[1, 2] == 10.0


def test_learn_does_not_bootstrap_when_terminated():
    agent = make_agent(alpha=1.0, gamma=1.0)
    agent.Q[3] = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    agent.learn(state=1, action=2, reward=0.0, next_state=3, terminated=True, truncated=False)
    assert agent.Q[1, 2] == 0.0


def test_learn_bootstraps_when_truncated_but_not_terminated():
    agent = make_agent(alpha=1.0, gamma=1.0)
    agent.Q[3] = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    agent.learn(state=1, action=2, reward=0.0, next_state=3, terminated=False, truncated=True)
    assert agent.Q[1, 2] == 10.0


def test_learn_returns_empty_dict():
    agent = make_agent()
    result = agent.learn(state=0, action=0, reward=0.0, next_state=0, terminated=False, truncated=False)
    assert result == {}


def test_determinism_with_same_seed():
    agent_a = make_agent(seed=42, epsilon=0.5)
    agent_b = make_agent(seed=42, epsilon=0.5)
    states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    actions_a = [agent_a.act(s) for s in states * 5]
    actions_b = [agent_b.act(s) for s in states * 5]
    assert actions_a == actions_b
