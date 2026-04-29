import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def make_agent(n_planning=5, theta=0.01, num_states=10, num_actions=6,
               alpha=0.1, epsilon=0.0, gamma=0.99, seed=0):
    from prioritized_sweeping_agent import PrioritizedSweepingAgent
    return PrioritizedSweepingAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        n_planning=n_planning,
        theta=theta,
        seed=seed,
    )


def test_is_subclass_of_dyna_q_agent():
    from dyna_q_agent import DynaQAgent
    agent = make_agent()
    assert isinstance(agent, DynaQAgent)


def test_priority_queue_empty_on_init():
    agent = make_agent()
    assert len(agent.priority_queue) == 0


def test_reverse_model_empty_on_init():
    agent = make_agent()
    assert len(agent.reverse_model) == 0


def test_learn_adds_real_transition_to_queue():
    agent = make_agent(n_planning=0)
    agent.learn(state=0, action=0, reward=1.0, next_state=1,
                terminated=False, truncated=False)
    assert len(agent.priority_queue) > 0


def test_learn_priority_equals_td_error():
    # alpha=1, gamma=0: TD error = |r + 0 - Q[s,a]| = |1.0 - 0.0| = 1.0
    agent = make_agent(n_planning=0, alpha=1.0, gamma=0.0, theta=0.0)
    agent.learn(state=0, action=0, reward=1.0, next_state=1,
                terminated=False, truncated=False)
    # heapq is a min-heap; priorities stored as negatives
    neg_priority, s, a = agent.priority_queue[0]
    assert abs(neg_priority) == pytest.approx(1.0)
    assert (s, a) == (0, 0)


def test_planning_pops_highest_priority_first():
    # Insert two transitions with different TD errors; planning should
    # update the higher-priority (s,a) first.
    agent = make_agent(n_planning=1, alpha=1.0, gamma=0.0, theta=0.0)
    # Manually seed the model and queue with two pairs
    agent.model.update(s=0, a=0, r=10.0, s_next=2)
    agent.model.update(s=1, a=0, r=1.0,  s_next=2)
    import heapq
    heapq.heappush(agent.priority_queue, (-10.0, 0, 0))
    heapq.heappush(agent.priority_queue,  (-1.0, 1, 0))

    q_before_low = float(agent.Q[1, 0])
    agent._run_planning()
    # High-priority pair (0,0) should have been updated; low-priority not yet
    assert agent.Q[0, 0] != q_before_low or agent.Q[0, 0] == pytest.approx(10.0)


def test_reverse_model_updated_after_learn():
    agent = make_agent(n_planning=0)
    agent.learn(state=0, action=0, reward=1.0, next_state=1,
                terminated=False, truncated=False)
    assert (0, 0) in agent.reverse_model.get(1, set())


def test_predecessors_above_threshold_added_to_queue():
    # n_planning=2: step 1 processes (1,0) and pushes predecessor (0,0),
    # step 2 processes (0,0). Check that Q[0,0] was updated via propagation.
    agent = make_agent(n_planning=2, alpha=1.0, gamma=0.99, theta=0.0)
    agent.learn(state=0, action=0, reward=0.0, next_state=1,
                terminated=False, truncated=False)
    agent.learn(state=1, action=0, reward=20.0, next_state=2,
                terminated=False, truncated=False)
    # Q[0,0] should be non-zero: the high reward at s=1 propagated back
    assert agent.Q[0, 0] == pytest.approx(0.99 * 20.0)


def test_learn_returns_planning_stats():
    agent = make_agent(n_planning=5)
    stats = agent.learn(state=0, action=0, reward=1.0, next_state=1,
                        terminated=False, truncated=False)
    assert "n_planning_updates" in stats
    assert "wall_time_planning" in stats
    assert stats["wall_time_planning"] >= 0.0
