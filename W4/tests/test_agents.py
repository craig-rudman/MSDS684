import numpy as np
import pytest
from agents import SARSAAgent, QLearningAgent
from schedules import ConstantSchedule


N_STATES = 48
N_ACTIONS = 4


class TestTDAgentBase:
    def test_qtable_shape(self, sarsa_agent):
        assert sarsa_agent.Q.shape == (N_STATES, N_ACTIONS)

    def test_qtable_dtype_is_float(self, sarsa_agent):
        assert sarsa_agent.Q.dtype == float

    def test_qtable_initialized_to_zeros(self, sarsa_agent):
        assert np.all(sarsa_agent.Q == 0.0)

    def test_reset_qtable_clears_values(self, sarsa_agent):
        sarsa_agent.Q[0, 0] = 99.0
        sarsa_agent.reset_qtable()
        assert np.all(sarsa_agent.Q == 0.0)

    def test_select_action_valid_range(self, sarsa_agent):
        action = sarsa_agent.select_action(0)
        assert 0 <= action < N_ACTIONS

    def test_select_action_explores_when_epsilon_one(self):
        schedule = ConstantSchedule(epsilon=1.0)
        agent = SARSAAgent(N_STATES, N_ACTIONS, alpha=0.1, gamma=1.0, schedule=schedule)
        actions = {agent.select_action(0) for _ in range(200)}
        assert len(actions) > 1

    def test_select_action_greedy_when_epsilon_zero(self):
        schedule = ConstantSchedule(epsilon=0.0)
        agent = SARSAAgent(N_STATES, N_ACTIONS, alpha=0.1, gamma=1.0, schedule=schedule)
        agent.Q[0, 2] = 5.0
        actions = {agent.select_action(0) for _ in range(20)}
        assert actions == {2}

    def test_decay_epsilon_delegates_to_schedule(self, sarsa_agent, constant_schedule):
        before = constant_schedule.value
        sarsa_agent.decay_epsilon()
        assert constant_schedule.value == before  # ConstantSchedule should not change


class TestSARSAUpdate:
    def test_qvalue_moves_toward_target(self, sarsa_agent):
        sarsa_agent.Q[0, 0] = 0.0
        sarsa_agent.Q[1, 1] = 2.0
        sarsa_agent.update(s=0, a=0, r=-1.0, s_next=1, a_next=1, terminated=False)
        assert sarsa_agent.Q[0, 0] != 0.0

    def test_terminal_state_zero_bootstrap(self, sarsa_agent):
        sarsa_agent.Q[0, 0] = 0.0
        sarsa_agent.Q[1, 1] = 100.0
        sarsa_agent.update(s=0, a=0, r=-1.0, s_next=1, a_next=1, terminated=True)
        # target should be r + 0, not r + gamma * Q[s_next, a_next]
        expected = 0.0 + 0.1 * (-1.0 - 0.0)
        assert sarsa_agent.Q[0, 0] == pytest.approx(expected)


class TestQLearningUpdate:
    def test_qvalue_moves_toward_target(self, qlearning_agent):
        qlearning_agent.Q[0, 0] = 0.0
        qlearning_agent.Q[1, :] = [1.0, 2.0, 3.0, 0.5]
        qlearning_agent.update(s=0, a=0, r=-1.0, s_next=1, a_next=0, terminated=False)
        assert qlearning_agent.Q[0, 0] != 0.0

    def test_uses_max_not_a_next(self, qlearning_agent):
        qlearning_agent.Q[0, 0] = 0.0
        qlearning_agent.Q[1, :] = [0.0, 0.0, 10.0, 0.0]
        # a_next=0 has value 0, but max is 10 at action 2
        qlearning_agent.update(s=0, a=0, r=-1.0, s_next=1, a_next=0, terminated=False)
        expected = 0.0 + 0.1 * (-1.0 + 1.0 * 10.0 - 0.0)
        assert qlearning_agent.Q[0, 0] == pytest.approx(expected)

    def test_terminal_state_zero_bootstrap(self, qlearning_agent):
        qlearning_agent.Q[0, 0] = 0.0
        qlearning_agent.Q[1, :] = [100.0, 100.0, 100.0, 100.0]
        qlearning_agent.update(s=0, a=0, r=-1.0, s_next=1, a_next=0, terminated=True)
        expected = 0.0 + 0.1 * (-1.0 - 0.0)
        assert qlearning_agent.Q[0, 0] == pytest.approx(expected)
