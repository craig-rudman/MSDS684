import numpy as np
import pytest

from src.sarsa_agent import SarsaAgent
from src.tile_coder import TileCoder


@pytest.fixture
def agent():
    tc = TileCoder()
    return SarsaAgent(tc, n_actions=3)


class TestSarsaAgentConstruction:
    def test_default_hyperparameters(self, agent):
        assert agent.alpha == 0.1
        assert agent.gamma == 1.0
        assert agent.epsilon == 0.1

    def test_custom_hyperparameters(self):
        tc = TileCoder()
        a = SarsaAgent(tc, n_actions=3, alpha=0.5, gamma=0.99, epsilon=0.2)
        assert a.alpha == 0.5
        assert a.gamma == 0.99
        assert a.epsilon == 0.2

    def test_alpha_eff_prorated_by_n_tilings(self, agent):
        assert agent.alpha_eff == agent.alpha / agent.tile_coder.n_tilings

    def test_weights_shape(self, agent):
        assert agent.weights.shape == (3, agent.tile_coder.num_features)

    def test_weights_initialized_to_zero(self, agent):
        assert np.all(agent.weights == 0.0)


class TestSarsaAgentQValue:
    def test_q_value_zero_with_zero_weights(self, agent):
        assert agent.q_value([-0.5, 0.0], 0) == 0.0

    def test_q_value_returns_float(self, agent):
        assert isinstance(agent.q_value([-0.5, 0.0], 0), float)


class TestSarsaAgentActionSelection:
    def test_greedy_action_in_valid_range(self, agent):
        assert 0 <= agent.greedy_action([-0.5, 0.0]) < 3

    def test_select_action_in_valid_range(self, agent):
        assert 0 <= agent.select_action([-0.5, 0.0]) < 3

    def test_greedy_with_epsilon_zero_matches_greedy_action(self):
        tc = TileCoder()
        a = SarsaAgent(tc, n_actions=3, epsilon=0.0)
        state = [-0.5, 0.0]
        assert a.select_action(state) == a.greedy_action(state)


class TestSarsaAgentUpdate:
    def test_update_changes_weights(self, agent):
        state, next_state = [-0.5, 0.0], [-0.4, 0.01]
        weights_before = agent.weights.copy()
        agent.update(state, 0, -1.0, next_state, 1, False)
        assert not np.all(agent.weights == weights_before)

    def test_update_on_terminal_step(self, agent):
        state, next_state = [-0.5, 0.0], [0.5, 0.05]
        agent.update(state, 0, -1.0, next_state, 0, True)


class TestSarsaAgentResetWeights:
    def test_reset_weights_returns_to_zero(self, agent):
        state, next_state = [-0.5, 0.0], [-0.4, 0.01]
        agent.update(state, 0, -1.0, next_state, 1, False)
        agent.reset_weights()
        assert np.all(agent.weights == 0.0)
