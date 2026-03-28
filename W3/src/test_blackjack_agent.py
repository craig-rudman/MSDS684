import pytest
from blackjack_agent import BlackjackAgent


class TestBlackjackAgentStub:
    """Tests for BlackjackAgent stub - verifies interface contracts."""

    def setup_method(self):
        self.agent = BlackjackAgent(epsilon=0.1, discount_factor=1.0)

    def test_constructor_sets_epsilon(self):
        assert self.agent.epsilon == 0.1

    def test_constructor_sets_discount_factor(self):
        assert self.agent.discount_factor == 1.0

    def test_select_action_returns_valid_action(self):
        state = (18, 6, False)  # player_sum, dealer_card, usable_ace
        action = self.agent.select_action(state)
        assert action in (0, 1), f"Action must be 0 (stick) or 1 (hit), got {action}"

    def test_select_action_returns_int(self):
        state = (12, 10, True)
        action = self.agent.select_action(state)
        assert isinstance(action, int)

    def test_update_accepts_episode(self):
        episode = [
            ((18, 6, False), 1, 0.0),
            ((20, 6, False), 0, 1.0),
        ]
        # Should not raise
        self.agent.update(episode)

    def test_get_policy_returns_dict(self):
        policy = self.agent.get_policy()
        assert isinstance(policy, dict)

    def test_get_value_function_returns_dict(self):
        vf = self.agent.get_value_function()
        assert isinstance(vf, dict)
