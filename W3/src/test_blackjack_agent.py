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


class TestBlackjackAgentEpsilonDecay:
    """Tests for injectable epsilon decay."""

    def test_linear_decay(self):
        decay = lambda eps: max(0.0, eps - 0.1)
        agent = BlackjackAgent(epsilon=1.0, decay_schedule=decay)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.9)

    def test_exponential_decay(self):
        decay = lambda eps: eps * 0.99
        agent = BlackjackAgent(epsilon=1.0, decay_schedule=decay)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.99)

    def test_decay_with_floor(self):
        decay = lambda eps: max(0.01, eps - 0.1)
        agent = BlackjackAgent(epsilon=0.05, decay_schedule=decay)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.01)

    def test_floor_holds(self):
        decay = lambda eps: max(0.01, eps - 0.1)
        agent = BlackjackAgent(epsilon=0.01, decay_schedule=decay)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.01)

    def test_no_decay_by_default(self):
        agent = BlackjackAgent(epsilon=0.1)
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.1)

    def test_multiple_linear_decays(self):
        decay = lambda eps: max(0.0, eps - 0.1)
        agent = BlackjackAgent(epsilon=1.0, decay_schedule=decay)
        for _ in range(10):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.0)

    def test_multiple_exponential_decays(self):
        decay = lambda eps: eps * 0.5
        agent = BlackjackAgent(epsilon=1.0, decay_schedule=decay)
        agent.decay_epsilon()
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.25)

    def test_custom_callable_object(self):
        """A callable object with internal state can serve as a decay schedule."""
        class CosineDecay:
            def __init__(self, initial, minimum, total_steps):
                self.initial = initial
                self.minimum = minimum
                self.total_steps = total_steps
                self.step = 0

            def __call__(self, eps):
                import math
                self.step += 1
                progress = min(self.step / self.total_steps, 1.0)
                return self.minimum + 0.5 * (self.initial - self.minimum) * (1 + math.cos(math.pi * progress))

        decay = CosineDecay(initial=1.0, minimum=0.01, total_steps=100)
        agent = BlackjackAgent(epsilon=1.0, decay_schedule=decay)
        agent.decay_epsilon()
        assert 0.01 <= agent.epsilon <= 1.0
        for _ in range(99):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.01)
