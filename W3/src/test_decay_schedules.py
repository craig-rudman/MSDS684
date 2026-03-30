import pytest
import math
from decay_schedules import LinearDecay, ExponentialDecay, CosineDecay, StepDecay, AdaptiveDecay


class TestLinearDecay:
    def test_decays_by_fixed_amount(self):
        decay = LinearDecay(eps_min=0.01, decay_rate=0.1)
        assert decay(1.0) == pytest.approx(0.9)

    def test_respects_floor(self):
        decay = LinearDecay(eps_min=0.01, decay_rate=0.1)
        assert decay(0.05) == pytest.approx(0.01)

    def test_at_floor_stays(self):
        decay = LinearDecay(eps_min=0.01, decay_rate=0.1)
        assert decay(0.01) == pytest.approx(0.01)


class TestExponentialDecay:
    def test_decays_by_factor(self):
        decay = ExponentialDecay(eps_min=0.01, decay_factor=0.99)
        assert decay(1.0) == pytest.approx(0.99)

    def test_respects_floor(self):
        decay = ExponentialDecay(eps_min=0.5, decay_factor=0.99)
        assert decay(0.5) == pytest.approx(0.5)

    def test_multiple_decays(self):
        decay = ExponentialDecay(eps_min=0.01, decay_factor=0.5)
        eps = 1.0
        eps = decay(eps)
        eps = decay(eps)
        assert eps == pytest.approx(0.25)


class TestCosineDecay:
    def test_starts_near_initial(self):
        decay = CosineDecay(eps_start=1.0, eps_min=0.01, total_steps=1000)
        eps = decay(1.0)
        assert eps > 0.99

    def test_ends_at_floor(self):
        decay = CosineDecay(eps_start=1.0, eps_min=0.01, total_steps=100)
        eps = 1.0
        for _ in range(100):
            eps = decay(eps)
        assert eps == pytest.approx(0.01)

    def test_midpoint_is_roughly_halfway(self):
        decay = CosineDecay(eps_start=1.0, eps_min=0.0, total_steps=100)
        eps = 1.0
        for _ in range(50):
            eps = decay(eps)
        assert eps == pytest.approx(0.5, abs=0.05)

    def test_clamps_beyond_total_steps(self):
        decay = CosineDecay(eps_start=1.0, eps_min=0.01, total_steps=10)
        eps = 1.0
        for _ in range(20):
            eps = decay(eps)
        assert eps == pytest.approx(0.01)


class TestStepDecay:
    def test_holds_between_steps(self):
        decay = StepDecay(eps_min=0.01, drop_factor=0.5, drop_every=100)
        eps = 1.0
        for _ in range(99):
            eps = decay(eps)
        assert eps == pytest.approx(1.0)

    def test_drops_at_interval(self):
        decay = StepDecay(eps_min=0.01, drop_factor=0.5, drop_every=100)
        eps = 1.0
        for _ in range(100):
            eps = decay(eps)
        assert eps == pytest.approx(0.5)

    def test_multiple_drops(self):
        decay = StepDecay(eps_min=0.01, drop_factor=0.5, drop_every=100)
        eps = 1.0
        for _ in range(200):
            eps = decay(eps)
        assert eps == pytest.approx(0.25)

    def test_respects_floor(self):
        decay = StepDecay(eps_min=0.1, drop_factor=0.5, drop_every=10)
        eps = 1.0
        for _ in range(100):
            eps = decay(eps)
        assert eps == pytest.approx(0.1)


class TestAdaptiveDecay:
    def test_decays_when_signal_high(self):
        """High exploration signal means confident estimates, so decay faster."""
        mock_agent = type("Agent", (), {"get_exploration_signal": lambda self: 100.0})()
        decay = AdaptiveDecay(eps_min=0.01, agent=mock_agent)
        eps = decay(1.0)
        assert eps < 1.0

    def test_holds_when_signal_low(self):
        """Low exploration signal means uncertain, so decay slowly or not at all."""
        mock_agent = type("Agent", (), {"get_exploration_signal": lambda self: 0.0})()
        decay = AdaptiveDecay(eps_min=0.01, agent=mock_agent)
        eps = decay(1.0)
        assert eps >= 0.99

    def test_respects_floor(self):
        mock_agent = type("Agent", (), {"get_exploration_signal": lambda self: 1000.0})()
        decay = AdaptiveDecay(eps_min=0.05, agent=mock_agent)
        eps = decay(0.05)
        assert eps == pytest.approx(0.05)

    def test_is_callable(self):
        mock_agent = type("Agent", (), {"get_exploration_signal": lambda self: 1.0})()
        decay = AdaptiveDecay(eps_min=0.01, agent=mock_agent)
        assert callable(decay)
