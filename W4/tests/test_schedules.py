from schedules import ConstantSchedule, LinearDecaySchedule, ExponentialDecaySchedule


class TestConstantSchedule:
    def test_value_is_constant(self, constant_schedule):
        initial = constant_schedule.value
        constant_schedule.step()
        constant_schedule.step()
        assert constant_schedule.value == initial

    def test_value_matches_epsilon(self):
        s = ConstantSchedule(epsilon=0.25)
        assert s.value == 0.25

    def test_reset_preserves_value(self, constant_schedule):
        constant_schedule.step()
        constant_schedule.reset()
        assert constant_schedule.value == 0.1


class TestLinearDecaySchedule:
    def test_initial_value(self, linear_schedule):
        assert linear_schedule.value == 1.0

    def test_value_at_end(self):
        s = LinearDecaySchedule(epsilon_start=1.0, epsilon_end=0.01, n_episodes=10)
        for _ in range(10):
            s.step()
        assert s.value == pytest.approx(0.01)

    def test_does_not_go_below_epsilon_end(self):
        s = LinearDecaySchedule(epsilon_start=1.0, epsilon_end=0.01, n_episodes=5)
        for _ in range(100):
            s.step()
        assert s.value >= 0.01

    def test_decreases_monotonically(self):
        s = LinearDecaySchedule(epsilon_start=1.0, epsilon_end=0.0, n_episodes=10)
        values = []
        for _ in range(10):
            values.append(s.value)
            s.step()
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_reset_restores_initial_value(self, linear_schedule):
        for _ in range(100):
            linear_schedule.step()
        linear_schedule.reset()
        assert linear_schedule.value == 1.0


class TestExponentialDecaySchedule:
    def test_initial_value(self, exp_schedule):
        assert exp_schedule.value == 1.0

    def test_decreases_monotonically(self, exp_schedule):
        values = []
        for _ in range(10):
            values.append(exp_schedule.value)
            exp_schedule.step()
        assert all(values[i] > values[i + 1] for i in range(len(values) - 1))

    def test_never_reaches_zero(self, exp_schedule):
        for _ in range(10000):
            exp_schedule.step()
        assert exp_schedule.value > 0.0

    def test_decay_formula(self):
        s = ExponentialDecaySchedule(epsilon_start=1.0, decay_rate=0.5)
        s.step()
        assert s.value == pytest.approx(0.5)
        s.step()
        assert s.value == pytest.approx(0.25)

    def test_reset_restores_initial_value(self, exp_schedule):
        for _ in range(100):
            exp_schedule.step()
        exp_schedule.reset()
        assert exp_schedule.value == 1.0


import pytest
