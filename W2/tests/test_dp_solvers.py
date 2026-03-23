"""Tests for src/dp_solvers.py -- PolicyIteration and ValueIteration."""

import numpy as np
import pytest


class TestPolicyIterationSync:
    """Synchronous policy iteration."""

    def test_converges_on_deterministic(self, pi_deterministic):
        V, policy = pi_deterministic.solve(mode="sync")
        assert V is not None
        assert policy is not None

    def test_terminal_state_value_zero(self, pi_deterministic):
        """Terminal states should have value 0."""
        V, _ = pi_deterministic.solve(mode="sync")
        env = pi_deterministic.env
        for t in env.terminal_states:
            ts = env.rc_to_state(*t)
            assert V[ts] == pytest.approx(0.0)

    def test_value_history_recorded(self, pi_deterministic):
        pi_deterministic.solve(mode="sync")
        assert len(pi_deterministic.value_history) > 0

    def test_policy_history_recorded(self, pi_deterministic):
        pi_deterministic.solve(mode="sync")
        assert len(pi_deterministic.policy_history) > 0

    def test_wall_clock_times_recorded(self, pi_deterministic):
        pi_deterministic.solve(mode="sync")
        assert len(pi_deterministic.wall_clock_times) > 0
        assert all(t >= 0 for t in pi_deterministic.wall_clock_times)


class TestPolicyIterationInplace:
    """In-place policy iteration."""

    def test_converges_on_deterministic(self, pi_deterministic):
        V, policy = pi_deterministic.solve(mode="inplace")
        assert V is not None
        assert policy is not None

    def test_terminal_state_value_zero(self, pi_deterministic):
        V, _ = pi_deterministic.solve(mode="inplace")
        env = pi_deterministic.env
        for t in env.terminal_states:
            ts = env.rc_to_state(*t)
            assert V[ts] == pytest.approx(0.0)


class TestPolicyIterationSyncVsInplace:
    """Sync and in-place produce equivalent results."""

    def test_same_final_values(self, env_4x4_deterministic):
        from src import PolicyIteration

        pi_sync = PolicyIteration(env_4x4_deterministic)
        V_sync, _ = pi_sync.solve(mode="sync")

        pi_inp = PolicyIteration(env_4x4_deterministic)
        V_inp, _ = pi_inp.solve(mode="inplace")

        np.testing.assert_allclose(V_sync, V_inp, atol=1e-6)

    def test_same_final_policy(self, env_4x4_deterministic):
        from src import PolicyIteration

        pi_sync = PolicyIteration(env_4x4_deterministic)
        _, pol_sync = pi_sync.solve(mode="sync")

        pi_inp = PolicyIteration(env_4x4_deterministic)
        _, pol_inp = pi_inp.solve(mode="inplace")

        np.testing.assert_array_equal(pol_sync, pol_inp)


class TestValueIterationSync:
    """Synchronous value iteration."""

    def test_converges_on_deterministic(self, vi_deterministic):
        V, policy = vi_deterministic.solve(mode="sync")
        assert V is not None
        assert policy is not None

    def test_terminal_state_value_zero(self, vi_deterministic):
        V, _ = vi_deterministic.solve(mode="sync")
        env = vi_deterministic.env
        for t in env.terminal_states:
            ts = env.rc_to_state(*t)
            assert V[ts] == pytest.approx(0.0)

    def test_value_history_recorded(self, vi_deterministic):
        vi_deterministic.solve(mode="sync")
        assert len(vi_deterministic.value_history) > 0

    def test_wall_clock_times_recorded(self, vi_deterministic):
        vi_deterministic.solve(mode="sync")
        assert len(vi_deterministic.wall_clock_times) > 0


class TestValueIterationInplace:
    """In-place value iteration."""

    def test_converges_on_deterministic(self, vi_deterministic):
        V, policy = vi_deterministic.solve(mode="inplace")
        assert V is not None
        assert policy is not None

    def test_terminal_state_value_zero(self, vi_deterministic):
        V, _ = vi_deterministic.solve(mode="inplace")
        env = vi_deterministic.env
        for t in env.terminal_states:
            ts = env.rc_to_state(*t)
            assert V[ts] == pytest.approx(0.0)


class TestValueIterationSyncVsInplace:
    """Sync and in-place produce equivalent results."""

    def test_same_final_values(self, env_4x4_deterministic):
        from src import ValueIteration

        vi_sync = ValueIteration(env_4x4_deterministic)
        V_sync, _ = vi_sync.solve(mode="sync")

        vi_inp = ValueIteration(env_4x4_deterministic)
        V_inp, _ = vi_inp.solve(mode="inplace")

        np.testing.assert_allclose(V_sync, V_inp, atol=1e-6)

    def test_same_final_policy(self, env_4x4_deterministic):
        from src import ValueIteration

        vi_sync = ValueIteration(env_4x4_deterministic)
        _, pol_sync = vi_sync.solve(mode="sync")

        vi_inp = ValueIteration(env_4x4_deterministic)
        _, pol_inp = vi_inp.solve(mode="inplace")

        np.testing.assert_array_equal(pol_sync, pol_inp)


class TestPIvsVIAgreement:
    """Policy iteration and value iteration should agree on optimal solution."""

    def test_same_optimal_values(self, env_4x4_deterministic):
        from src import PolicyIteration, ValueIteration

        pi = PolicyIteration(env_4x4_deterministic)
        V_pi, _ = pi.solve(mode="sync")

        vi = ValueIteration(env_4x4_deterministic)
        V_vi, _ = vi.solve(mode="sync")

        np.testing.assert_allclose(V_pi, V_vi, atol=1e-6)

    def test_same_optimal_policy(self, env_4x4_deterministic):
        from src import PolicyIteration, ValueIteration

        pi = PolicyIteration(env_4x4_deterministic)
        _, pol_pi = pi.solve(mode="sync")

        vi = ValueIteration(env_4x4_deterministic)
        _, pol_vi = vi.solve(mode="sync")

        np.testing.assert_array_equal(pol_pi, pol_vi)


class TestStochasticSolvers:
    """Solvers work correctly under stochastic dynamics."""

    def test_pi_converges_stochastic(self, pi_stochastic):
        V, policy = pi_stochastic.solve(mode="sync")
        assert V is not None
        assert policy is not None

    def test_vi_converges_stochastic(self, vi_stochastic):
        V, policy = vi_stochastic.solve(mode="sync")
        assert V is not None
        assert policy is not None

    def test_pi_vi_agree_stochastic(self, env_4x4_stochastic):
        from src import PolicyIteration, ValueIteration

        pi = PolicyIteration(env_4x4_stochastic)
        V_pi, _ = pi.solve(mode="sync")

        vi = ValueIteration(env_4x4_stochastic)
        V_vi, _ = vi.solve(mode="sync")

        np.testing.assert_allclose(V_pi, V_vi, atol=1e-6)


class TestAnalytical2x2:
    """Verify against a hand-solvable 2x2 grid."""

    def test_optimal_policy_points_to_goal(self, env_2x2_simple):
        """In a 2x2 grid with goal at (1,1), optimal moves from each cell
        should head toward the goal."""
        from src import ValueIteration

        vi = ValueIteration(env_2x2_simple, gamma=0.99)
        _, policy = vi.solve(mode="sync")

        env = env_2x2_simple
        # From (0,0): RIGHT or DOWN are both optimal paths
        assert policy[env.rc_to_state(0, 0)] in [env.RIGHT, env.DOWN]
        # From (0,1): DOWN leads to goal
        assert policy[env.rc_to_state(0, 1)] == env.DOWN
        # From (1,0): RIGHT leads to goal
        assert policy[env.rc_to_state(1, 0)] == env.RIGHT
