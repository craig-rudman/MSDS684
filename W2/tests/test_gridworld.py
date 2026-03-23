"""Tests for src/gridworld.py -- GridWorld environment."""

import numpy as np
import pytest


class TestCoordinateHelpers:
    """state_to_rc and rc_to_state conversions."""

    def test_rc_to_state(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        assert env.rc_to_state(0, 0) == 0
        assert env.rc_to_state(0, 3) == 3
        assert env.rc_to_state(3, 3) == 15

    def test_state_to_rc(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        assert env.state_to_rc(0) == (0, 0)
        assert env.state_to_rc(3) == (0, 3)
        assert env.state_to_rc(15) == (3, 3)

    def test_round_trip(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        for s in range(env.n_states):
            r, c = env.state_to_rc(s)
            assert env.rc_to_state(r, c) == s


class TestConfigurableGeometry:
    """Obstacles, rewards, and terminal states are reflected in the model."""

    def test_obstacle_is_impassable(self, env_4x4_deterministic):
        """Moving into an obstacle cell keeps the agent in place."""
        env = env_4x4_deterministic
        obstacle_state = env.rc_to_state(1, 1)
        # From (1, 0) moving RIGHT should stay at (1, 0)
        from_state = env.rc_to_state(1, 0)
        next_state = env._get_next_state(from_state, env.RIGHT)
        assert next_state == from_state

    def test_terminal_state_reward(self, env_4x4_deterministic):
        """Entering the terminal cell yields the configured reward."""
        env = env_4x4_deterministic
        terminal = env.rc_to_state(3, 3)
        # From (3, 2) moving RIGHT should reach terminal with reward 1.0
        from_state = env.rc_to_state(3, 2)
        transitions = env.P[from_state][env.RIGHT]
        rewards = [r for (_, _, r, _) in transitions]
        assert any(r == 1.0 for r in rewards)

    def test_default_reward_applied(self, env_4x4_deterministic):
        """Non-special transitions yield the default reward."""
        env = env_4x4_deterministic
        from_state = env.rc_to_state(0, 0)
        transitions = env.P[from_state][env.RIGHT]
        rewards = [r for (_, _, r, _) in transitions]
        assert all(r == -0.04 for r in rewards)

    def test_terminal_state_is_done(self, env_4x4_deterministic):
        """Transitions into a terminal state flag done=True."""
        env = env_4x4_deterministic
        from_state = env.rc_to_state(3, 2)
        transitions = env.P[from_state][env.RIGHT]
        done_flags = [d for (_, s, _, d) in transitions if s == env.rc_to_state(3, 3)]
        assert all(done_flags)


class TestDeterministicTransitions:
    """Deterministic movement and boundary handling."""

    def test_move_right(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        s = env.rc_to_state(0, 0)
        ns = env._get_next_state(s, env.RIGHT)
        assert ns == env.rc_to_state(0, 1)

    def test_move_down(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        s = env.rc_to_state(0, 0)
        ns = env._get_next_state(s, env.DOWN)
        assert ns == env.rc_to_state(1, 0)

    def test_boundary_keeps_agent(self, env_4x4_deterministic):
        """Moving off the grid keeps the agent in place."""
        env = env_4x4_deterministic
        s = env.rc_to_state(0, 0)
        assert env._get_next_state(s, env.UP) == s
        assert env._get_next_state(s, env.LEFT) == s

    def test_deterministic_single_outcome(self, env_4x4_deterministic):
        """Each (s, a) in deterministic mode has exactly one transition with prob 1.0."""
        env = env_4x4_deterministic
        for s in range(env.n_states):
            for a in range(env.n_actions):
                transitions = env.P[s][a]
                assert len(transitions) == 1
                prob, _, _, _ = transitions[0]
                assert prob == pytest.approx(1.0)


class TestStochasticTransitions:
    """Stochastic transition probabilities."""

    def test_probabilities_sum_to_one(self, env_4x4_stochastic):
        env = env_4x4_stochastic
        for s in range(env.n_states):
            for a in range(env.n_actions):
                total = sum(p for p, _, _, _ in env.P[s][a])
                assert total == pytest.approx(1.0)

    def test_intended_direction_probability(self, env_4x4_stochastic):
        """The intended direction gets the configured probability."""
        env = env_4x4_stochastic
        # From center-ish cell (2, 0), moving RIGHT
        s = env.rc_to_state(2, 0)
        intended_next = env._get_next_state(s, env.RIGHT)
        transitions = env.P[s][env.RIGHT]
        intended_prob = sum(p for p, ns, _, _ in transitions if ns == intended_next)
        assert intended_prob == pytest.approx(0.8)

    def test_perpendicular_directions_share_remainder(self, env_4x4_stochastic):
        """The two perpendicular directions each get 10%."""
        env = env_4x4_stochastic
        s = env.rc_to_state(2, 0)
        # For action RIGHT, perpendicular actions are UP and DOWN
        perp_up_next = env._get_next_state(s, env.UP)
        perp_down_next = env._get_next_state(s, env.DOWN)
        transitions = env.P[s][env.RIGHT]
        for ns_expected in [perp_up_next, perp_down_next]:
            prob = sum(p for p, ns, _, _ in transitions if ns == ns_expected)
            assert prob == pytest.approx(0.1)


class TestTransitionModelCompleteness:
    """P[s][a] exists for all valid (s, a) pairs."""

    def test_all_states_present(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        assert set(env.P.keys()) == set(range(env.n_states))

    def test_all_actions_present(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        for s in range(env.n_states):
            assert set(env.P[s].keys()) == set(range(env.n_actions))

    def test_transition_tuple_shape(self, env_4x4_deterministic):
        """Each entry is a list of (prob, next_state, reward, done) tuples."""
        env = env_4x4_deterministic
        for s in range(env.n_states):
            for a in range(env.n_actions):
                for entry in env.P[s][a]:
                    assert len(entry) == 4
                    prob, ns, reward, done = entry
                    assert 0.0 <= prob <= 1.0
                    assert 0 <= ns < env.n_states
                    assert isinstance(done, bool)


class TestGymnasiumAPI:
    """reset() and step() conform to the Gymnasium interface."""

    def test_reset_returns_valid_state(self, env_4x4_deterministic):
        obs, info = env_4x4_deterministic.reset()
        assert 0 <= obs < env_4x4_deterministic.n_states
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, env_4x4_deterministic):
        env = env_4x4_deterministic
        env.reset()
        result = env.step(env.RIGHT)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert 0 <= obs < env.n_states
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_terminal_ends_episode(self, env_2x2_simple):
        """Stepping into the terminal state sets terminated=True."""
        env = env_2x2_simple
        env.reset()
        # In a 2x2 grid starting at (0,0), moving DOWN then RIGHT reaches (1,1)
        env.step(env.DOWN)
        _, _, terminated, _, _ = env.step(env.RIGHT)
        assert terminated is True
