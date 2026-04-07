import pytest


class TestEnvironmentManager:
    def test_n_states(self, env_manager):
        assert env_manager.n_states == 48

    def test_n_actions(self, env_manager):
        assert env_manager.n_actions == 4

    def test_reset_returns_start_state(self, env_manager):
        state = env_manager.reset(seed=42)
        assert state == 36

    def test_reset_returns_integer(self, env_manager):
        state = env_manager.reset()
        assert isinstance(state, int)

    def test_reset_state_in_valid_range(self, env_manager):
        state = env_manager.reset()
        assert 0 <= state < env_manager.n_states

    def test_step_returns_four_values(self, env_manager):
        env_manager.reset(seed=0)
        result = env_manager.step(action=0)
        assert len(result) == 4

    def test_step_state_is_integer(self, env_manager):
        env_manager.reset(seed=0)
        state, _, _, _ = env_manager.step(action=0)
        assert isinstance(state, int)

    def test_step_reward_is_float(self, env_manager):
        env_manager.reset(seed=0)
        _, reward, _, _ = env_manager.step(action=0)
        assert isinstance(reward, float)

    def test_step_normal_reward_is_minus_one(self, env_manager):
        env_manager.reset(seed=0)
        # Move up from start (state 36) -- safe, not cliff
        _, reward, _, _ = env_manager.step(action=0)
        assert reward == -1.0

    def test_step_terminated_and_truncated_are_bool(self, env_manager):
        env_manager.reset(seed=0)
        _, _, terminated, truncated = env_manager.step(action=0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
