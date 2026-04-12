import gymnasium as gym
import pytest

from src.environment import MountainCarEnvironment


class TestMountainCarEnvironment:
    @pytest.fixture
    def env(self):
        e = MountainCarEnvironment()
        yield e
        e.close()

    def test_is_singleton(self):
        assert MountainCarEnvironment() is MountainCarEnvironment()

    def test_observation_space_matches_gymnasium(self, env):
        expected = gym.make("MountainCar-v0").observation_space
        assert env.observation_space == expected

    def test_action_space_matches_gymnasium(self, env):
        expected = gym.make("MountainCar-v0").action_space
        assert env.action_space == expected

    def test_reset_returns_observation_in_space(self, env):
        obs, info = env.reset()
        assert env.observation_space.contains(obs)

    def test_step_returns_expected_tuple(self, env):
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_close_does_not_raise(self, env):
        env.reset()
        env.close()  # should not raise
