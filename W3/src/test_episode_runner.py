import pytest
import gymnasium as gym
from episode_runner import EpisodeRunner
from blackjack_agent import BlackjackAgent


class TestEpisodeRunnerStub:
    """Tests for EpisodeRunner stub - verifies interface contracts."""

    def setup_method(self):
        self.env = gym.make("Blackjack-v1")
        self.runner = EpisodeRunner(self.env)
        self.agent = BlackjackAgent(epsilon=0.1, discount_factor=1.0)

    def test_run_episode_returns_list(self):
        episode = self.runner.run_episode(self.agent)
        assert isinstance(episode, list)

    def test_run_episode_returns_nonempty(self):
        episode = self.runner.run_episode(self.agent)
        assert len(episode) > 0

    def test_run_episode_tuple_structure(self):
        episode = self.runner.run_episode(self.agent)
        for step in episode:
            assert len(step) == 3, "Each step should be (state, action, reward)"
            state, action, reward = step
            assert isinstance(state, tuple) and len(state) == 3
            assert action in (0, 1)
            assert isinstance(reward, (int, float))

    def test_run_training_returns_rewards_list(self):
        rewards = self.runner.run_training(self.agent, num_episodes=5)
        assert isinstance(rewards, list)
        assert len(rewards) == 5

    def test_run_training_rewards_are_numeric(self):
        rewards = self.runner.run_training(self.agent, num_episodes=5)
        for r in rewards:
            assert isinstance(r, (int, float))
