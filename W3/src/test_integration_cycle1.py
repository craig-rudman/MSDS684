import pytest
import gymnasium as gym
from blackjack_agent import BlackjackAgent
from episode_runner import EpisodeRunner


class TestAgentRunnerIntegration:
    """Integration tests: BlackjackAgent + EpisodeRunner work together."""

    def setup_method(self):
        self.env = gym.make("Blackjack-v1", sab=True)
        self.agent = BlackjackAgent(epsilon=0.1, discount_factor=1.0)
        self.runner = EpisodeRunner(self.env)

    def test_episode_feeds_into_agent_update(self):
        episode = self.runner.run_episode(self.agent)
        # Agent update should accept the episode without error
        self.agent.update(episode)

    def test_training_loop_updates_agent(self):
        rewards = self.runner.run_training(self.agent, num_episodes=10)
        # After training, agent should have some Q-value entries
        policy = self.agent.get_policy()
        assert len(policy) > 0, "Agent should have learned something after 10 episodes"

    def test_training_loop_collects_rewards(self):
        rewards = self.runner.run_training(self.agent, num_episodes=10)
        assert len(rewards) == 10
        for r in rewards:
            assert isinstance(r, (int, float))

    def test_agent_actions_are_valid_throughout_episode(self):
        episode = self.runner.run_episode(self.agent)
        for state, action, reward in episode:
            player_sum, dealer_card, usable_ace = state
            assert 4 <= player_sum <= 21
            assert 1 <= dealer_card <= 10
            assert usable_ace in (True, False, 0, 1)
            assert action in (0, 1)
