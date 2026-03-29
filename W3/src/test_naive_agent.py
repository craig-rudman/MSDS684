import pytest
import gymnasium as gym
from naive_agent import NaiveAgent
from episode_runner import EpisodeRunner


class TestNaiveAgentActions:
    """Tests for NaiveAgent action selection."""

    def test_random_returns_valid_action(self):
        agent = NaiveAgent(strategy="random")
        state = (16, 10, False)
        action = agent.select_action(state)
        assert action in (0, 1)

    def test_always_hit_returns_hit(self):
        agent = NaiveAgent(strategy="always_hit")
        state = (16, 10, False)
        assert agent.select_action(state) == 1

    def test_always_stick_returns_stick(self):
        agent = NaiveAgent(strategy="always_stick")
        state = (16, 10, False)
        assert agent.select_action(state) == 0

    def test_select_action_returns_int(self):
        agent = NaiveAgent(strategy="random")
        state = (18, 6, False)
        assert isinstance(agent.select_action(state), int)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            NaiveAgent(strategy="invalid")


class TestNaiveAgentIntegration:
    """Integration: NaiveAgent with EpisodeRunner."""

    def test_runs_episodes(self):
        env = gym.make("Blackjack-v1", sab=True)
        agent = NaiveAgent(strategy="random")
        runner = EpisodeRunner(env)
        episode = runner.run_episode(agent)
        assert len(episode) > 0
        for state, action, reward in episode:
            assert action in (0, 1)

    def test_runs_training_loop(self):
        env = gym.make("Blackjack-v1", sab=True)
        agent = NaiveAgent(strategy="random")
        runner = EpisodeRunner(env)
        rewards = runner.run_training(agent, num_episodes=100)
        assert len(rewards) == 100
        for r in rewards:
            assert isinstance(r, (int, float))
