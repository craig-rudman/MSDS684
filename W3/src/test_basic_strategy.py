import pytest
import gymnasium as gym
from basic_strategy import BasicStrategy
from episode_runner import EpisodeRunner


class TestBasicStrategyActions:
    """Tests for BasicStrategy action lookup."""

    def setup_method(self):
        self.strategy = BasicStrategy()

    # Hard totals - obvious cases
    def test_stand_on_20(self):
        assert self.strategy.get_action((20, 6, False)) == 0  # stick

    def test_stand_on_19(self):
        assert self.strategy.get_action((19, 10, False)) == 0  # stick

    def test_hit_on_8(self):
        assert self.strategy.get_action((8, 10, False)) == 1  # hit

    def test_hit_on_5(self):
        assert self.strategy.get_action((5, 6, False)) == 1  # hit

    # Hard totals - boundary cases
    def test_hit_on_12_vs_dealer_2(self):
        assert self.strategy.get_action((12, 2, False)) == 1  # hit

    def test_stand_on_12_vs_dealer_4(self):
        assert self.strategy.get_action((12, 4, False)) == 0  # stick

    def test_hit_on_16_vs_dealer_10(self):
        assert self.strategy.get_action((16, 10, False)) == 1  # hit

    def test_stand_on_16_vs_dealer_6(self):
        assert self.strategy.get_action((16, 6, False)) == 0  # stick

    # Soft totals (usable ace)
    def test_stand_on_soft_20(self):
        assert self.strategy.get_action((20, 6, True)) == 0  # stick

    def test_stand_on_soft_19(self):
        assert self.strategy.get_action((19, 6, True)) == 0  # stick

    def test_stand_on_soft_18_vs_dealer_6(self):
        assert self.strategy.get_action((18, 6, True)) == 0  # stick

    def test_hit_on_soft_18_vs_dealer_9(self):
        assert self.strategy.get_action((18, 9, True)) == 1  # hit

    def test_hit_on_soft_17(self):
        assert self.strategy.get_action((17, 6, True)) == 1  # hit

    def test_hit_on_soft_13(self):
        assert self.strategy.get_action((13, 6, True)) == 1  # hit


class TestBasicStrategyIntegration:
    """Integration: BasicStrategy with EpisodeRunner."""

    def test_basic_strategy_runs_episodes(self):
        """BasicStrategy can be used as an agent with EpisodeRunner."""
        env = gym.make("Blackjack-v1", sab=True)
        strategy = BasicStrategy()
        runner = EpisodeRunner(env)
        episode = runner.run_episode(strategy)
        assert len(episode) > 0
        for state, action, reward in episode:
            assert action in (0, 1)

    def test_basic_strategy_runs_training_loop(self):
        """BasicStrategy can run through run_training (update is a no-op)."""
        env = gym.make("Blackjack-v1", sab=True)
        strategy = BasicStrategy()
        runner = EpisodeRunner(env)
        rewards = runner.run_training(strategy, num_episodes=100)
        assert len(rewards) == 100
