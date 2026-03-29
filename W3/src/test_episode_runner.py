import pytest
import gymnasium as gym
from io import StringIO
from episode_runner import EpisodeRunner
from blackjack_agent import BlackjackAgent


class TestEpisodeRunnerStub:
    """Tests for EpisodeRunner stub - verifies interface contracts."""

    def setup_method(self):
        self.env = gym.make("Blackjack-v1", sab=True)
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


class TestEpisodeRunnerVerbose:
    """Tests for verbose episode output."""

    def setup_method(self):
        self.env = gym.make("Blackjack-v1", sab=True)
        self.runner = EpisodeRunner(self.env)
        self.agent = BlackjackAgent(epsilon=0.1, discount_factor=1.0)

    def test_verbose_false_produces_no_output(self, capsys):
        self.runner.run_episode(self.agent, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_produces_output(self, capsys):
        self.runner.run_episode(self.agent, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_verbose_output_contains_step_info(self, capsys):
        self.runner.run_episode(self.agent, verbose=True)
        captured = capsys.readouterr()
        assert "Step" in captured.out
        assert "State" in captured.out

    def test_verbose_default_is_false(self, capsys):
        self.runner.run_episode(self.agent)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestEpisodeRunnerCallback:
    """Tests for training callback functionality."""

    def setup_method(self):
        self.env = gym.make("Blackjack-v1", sab=True)
        self.runner = EpisodeRunner(self.env)
        self.agent = BlackjackAgent(epsilon=0.1, discount_factor=1.0)

    def test_callback_is_called(self):
        calls = []
        def cb(episode_num, rewards, agent):
            calls.append(episode_num)
        self.runner.run_training(self.agent, num_episodes=100,
                                 callback=cb, callback_interval=50)
        assert len(calls) == 2  # called at episode 50 and 100

    def test_callback_receives_correct_episode_num(self):
        seen = []
        def cb(episode_num, rewards, agent):
            seen.append(episode_num)
        self.runner.run_training(self.agent, num_episodes=100,
                                 callback=cb, callback_interval=25)
        assert seen == [25, 50, 75, 100]

    def test_callback_receives_rewards_list(self):
        captured_rewards = []
        def cb(episode_num, rewards, agent):
            captured_rewards.append(len(rewards))
        self.runner.run_training(self.agent, num_episodes=100,
                                 callback=cb, callback_interval=50)
        assert captured_rewards == [50, 100]

    def test_callback_receives_agent(self):
        captured_agents = []
        def cb(episode_num, rewards, agent):
            captured_agents.append(agent)
        self.runner.run_training(self.agent, num_episodes=50,
                                 callback=cb, callback_interval=50)
        assert captured_agents[0] is self.agent

    def test_no_callback_by_default(self):
        # Should not raise even without callback
        rewards = self.runner.run_training(self.agent, num_episodes=10)
        assert len(rewards) == 10

    def test_callback_with_decay(self):
        decay_agent = BlackjackAgent(epsilon=1.0,
                                     decay_schedule=lambda eps: max(0.01, eps - 0.001))
        calls = []
        def cb(episode_num, rewards, agent):
            calls.append(agent.epsilon)
        self.runner.run_training(decay_agent, num_episodes=50,
                                 callback=cb, callback_interval=25)
        assert len(calls) == 2
        assert calls[1] < calls[0]  # epsilon should have decayed
