import numpy as np
import pytest


class TestEpisodeRunner:
    def test_run_episode_returns_scalar(self, sarsa_agent, env_manager):
        from runner import EpisodeRunner
        runner = EpisodeRunner(sarsa_agent, env_manager)
        result = runner.run_episode()
        assert isinstance(result, (int, float))

    def test_run_episode_reward_is_negative(self, sarsa_agent, env_manager):
        from runner import EpisodeRunner
        runner = EpisodeRunner(sarsa_agent, env_manager)
        result = runner.run_episode()
        assert result <= -1.0

    def test_run_experiment_shape(self, sarsa_agent, env_manager):
        from runner import EpisodeRunner
        runner = EpisodeRunner(sarsa_agent, env_manager)
        matrix = runner.run_experiment(n_seeds=3, n_episodes=5)
        assert matrix.shape == (3, 5)

    def test_run_experiment_all_negative(self, sarsa_agent, env_manager):
        from runner import EpisodeRunner
        runner = EpisodeRunner(sarsa_agent, env_manager)
        matrix = runner.run_experiment(n_seeds=3, n_episodes=5)
        assert np.all(matrix <= -1.0)

    def test_run_experiment_returns_ndarray(self, sarsa_agent, env_manager):
        from runner import EpisodeRunner
        runner = EpisodeRunner(sarsa_agent, env_manager)
        matrix = runner.run_experiment(n_seeds=2, n_episodes=3)
        assert isinstance(matrix, np.ndarray)
