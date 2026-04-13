import pytest

from src.environment import MountainCarEnvironment
from src.episode_runner import EpisodeRunner
from src.sarsa_agent import SarsaAgent
from src.tile_coder import TileCoder


@pytest.fixture
def runner():
    env = MountainCarEnvironment()
    agent = SarsaAgent(TileCoder(), n_actions=3)
    return EpisodeRunner(env, agent)


class TestEpisodeRunnerConstruction:
    def test_stores_env(self):
        env = MountainCarEnvironment()
        agent = SarsaAgent(TileCoder(), n_actions=3)
        runner = EpisodeRunner(env, agent)
        assert runner.env is env

    def test_stores_agent(self):
        env = MountainCarEnvironment()
        agent = SarsaAgent(TileCoder(), n_actions=3)
        runner = EpisodeRunner(env, agent)
        assert runner.agent is agent

    def test_recorder_defaults_to_none(self):
        env = MountainCarEnvironment()
        agent = SarsaAgent(TileCoder(), n_actions=3)
        runner = EpisodeRunner(env, agent)
        assert runner.recorder is None


class TestEpisodeRunnerRunEpisode:
    def test_returns_int(self, runner):
        assert isinstance(runner.run_episode(), int)

    def test_returns_positive_step_count(self, runner):
        assert runner.run_episode() >= 1

    def test_step_count_within_episode_limit(self, runner):
        assert runner.run_episode() <= 200
