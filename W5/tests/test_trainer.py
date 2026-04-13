import pytest

from src.environment import MountainCarEnvironment
from src.sarsa_agent import SarsaAgent
from src.tile_coder import TileCoder
from src.trainer import Trainer


@pytest.fixture
def trainer():
    env = MountainCarEnvironment()
    agent = SarsaAgent(TileCoder(), n_actions=3)
    return Trainer(env, agent, n_episodes=3)


class TestTrainerConstruction:
    def test_stores_env(self):
        env = MountainCarEnvironment()
        agent = SarsaAgent(TileCoder(), n_actions=3)
        t = Trainer(env, agent, n_episodes=3)
        assert t.env is env

    def test_stores_agent(self):
        env = MountainCarEnvironment()
        agent = SarsaAgent(TileCoder(), n_actions=3)
        t = Trainer(env, agent, n_episodes=3)
        assert t.agent is agent

    def test_stores_n_episodes(self):
        env = MountainCarEnvironment()
        agent = SarsaAgent(TileCoder(), n_actions=3)
        t = Trainer(env, agent, n_episodes=5)
        assert t.n_episodes == 5

    def test_recorder_defaults_to_none(self):
        env = MountainCarEnvironment()
        agent = SarsaAgent(TileCoder(), n_actions=3)
        t = Trainer(env, agent, n_episodes=3)
        assert t.recorder is None


class TestTrainerTrain:
    def test_returns_list(self, trainer):
        assert isinstance(trainer.train(), list)

    def test_returns_one_entry_per_episode(self, trainer):
        assert len(trainer.train()) == 3

    def test_each_entry_is_positive_int(self, trainer):
        for steps in trainer.train():
            assert isinstance(steps, int)
            assert steps >= 1
