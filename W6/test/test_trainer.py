import pytest
import torch
import gymnasium as gym
import pandas as pd
import os

LABEL = "test_run"
NUM_EPISODES = 2

STEP_COLUMNS = {"episode", "step", "reward", "td_error", "entropy", "x", "y", "angle"}
EPISODE_COLUMNS = {"episode_return", "episode_length"}
CONFIG_COLUMNS = {"label", "obs_dim", "act_dim", "actor_lr", "critic_lr", "gamma", "seed"}


@pytest.fixture
def env():
    e = gym.make("LunarLanderContinuous-v3")
    yield e
    e.close()


@pytest.fixture
def agent(env):
    from src.agent import Agent
    act_low  = torch.tensor(env.action_space.low)
    act_high = torch.tensor(env.action_space.high)
    return Agent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        act_low=act_low,
        act_high=act_high,
    )


@pytest.fixture
def trainer(agent, env):
    from src.trainer import Trainer
    return Trainer(agent, env, label=LABEL)


@pytest.fixture
def results(trainer):
    return trainer.train(num_episodes=NUM_EPISODES)


class TestTrainerReturnsDataFrame:
    def test_returns_dataframe(self, results):
        assert isinstance(results, pd.DataFrame)

    def test_has_rows(self, results):
        assert len(results) > 0


class TestTrainerColumns:
    def test_step_columns_present(self, results):
        assert STEP_COLUMNS.issubset(results.columns)

    def test_episode_columns_present(self, results):
        assert EPISODE_COLUMNS.issubset(results.columns)

    def test_config_columns_present(self, results):
        assert CONFIG_COLUMNS.issubset(results.columns)


class TestTrainerValues:
    def test_td_error_finite(self, results):
        assert results["td_error"].apply(lambda x: pd.notna(x) and abs(x) < 1e6).all()

    def test_entropy_positive(self, results):
        assert (results["entropy"] > 0).all()

    def test_episode_return_finite(self, results):
        assert results["episode_return"].apply(pd.notna).all()

    def test_correct_num_episodes(self, results):
        assert results["episode"].nunique() == NUM_EPISODES

    def test_label_column(self, results):
        assert (results["label"] == LABEL).all()

    def test_episode_length_positive(self, results):
        assert (results["episode_length"] > 0).all()


class TestTrainerSeed:
    def test_seed_column_present_when_set(self, agent, env):
        from src.trainer import Trainer
        trainer = Trainer(agent, env, label=LABEL, seed=42)
        df = trainer.train(num_episodes=1)
        assert "seed" in df.columns

    def test_seed_column_value(self, agent, env):
        from src.trainer import Trainer
        trainer = Trainer(agent, env, label=LABEL, seed=42)
        df = trainer.train(num_episodes=1)
        assert (df["seed"] == 42).all()

    def test_no_seed_column_when_none(self, agent, env):
        from src.trainer import Trainer
        trainer = Trainer(agent, env, label=LABEL, seed=None)
        df = trainer.train(num_episodes=1)
        assert "seed" not in df.columns

    def test_reproducibility(self, env):
        from src.agent import Agent
        from src.trainer import Trainer

        def make_agent():
            act_low  = torch.tensor(env.action_space.low)
            act_high = torch.tensor(env.action_space.high)
            torch.manual_seed(42)
            return Agent(
                obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.shape[0],
                actor_lr=3e-4, critic_lr=1e-3, gamma=0.99,
                act_low=act_low, act_high=act_high,
            )

        df1 = Trainer(make_agent(), env, label=LABEL, seed=42).train(num_episodes=1)
        df2 = Trainer(make_agent(), env, label=LABEL, seed=42).train(num_episodes=1)
        assert df1["episode_return"].iloc[0] == df2["episode_return"].iloc[0]


class TestTrainerCSV:
    def test_csv_written(self, trainer, tmp_path):
        trainer.data_dir = str(tmp_path)
        trainer.train(num_episodes=1)
        csv_files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]
        assert len(csv_files) == 1

    def test_csv_filename_contains_label(self, trainer, tmp_path):
        trainer.data_dir = str(tmp_path)
        trainer.train(num_episodes=1)
        csv_files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]
        assert LABEL in csv_files[0]
