import os
import numpy as np
import pytest
import pandas as pd
from experiment import ExperimentConfig, ExperimentResult, ExperimentSuite
from agents import SARSAAgent, QLearningAgent
from schedules import ConstantSchedule


def make_config(label='test_sarsa'):
    return ExperimentConfig(
        label=label,
        agent_class=SARSAAgent,
        alpha=0.1,
        epsilon_schedule=ConstantSchedule(0.1),
        n_seeds=2,
        n_episodes=5,
    )


def make_result(label='test_sarsa'):
    config = make_config(label)
    reward_matrix = np.full((2, 5), -10.0)
    return ExperimentResult(config, reward_matrix)


class TestExperimentConfig:
    def test_fields_accessible(self):
        config = make_config()
        assert config.label == 'test_sarsa'
        assert config.agent_class is SARSAAgent
        assert config.alpha == 0.1
        assert config.n_seeds == 2
        assert config.n_episodes == 5


class TestExperimentResult:
    def test_final_performance_is_scalar(self):
        result = make_result()
        assert isinstance(result.final_performance(), float)

    def test_final_performance_is_negative(self):
        result = make_result()
        assert result.final_performance() <= -1.0

    def test_learning_speed_is_scalar(self):
        result = make_result()
        assert isinstance(result.learning_speed(), (int, float))

    def test_sample_efficiency_is_scalar(self):
        result = make_result()
        assert isinstance(result.sample_efficiency(), float)

    def test_save_writes_file(self, tmp_path):
        result = make_result(label='sarsa_test')
        result.save(str(tmp_path))
        assert os.path.exists(os.path.join(tmp_path, 'sarsa_test.npy'))


class TestExperimentSuite:
    def test_summarize_returns_dataframe(self, env_manager):
        configs = [make_config('sarsa'), make_config('qlearn')]
        configs[1] = ExperimentConfig(
            label='qlearn',
            agent_class=QLearningAgent,
            alpha=0.1,
            epsilon_schedule=ConstantSchedule(0.1),
            n_seeds=2,
            n_episodes=5,
        )
        suite = ExperimentSuite(configs, env_manager)
        suite.run()
        df = suite.summarize()
        assert isinstance(df, pd.DataFrame)

    def test_summarize_one_row_per_config(self, env_manager):
        configs = [make_config('a'), make_config('b')]
        suite = ExperimentSuite(configs, env_manager)
        suite.run()
        df = suite.summarize()
        assert len(df) == 2

    def test_summarize_expected_columns(self, env_manager):
        configs = [make_config()]
        suite = ExperimentSuite(configs, env_manager)
        suite.run()
        df = suite.summarize()
        for col in ['label', 'final_performance', 'learning_speed', 'sample_efficiency']:
            assert col in df.columns

    def test_summarize_writes_csv(self, env_manager, tmp_path):
        configs = [make_config()]
        suite = ExperimentSuite(configs, env_manager)
        suite.run()
        out = str(tmp_path / 'summary.csv')
        suite.summarize(output_path=out)
        assert os.path.exists(out)

    def test_summarize_uses_consistent_learning_speed_threshold(self):
        # Result A improves from -100 to -10 (crosses -75 partway through).
        # Result B only reaches -80 (never crosses -75).
        # summarize() must use the same threshold for both rows.
        n_episodes = 100
        matrix_a = np.linspace(-100, -10, n_episodes).reshape(1, n_episodes)
        matrix_b = np.linspace(-100, -80, n_episodes).reshape(1, n_episodes)
        result_a = ExperimentResult(make_config('A'), matrix_a)
        result_b = ExperimentResult(make_config('B'), matrix_b)

        # A crosses -75; B never does
        assert result_a.learning_speed(threshold=-75.0) < n_episodes
        assert result_b.learning_speed(threshold=-75.0) == n_episodes

        # summarize() must report the same split
        suite = ExperimentSuite([], None)
        suite.results = [result_a, result_b]
        df = suite.summarize(learning_speed_threshold=-75.0)
        assert df[df['label'] == 'A']['learning_speed'].iloc[0] < n_episodes
        assert df[df['label'] == 'B']['learning_speed'].iloc[0] == n_episodes
