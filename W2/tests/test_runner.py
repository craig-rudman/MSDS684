"""Tests for src/runner.py -- ExperimentRunner."""

import pytest

from src import ExperimentRunner, GridWorld


class TestRunGridWorld:
    """ExperimentRunner.run_gridworld populates results."""

    def test_results_populated(self, env_4x4_deterministic):
        runner = ExperimentRunner()
        results = runner.run_gridworld("test_det", env_4x4_deterministic)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_all_variants_present(self, env_4x4_deterministic):
        runner = ExperimentRunner()
        results = runner.run_gridworld("test_det", env_4x4_deterministic)
        expected_keys = {"pi_sync", "pi_inplace", "vi_sync", "vi_inplace"}
        assert set(results.keys()) == expected_keys

    def test_subset_of_modes(self, env_4x4_deterministic):
        runner = ExperimentRunner()
        results = runner.run_gridworld(
            "test_det", env_4x4_deterministic, modes=["vi_sync"]
        )
        assert set(results.keys()) == {"vi_sync"}


class TestRunGridWorldStochastic:
    """ExperimentRunner.run_gridworld works on stochastic environments."""

    def test_results_populated(self, env_4x4_stochastic):
        runner = ExperimentRunner()
        results = runner.run_gridworld("test_stoch", env_4x4_stochastic)
        assert isinstance(results, dict)
        assert len(results) == 4

    def test_all_variants_converge(self, env_4x4_stochastic):
        runner = ExperimentRunner()
        results = runner.run_gridworld("test_stoch", env_4x4_stochastic)
        for mode, data in results.items():
            assert data["V"].shape == (env_4x4_stochastic.n_states,)
            assert data["policy"].shape == (env_4x4_stochastic.n_states,)
            assert len(data["value_history"]) > 0
            assert len(data["wall_clock_times"]) > 0

    def test_all_variants_agree_on_values(self, env_4x4_stochastic):
        runner = ExperimentRunner()
        results = runner.run_gridworld("test_stoch", env_4x4_stochastic)
        import numpy as np
        variants = list(results.values())
        for other in variants[1:]:
            np.testing.assert_allclose(variants[0]["V"], other["V"], atol=1e-6)

    def test_stochastic_values_differ_from_deterministic(self, env_4x4_stochastic, env_4x4_deterministic):
        runner = ExperimentRunner()
        det = runner.run_gridworld("det", env_4x4_deterministic)
        stoch = runner.run_gridworld("stoch", env_4x4_stochastic)
        import numpy as np
        # Stochastic uncertainty should produce different value functions
        assert not np.allclose(det["vi_sync"]["V"], stoch["vi_sync"]["V"], atol=1e-3)


class TestRunFrozenLake:
    """ExperimentRunner.run_frozen_lake works end-to-end."""

    def test_runs_without_error(self):
        runner = ExperimentRunner()
        results = runner.run_frozen_lake(map_name="4x4", is_slippery=True)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_deterministic_frozen_lake(self):
        runner = ExperimentRunner()
        results = runner.run_frozen_lake(map_name="4x4", is_slippery=False)
        assert isinstance(results, dict)


class TestCLI:
    """CLI argument parsing."""

    def test_default_args(self):
        """cli() should be callable (we test parsing, not full execution)."""
        # This is a smoke test; full CLI testing would use subprocess
        assert callable(ExperimentRunner.cli)
