import pytest
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
from blackjack_simulation import BlackjackSimulation, parse_args


class TestBlackjackSimulationRun:
    """Tests for the simulation orchestrator."""

    def test_run_completes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100, output_dir=tmpdir)
            sim.run()

    def test_run_creates_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100, output_dir=tmpdir)
            sim.run()
            files = os.listdir(tmpdir)
            assert len(files) > 0

    def test_run_creates_bakeoff_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100, output_dir=tmpdir)
            sim.run()
            assert os.path.exists(os.path.join(tmpdir, "bakeoff_learning_curves.png"))

    def test_run_creates_value_surface_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100, output_dir=tmpdir)
            sim.run()
            assert os.path.exists(os.path.join(tmpdir, "value_surface_usable_ace.png"))
            assert os.path.exists(os.path.join(tmpdir, "value_surface_no_usable_ace.png"))

    def test_run_creates_outcome_breakdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100, output_dir=tmpdir)
            sim.run()
            assert os.path.exists(os.path.join(tmpdir, "outcome_breakdown.png"))

    def test_run_creates_learning_curve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100, output_dir=tmpdir)
            sim.run()
            assert os.path.exists(os.path.join(tmpdir, "mc_learning_curve.png"))

    def test_custom_epsilon(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100, epsilon=0.5,
                                      output_dir=tmpdir)
            sim.run()

    def test_custom_decay_schedule(self):
        from decay_schedules import LinearDecay
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = BlackjackSimulation(num_episodes=100,
                                      decay_schedule=LinearDecay(eps_min=0.01,
                                                                  decay_rate=0.01),
                                      output_dir=tmpdir)
            sim.run()


class TestBlackjackSimulationParseArgs:
    """Tests for CLI argument parsing."""

    def test_defaults(self):
        args = parse_args([])
        assert args.num_episodes == 500_000
        assert args.epsilon == 1.0
        assert args.output_dir == "../img"

    def test_custom_episodes(self):
        args = parse_args(["--num-episodes", "1000"])
        assert args.num_episodes == 1000

    def test_custom_epsilon(self):
        args = parse_args(["--epsilon", "0.5"])
        assert args.epsilon == 0.5

    def test_custom_output_dir(self):
        args = parse_args(["--output-dir", "/tmp/plots"])
        assert args.output_dir == "/tmp/plots"

    def test_decay_schedule_linear(self):
        args = parse_args(["--decay", "linear"])
        assert args.decay == "linear"

    def test_decay_schedule_cosine(self):
        args = parse_args(["--decay", "cosine"])
        assert args.decay == "cosine"
