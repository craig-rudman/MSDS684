import argparse
import os
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")

from blackjack_agent import BlackjackAgent
from naive_agent import NaiveAgent
from basic_strategy import BasicStrategy
from episode_runner import EpisodeRunner
from visualizer import Visualizer
from decay_schedules import LinearDecay, CosineDecay


class BlackjackSimulation:
    """CLI orchestrator for Blackjack MC control experiments."""

    def __init__(self, num_episodes: int = 500_000, epsilon: float = 1.0,
                 decay_schedule=None, output_dir: str = "../img"):
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.decay_schedule = decay_schedule or LinearDecay(
            eps_min=0.01, decay_rate=1 / num_episodes)
        self.output_dir = output_dir
        self.env = gym.make("Blackjack-v1", sab=True)
        self.runner = EpisodeRunner(self.env)
        self.viz = Visualizer()

    def run(self):
        """Train MC agent, run baselines, generate and save all visualizations."""
        # Train MC agent
        mc_agent = BlackjackAgent(epsilon=self.epsilon,
                                   decay_schedule=self.decay_schedule)
        mc_rewards = self.runner.run_training(mc_agent,
                                              num_episodes=self.num_episodes)

        # Run baselines
        naive_agent = NaiveAgent(strategy="random")
        naive_rewards = self.runner.run_training(naive_agent,
                                                  num_episodes=self.num_episodes)

        basic_agent = BasicStrategy()
        basic_rewards = self.runner.run_training(basic_agent,
                                                  num_episodes=self.num_episodes)

        # Bake-off learning curves
        reward_series = {
            "NaiveRandom": naive_rewards,
            "BasicStrategy": basic_rewards,
            "MC Agent": mc_rewards,
        }
        window = min(5000, self.num_episodes // 10)

        fig = self.viz.plot_bakeoff_curves(reward_series, window_size=window)
        self.viz.save_plot(fig, os.path.join(self.output_dir,
                                              "bakeoff_learning_curves.png"))

        # Outcome breakdown
        fig = self.viz.plot_outcome_breakdown(reward_series)
        self.viz.save_plot(fig, os.path.join(self.output_dir,
                                              "outcome_breakdown.png"))

        # MC agent learning curve
        fig = self.viz.plot_learning_curve(mc_rewards, window_size=window,
                                           show_ci=True)
        self.viz.save_plot(fig, os.path.join(self.output_dir,
                                              "mc_learning_curve.png"))

        # Value surface plots
        value_fn = mc_agent.get_value_function()
        fig = self.viz.plot_value_surface(value_fn, "Value Function (Usable Ace)",
                                          usable_ace=True)
        self.viz.save_plot(fig, os.path.join(self.output_dir,
                                              "value_surface_usable_ace.png"))

        fig = self.viz.plot_value_surface(value_fn, "Value Function (No Usable Ace)",
                                          usable_ace=False)
        self.viz.save_plot(fig, os.path.join(self.output_dir,
                                              "value_surface_no_usable_ace.png"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Blackjack MC control simulation")
    parser.add_argument("--num-episodes", type=int, default=500_000,
                        help="Number of training episodes (default: 500000)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Starting epsilon (default: 1.0)")
    parser.add_argument("--decay", type=str, default="linear",
                        choices=["linear", "exponential", "cosine", "none"],
                        help="Decay schedule (default: linear)")
    parser.add_argument("--output-dir", type=str, default="../img",
                        help="Output directory for plots (default: ../img)")
    return parser.parse_args(argv)


def build_decay_schedule(name, epsilon, num_episodes):
    if name == "linear":
        return LinearDecay(eps_min=0.01, decay_rate=1 / num_episodes)
    elif name == "cosine":
        return CosineDecay(eps_start=epsilon, eps_min=0.01,
                           total_steps=num_episodes)
    elif name == "none":
        return None
    else:
        return LinearDecay(eps_min=0.01, decay_rate=1 / num_episodes)


def main():
    args = parse_args()
    decay = build_decay_schedule(args.decay, args.epsilon, args.num_episodes)
    sim = BlackjackSimulation(
        num_episodes=args.num_episodes,
        epsilon=args.epsilon,
        decay_schedule=decay,
        output_dir=args.output_dir,
    )
    sim.run()


if __name__ == "__main__":
    main()
