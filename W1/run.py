#!/usr/bin/env python3
"""Command-line harness for running Part 1 (bandits) and Part 2 (Gym envs)."""

import argparse
import sys


def run_part1(args):
    """Run the multi-armed bandit comparison."""
    from part_01.src.factories import make_bandit_env, make_agent
    from part_01.src.runner import run_simulation
    from part_01.src.plotting import plot_comparison

    env = make_bandit_env(k=args.k, randomize=True, max_steps=args.steps)

    agents = [
        make_agent("epsilon_greedy", k=args.k, epsilon=0.01),
        make_agent("epsilon_greedy", k=args.k, epsilon=0.1),
        make_agent("epsilon_greedy", k=args.k, epsilon=0.2),
        make_agent("ucb", k=args.k, c=1.0),
        make_agent("ucb", k=args.k, c=2.0),
    ]

    results = []
    for agent in agents:
        print(f"Running {agent.agent_name}...")
        result = run_simulation(env, agent, n_runs=args.runs, agent_name=agent.agent_name)
        s = result.summary()
        print(f"  Reward: {s['final_mean_reward']:.3f} | "
              f"Optimal: {s['final_optimal_pct']:.1f}% | "
              f"Regret: {s['final_cumulative_regret']:.1f}")
        results.append(result)

    save_path = args.save or "part_01/img/bandit_comparison.png"
    plot_comparison(results, panels=["reward", "optimal_pct"], smoothing=20, save_path=save_path)
    print(f"\nPlot saved to {save_path}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


def run_part2(args):
    """Run the Gymnasium environment exploration."""
    import gymnasium as gym
    from part_02.src.env_inspector import inspect_env, format_inspection, get_transition_table
    from part_02.src.agents import RandomAgent
    from part_02.src.runner import run_episodes
    from part_02.src.plotting import plot_episode_metrics

    envs = [
        {"id": "FrozenLake-v1", "kwargs": {"is_slippery": True}, "success_reward": 0.0,
         "save": "part_02/img/frozenlake_random.png"},
        {"id": "Taxi-v3", "kwargs": {}, "success_reward": 19.0,
         "save": "part_02/img/taxi_random.png"},
    ]

    for env_cfg in envs:
        env = gym.make(env_cfg["id"], **env_cfg["kwargs"])
        info = inspect_env(env)
        print(format_inspection(info))
        print()

        agent = RandomAgent(env.action_space)
        result = run_episodes(env, agent, n_episodes=args.episodes, success_reward=env_cfg["success_reward"])

        s = result.summary()
        print(f"{env_cfg['id']} — Random Agent ({s['n_episodes']} episodes)")
        print(f"  Mean return:  {s['mean_return']:.4f} ± {s['std_return']:.4f}")
        print(f"  Mean length:  {s['mean_length']:.1f} steps")
        print(f"  Success rate: {s['success_rate']:.1f}%")

        save_path = env_cfg["save"]
        plot_episode_metrics(result, save_path=save_path)
        print(f"  Plot saved to {save_path}\n")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="MSDS684 Week 1 Lab Runner")
    sub = parser.add_subparsers(dest="part", required=True)

    p1 = sub.add_parser("part1", help="Multi-armed bandit comparison")
    p1.add_argument("--k", type=int, default=10, help="Number of arms (default: 10)")
    p1.add_argument("--steps", type=int, default=2000, help="Steps per run (default: 2000)")
    p1.add_argument("--runs", type=int, default=1000, help="Independent runs (default: 1000)")
    p1.add_argument("--save", type=str, default=None, help="Save path for plot")
    p1.add_argument("--show", action="store_true", help="Open plot in an interactive window")

    p2 = sub.add_parser("part2", help="Gymnasium environment exploration")
    p2.add_argument("--episodes", type=int, default=1000, help="Episodes per env (default: 1000)")
    p2.add_argument("--show", action="store_true", help="Open plot in an interactive window")

    args = parser.parse_args()

    if args.part == "part1":
        run_part1(args)
    elif args.part == "part2":
        run_part2(args)


if __name__ == "__main__":
    main()
