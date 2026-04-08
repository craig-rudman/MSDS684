import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np

from agents import SARSAAgent, QLearningAgent
from environment import EnvironmentManager
from experiment import ExperimentConfig, ExperimentSuite
from schedules import ConstantSchedule, LinearDecaySchedule, ExponentialDecaySchedule
from visualization import Visualizer


PERFORMANCE_WEIGHTS = {
    'final_performance': 0.5,
    'learning_speed':    0.1,
    'sample_efficiency': 0.4,
}

ALPHA_SWEEP    = [0.05, 0.3, 0.5]   # baseline (0.1) included via baseline_suite
BASELINE_ALPHA = 0.1
BASELINE_EPS   = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CliffWalking SARSA vs Q-Learning experiments')
    parser.add_argument('--n-seeds',    type=int, default=30,          help='Number of random seeds per config (default: 30)')
    parser.add_argument('--n-episodes', type=int, default=500,         help='Episodes per seed (default: 500)')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory for plots and CSVs (default: ./results)')
    return parser.parse_args()


def build_baseline_configs(n_seeds: int, n_episodes: int) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            label='SARSA α=0.1 ε=0.1',
            agent_class=SARSAAgent,
            alpha=BASELINE_ALPHA,
            epsilon_schedule=ConstantSchedule(BASELINE_EPS),
            n_seeds=n_seeds,
            n_episodes=n_episodes,
        ),
        ExperimentConfig(
            label='Q-Learning α=0.1 ε=0.1',
            agent_class=QLearningAgent,
            alpha=BASELINE_ALPHA,
            epsilon_schedule=ConstantSchedule(BASELINE_EPS),
            n_seeds=n_seeds,
            n_episodes=n_episodes,
        ),
    ]


def build_alpha_configs(n_seeds: int, n_episodes: int) -> list[ExperimentConfig]:
    configs = []
    for alpha in ALPHA_SWEEP:
        configs.append(ExperimentConfig(
            label=f'SARSA α={alpha}',
            agent_class=SARSAAgent,
            alpha=alpha,
            epsilon_schedule=ConstantSchedule(BASELINE_EPS),
            n_seeds=n_seeds,
            n_episodes=n_episodes,
        ))
        configs.append(ExperimentConfig(
            label=f'Q-Learning α={alpha}',
            agent_class=QLearningAgent,
            alpha=alpha,
            epsilon_schedule=ConstantSchedule(BASELINE_EPS),
            n_seeds=n_seeds,
            n_episodes=n_episodes,
        ))
    return configs


def build_schedule_configs(n_seeds: int, n_episodes: int) -> list[ExperimentConfig]:
    schedules = {
        'ε=LinearDecay':      LinearDecaySchedule(epsilon_start=BASELINE_EPS, epsilon_end=0.001, n_episodes=n_episodes),
        'ε=ExponentialDecay': ExponentialDecaySchedule(epsilon_start=BASELINE_EPS, epsilon_end=0.001, n_episodes=n_episodes),
    }
    configs = []
    for label, schedule in schedules.items():
        configs.append(ExperimentConfig(
            label=f'SARSA {label}',
            agent_class=SARSAAgent,
            alpha=BASELINE_ALPHA,
            epsilon_schedule=schedule,
            n_seeds=n_seeds,
            n_episodes=n_episodes,
        ))
        configs.append(ExperimentConfig(
            label=f'Q-Learning {label}',
            agent_class=QLearningAgent,
            alpha=BASELINE_ALPHA,
            epsilon_schedule=schedule,
            n_seeds=n_seeds,
            n_episodes=n_episodes,
        ))
    return configs


def greedy_return(q_table: np.ndarray, env: EnvironmentManager) -> float:
    state = env.reset(seed=0)
    total = 0.0
    while True:
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output directory: {os.path.abspath(args.output_dir)}')
    print(f'n_seeds={args.n_seeds}  n_episodes={args.n_episodes}')

    env = EnvironmentManager()
    vis = Visualizer(env)

    # --- Baseline ---
    print('\nRunning baseline...')
    baseline_suite = ExperimentSuite(build_baseline_configs(args.n_seeds, args.n_episodes), env)
    baseline_suite.run()

    vis.plot_learning_curves(
        baseline_suite.results, ylim=(-200, 0), smooth_window=10,
        output_dir=args.output_dir, filename='baseline_learning_curves',
        title='Baseline: SARSA vs Q-Learning',
    )
    for r in baseline_suite.results:
        vis.plot_policy_arrows(r.q_table, label=r.config.label, output_dir=args.output_dir)
        vis.plot_value_heatmap(r.q_table, label=r.config.label, output_dir=args.output_dir)
        vis.plot_trajectory(r.q_table,    label=r.config.label, output_dir=args.output_dir)
    vis.plot_trajectory_comparison(baseline_suite.results, output_dir=args.output_dir)

    # --- Greedy inference ---
    print('Running greedy inference...')
    for r in baseline_suite.results:
        ret = greedy_return(r.q_table, env)
        print(f'  {r.config.label}: greedy return={ret:.0f}')

    # --- Alpha sweep ---
    print('\nRunning alpha sweep...')
    alpha_suite = ExperimentSuite(build_alpha_configs(args.n_seeds, args.n_episodes), env)
    alpha_suite.run()

    alpha_all = ExperimentSuite([], env)
    alpha_all.results = baseline_suite.results + alpha_suite.results
    alpha_all.summarize(
        output_path=os.path.join(args.output_dir, 'alpha_sweep_summary.csv'),
        weights=PERFORMANCE_WEIGHTS,
        sort_by='performance_index',
        ascending=False,
    )
    for agent_label in ['SARSA', 'Q-Learning']:
        subset = [r for r in alpha_all.results if r.config.label.startswith(agent_label)]
        vis.plot_learning_curves(
            subset, ylim=(-200, 0), smooth_window=10,
            output_dir=args.output_dir, filename=f'{agent_label}_alpha_sweep',
            title=f'{agent_label}: Effect of Learning Rate (α)',
        )

    # --- Schedule sweep ---
    print('Running schedule sweep...')
    schedule_suite = ExperimentSuite(build_schedule_configs(args.n_seeds, args.n_episodes), env)
    schedule_suite.run()

    schedule_all = ExperimentSuite([], env)
    schedule_all.results = baseline_suite.results + schedule_suite.results
    schedule_all.summarize(
        output_path=os.path.join(args.output_dir, 'schedule_sweep_summary.csv'),
        weights=PERFORMANCE_WEIGHTS,
        sort_by='performance_index',
        ascending=False,
    )
    for agent_label in ['SARSA', 'Q-Learning']:
        subset = [r for r in schedule_all.results if r.config.label.startswith(agent_label)]
        vis.plot_learning_curves(
            subset, ylim=(-200, 0), smooth_window=10,
            output_dir=args.output_dir, filename=f'{agent_label}_schedule_sweep',
            title=f'{agent_label}: Effect of Epsilon Schedule',
            show_learning_speed=True,
        )

    # --- Combined summary ---
    print('Generating combined summary...')
    combined = ExperimentSuite([], env)
    combined.results = baseline_suite.results + alpha_suite.results + schedule_suite.results
    combined.summarize(
        output_path=os.path.join(args.output_dir, 'combined_summary.csv'),
        weights=PERFORMANCE_WEIGHTS,
        sort_by='performance_index',
        ascending=False,
    )

    print(f'\nDone. Results written to {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
