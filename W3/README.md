# Lab 3: First-Visit Monte Carlo Control for Blackjack

## Overview

This lab implements first-visit Monte Carlo control with epsilon-soft policies on Gymnasium's Blackjack-v1 environment (Sutton and Barto mode). The agent learns to play Blackjack through experience, with configurable exploration strategies including linear, exponential, cosine, step, and adaptive epsilon decay. Results are evaluated via a bake-off comparing the trained agent against naive and rules-based baselines.

## Project Structure

```
W3/
├── src/
│   ├── blackjack_agent.py         # First-visit MC control agent
│   ├── episode_runner.py          # Episode generation and training loop
│   ├── basic_strategy.py          # Rules-based Blackjack strategy
│   ├── naive_agent.py             # Baseline agents (random, always hit, always stick)
│   ├── decay_schedules.py         # Epsilon decay strategies (linear, exponential, cosine, step, adaptive)
│   ├── visualizer.py              # Plotting utilities (3D surfaces, learning curves, bake-off charts)
│   ├── blackjack_simulation.py    # CLI orchestrator
│   ├── test_blackjack_agent.py    # Agent tests
│   ├── test_episode_runner.py     # Runner tests (including verbose and callback)
│   ├── test_basic_strategy.py     # Basic strategy tests
│   ├── test_naive_agent.py        # Naive agent tests
│   ├── test_decay_schedules.py    # Decay schedule tests
│   ├── test_visualizer.py         # Visualization tests
│   ├── test_blackjack_simulation.py # CLI orchestrator tests
│   └── test_integration_cycle1.py # Integration tests
├── notebooks/
│   ├── development.ipynb          # Development notebook (orchestration and experimentation)
│   └── report.ipynb               # Report notebook (summaries and narrative)
├── img/                           # Generated figures
├── environment.yml                # Conda environment (gym_env)
└── README.md
```

## Classes

| Class | Module | Purpose |
|---|---|---|
| `BlackjackAgent` | `blackjack_agent.py` | First-visit MC control with epsilon-soft policy, injectable decay and exploration metrics |
| `EpisodeRunner` | `episode_runner.py` | Runs episodes and training loops with verbose output and callback support |
| `BasicStrategy` | `basic_strategy.py` | Rules-based Blackjack strategy for benchmarking |
| `NaiveAgent` | `naive_agent.py` | Baseline agents: random, always hit, always stick |
| `LinearDecay` | `decay_schedules.py` | Fixed-amount epsilon decay per step |
| `ExponentialDecay` | `decay_schedules.py` | Multiplicative epsilon decay per step |
| `CosineDecay` | `decay_schedules.py` | Cosine annealing with internal step counter |
| `StepDecay` | `decay_schedules.py` | Drops epsilon by a factor at fixed intervals |
| `AdaptiveDecay` | `decay_schedules.py` | Adapts decay rate based on agent's exploration signal |
| `Visualizer` | `visualizer.py` | 3D value surfaces, learning curves with 95% CI, bake-off charts, outcome breakdowns |
| `BlackjackSimulation` | `blackjack_simulation.py` | CLI orchestrator for running the full simulation |

## Usage

### From the command line

```bash
cd src
python blackjack_simulation.py
python blackjack_simulation.py --num-episodes 500000 --decay cosine --output-dir ../img
python blackjack_simulation.py --epsilon 0.1 --decay none
```

| Argument | Default | Options |
|---|---|---|
| `--num-episodes` | 500,000 | any integer |
| `--epsilon` | 1.0 | any float 0-1 |
| `--decay` | linear | linear, cosine, exponential, none |
| `--output-dir` | ../img | any path |

### From a notebook

```python
import gymnasium as gym
from blackjack_agent import BlackjackAgent
from episode_runner import EpisodeRunner
from decay_schedules import CosineDecay
from visualizer import Visualizer

env = gym.make("Blackjack-v1", sab=True)
runner = EpisodeRunner(env)

agent = BlackjackAgent(epsilon=1.0,
                       decay_schedule=CosineDecay(eps_start=1.0, eps_min=0.01,
                                                   total_steps=500_000))
rewards = runner.run_training(agent, num_episodes=500_000)

viz = Visualizer()
fig = viz.plot_learning_curve(rewards, window_size=5000, show_ci=True)
```

## Experiments

Three hypotheses are tested in the development notebook:

- **H1 (Fixed epsilon):** An optimal fixed epsilon balances exploration and exploitation
- **H2 (Scheduled decay):** Decaying epsilon over time outperforms fixed epsilon; schedule shape affects convergence
- **H3 (Adaptive decay):** Decay that responds to Q-value estimation confidence outperforms fixed schedules

## Environment Setup

```bash
conda env create -f environment.yml
conda activate gym_env
```

## Running Tests

```bash
cd src
python -m pytest -v
```

## Current Status

| Component | Module | Tests |
|---|---|---|
| `BlackjackAgent` | `test_blackjack_agent.py` | 20/20 passing |
| `EpisodeRunner` | `test_episode_runner.py` | 15/15 passing |
| `BasicStrategy` | `test_basic_strategy.py` | 16/16 passing |
| `NaiveAgent` | `test_naive_agent.py` | 7/7 passing |
| `DecaySchedules` | `test_decay_schedules.py` | 18/18 passing |
| `Visualizer` | `test_visualizer.py` | 19/19 passing |
| `BlackjackSimulation` | `test_blackjack_simulation.py` | 14/14 passing |
| Integration | `test_integration_cycle1.py` | 4/4 passing |
| **Total** | | **113/113 passing** |
