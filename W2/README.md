# Lab 2: Dynamic Programming on GridWorld

## Overview

This lab implements dynamic programming algorithms (policy iteration and value iteration) on a custom GridWorld environment built with the Gymnasium API. It supports configurable rewards, obstacles, terminal states, and stochastic transitions. Results can be visualized as heatmaps, quiver plots, and convergence curves.

## Project Structure

```
W2/
├── src/
│   ├── __init__.py            # Package exports
│   ├── gridworld.py           # Custom GridWorld environment (Gymnasium API)
│   ├── dp_solvers.py          # PolicyIteration and ValueIteration classes
│   ├── visualization.py       # Heatmaps, quiver plots, convergence curves
│   └── runner.py              # ExperimentRunner and CLI entry point
├── tests/
│   ├── conftest.py            # Shared pytest fixtures
│   ├── test_gridworld.py      # Environment tests
│   ├── test_dp_solvers.py     # Solver correctness tests
│   ├── test_visualization.py  # Visualization smoke tests
│   └── test_runner.py         # Runner and CLI tests
├── notebook/
│   ├── Lab 2 development.ipynb    # Development notebook
│   └── Rudman_Craig_Lab2.ipynb    # Final submission notebook
├── img/                       # Generated figures
├── environment.yml            # Conda environment (gym_env)
└── README.md
```

## Classes

| Class | Module | Purpose |
|---|---|---|
| `GridWorld` | `src/gridworld.py` | Custom Gymnasium environment with configurable geometry and dynamics |
| `PolicyIteration` | `src/dp_solvers.py` | Synchronous and in-place policy iteration |
| `ValueIteration` | `src/dp_solvers.py` | Synchronous and in-place value iteration |
| `GridWorldVisualizer` | `src/visualization.py` | Plotting utilities for value functions, policies, and convergence |
| `ExperimentRunner` | `src/runner.py` | Orchestrates experiments and provides CLI access |

## Usage

### From the command line

```bash
python -m src.runner
```

### From a notebook

```python
from src import GridWorld, PolicyIteration, ValueIteration, GridWorldVisualizer

env = GridWorld(size=4, stochastic=True, intended_prob=0.8)
vi = ValueIteration(env, gamma=0.99)
V, policy = vi.solve(mode="sync")

viz = GridWorldVisualizer(env)
viz.plot_value_function(V)
viz.plot_policy(policy)
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate gym_env
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Current Status

| Component | Status | Tests |
|---|---|---|
| `GridWorld` | Implemented | 20/20 passing |
| `PolicyIteration` | Stub | Pending |
| `ValueIteration` | Stub | Pending |
| `GridWorldVisualizer` | Stub | Pending |
| `ExperimentRunner` | Stub | Pending |

Implementation is proceeding one TODO at a time following a red-green-refactor workflow.
