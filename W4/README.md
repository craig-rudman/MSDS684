# MSDS684 Lab 4 — SARSA vs Q-Learning on CliffWalking

Implements and compares two temporal-difference (TD) control algorithms — SARSA (on-policy) and Q-learning (off-policy) — on Gymnasium's `CliffWalking-v1` environment. The project follows strict TDD throughout and is structured around a 10-phase development plan documented in `ipynb/development.ipynb`.

## Background

CliffWalking is the canonical environment from Sutton & Barto Chapter 6 for contrasting on-policy and off-policy TD methods. The agent navigates a 4×12 grid from start (bottom-left) to goal (bottom-right), avoiding a cliff that incurs a -100 penalty. SARSA learns a safe inland path; Q-learning learns the shorter cliff-edge path at the cost of more falls during training.

## Project Structure

```
W4/
├── src/
│   ├── agents.py         # TDAgent (base), SARSAAgent, QLearningAgent
│   ├── environment.py    # EnvironmentManager (wraps Gymnasium)
│   ├── experiment.py     # ExperimentConfig, ExperimentResult, ExperimentSuite
│   ├── runner.py         # EpisodeRunner
│   ├── schedules.py      # ConstantSchedule, LinearDecaySchedule, ExponentialDecaySchedule
│   ├── visualization.py  # Visualizer (learning curves, policy arrows, heatmaps, trajectories)
│   └── main.py           # CLI entry point
├── tests/                # 55 unit tests (pytest)
├── ipynb/
│   └── development.ipynb # Phase-by-phase development notebook
├── results/              # Generated plots and CSVs
└── environment.yml       # Conda environment
```

## Setup

```bash
conda env create -f environment.yml
conda activate gym_env
```

## Running Experiments

Run the full experiment suite from the `src/` directory:

```bash
cd src
python main.py
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n-seeds` | 30 | Random seeds per configuration |
| `--n-episodes` | 500 | Training episodes per seed |
| `--output-dir` | `./results` | Directory for plots and CSVs |

**Example — quick smoke test:**

```bash
python main.py --n-seeds 2 --n-episodes 10 --output-dir /tmp/test_run
```

**Example — full production run:**

```bash
python main.py --n-seeds 30 --n-episodes 500 --output-dir ../results
```

## Outputs

All outputs are written to `--output-dir`:

| File | Description |
|------|-------------|
| `baseline_learning_curves.png` | SARSA vs Q-learning training curves with 95% CI |
| `policy_arrows_<label>.png` | Greedy action at every state shown as arrows |
| `value_heatmap_<label>.png` | Max Q-value per state (RdYlGn colormap) |
| `trajectory_<label>.png` | Greedy rollout path from start to goal |
| `trajectory_comparison.png` | Both agents' paths overlaid on the same grid |
| `SARSA_alpha_sweep.png` | Effect of α on SARSA learning curves |
| `Q-Learning_alpha_sweep.png` | Effect of α on Q-learning learning curves |
| `SARSA_schedule_sweep.png` | Effect of ε schedule on SARSA |
| `Q-Learning_schedule_sweep.png` | Effect of ε schedule on Q-learning |
| `alpha_sweep_summary.csv` | Metrics + performance index for all α configs |
| `schedule_sweep_summary.csv` | Metrics + performance index for all schedule configs |
| `combined_summary.csv` | Globally-normalized performance index across all configs |

## Running Tests

From the project root:

```bash
/path/to/gym_env/bin/python -m pytest tests/ -v
```

Or with the active conda environment:

```bash
python -m pytest tests/ -v
```

55 tests across agents, environment, experiment, runner, schedules, and visualization.

## Key Results

- **Training performance**: SARSA significantly outperforms Q-learning (~-20 vs ~-47 mean episode return) because on-policy updates penalize exploratory cliff falls.
- **Greedy inference**: Q-learning's trained policy completes the task in 13 steps (cliff-edge path); SARSA's in 15 steps (safe inland path). The 2-step gap is the measurable cost of SARSA's safety margin.
- **Best configuration**: SARSA with α=0.1, constant ε=0.1 achieves the best balance of final performance, learning speed, and sample efficiency on a 500-episode budget.

## References

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Example 6.6 (pp. 132–134).
