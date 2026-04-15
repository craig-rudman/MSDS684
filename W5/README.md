# MSDS684 Lab 5 — Function Approximation with Value Methods

Implements semi-gradient SARSA with tile coding on Gymnasium's `MountainCar-v0` environment. The project follows strict TDD throughout and is structured around a phase-by-phase development plan documented in `ipynb/development.ipynb`.

## Background

MountainCar-v0 is a continuous state space environment where a car must build enough momentum to summit a hill. The state is described by two continuous dimensions (position, velocity), making tabular methods intractable. This lab implements tile coding to discretize the continuous state space into overlapping grids, enabling linear function approximation of Q-values. Semi-gradient SARSA (Sutton & Barto, Chapters 9–10) is used for on-policy control with epsilon-greedy exploration.

Five experiments vary tiling count and tile resolution to study their effects on learning speed, final performance, and training cost.

## Project Structure

```
W5/
├── src/
│   ├── environment.py      # MountainCarEnvironment (wraps Gymnasium)
│   ├── tile_coder.py       # TileCoder (configurable tilings, tiles per dim, offsets)
│   ├── sarsa_agent.py      # SarsaAgent (linear Q-value estimation, semi-gradient updates)
│   ├── episode_runner.py   # EpisodeRunner (single episode execution)
│   ├── episode_recorder.py # EpisodeRecorder (trajectory CSV logging)
│   ├── trainer.py          # Trainer (full training loop)
│   └── visualizer.py       # Visualizer (value function, policy, trajectories, animation)
├── tests/                  # 66 unit tests (pytest)
├── ipynb/
│   ├── development.ipynb   # Phase-by-phase development and experiment notebook
│   └── report.ipynb        # Lab report
├── results/                # Trained agents (.npz), experiment registry, trajectories
├── img/                    # Generated visualizations
└── environment.yml         # Conda environment
```

## Setup

```bash
conda env create -f environment.yml
conda activate gym_env
```

## Running Experiments

Experiments are run and cached from `ipynb/development.ipynb`. Open the notebook and run all cells. Already-trained agents are loaded from `results/` automatically; training is skipped if the agent file exists.

## Outputs

| File | Description |
|------|-------------|
| `results/agent_Ex*.npz` | Trained agent weight vectors for each experiment |
| `results/experiments.csv` | Aggregate metrics for all experiments |
| `results/trajectories.csv` | Per-step trajectory log for all training episodes |
| `results/episode_player.html` | Animated episode playback (car + state space view) |
| `img/convergence_curves.png` | Episode length and success rate curves by configuration |
| `img/learning_rates.png` | Success rate curves with 50% threshold marker |
| `img/value_function_plot.png` | Learned value function heatmap (max Q(s,a)) |
| `img/learned_policy_plot.png` | Greedy action across the state space |
| `img/trajectories_overlay.png` | Greedy rollouts from 5 starting positions (overlay) |
| `img/trajectories_thumbnails.png` | Greedy rollouts from 5 starting positions (individual) |

## Running Tests

From the project root:

```bash
/path/to/gym_env/bin/python -m pytest tests/ -v
```

66 tests across environment, tile coder, agent, episode runner, recorder, trainer, and visualizer.

## Key Results

- **All five agents solved MountainCar**, with success rates ranging from 66.2% (Ex5, 16×16 tiles) to 91.7% (Ex3, 16 tilings).
- **Tile resolution drives outcome quality**: 8×8 tiles produced the highest success rates; 16×16 tiles hurt performance within the 2500-episode budget due to sparse feature overlap.
- **Tiling count drives training cost**: wall time scales nearly linearly with tiling count (26.2s, 42.8s, 74.3s for 4, 8, and 16 tilings), with modest gains in success rate.
- **Best configuration by success rate**: Ex3 (16 tilings, 8×8, 91.7%).
- **Best configuration by efficiency**: Ex2 (4 tilings, 8×8, 3.32 successes/sec, std=8.7).

## References

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapters 9–10 (Function Approximation), p. 216 (tile coding receptive fields).
