# Week 8: Modern Deep RL Exploration

Craig Rudman  
Regis University — MSDS684 Reinforcement Learning  
Prof. Mike Busch

## Overview

This project applies Deep Q-Networks (DQN) via Stable-Baselines3 to LunarLander-v3, using the rl-baselines3-zoo tuned configuration as a baseline. The study conducts hyperparameter sweeps across learning rate, exploration fraction, and target update interval, and ablation studies removing experience replay and the target network. Results are compared against the one-step actor-critic implementation from Week 6.

The central finding is methodological: DQN's performance distribution across 20 seeds is wide, right-skewed, and highly seed-dependent. A published benchmark number is a point estimate from that distribution, not a characterization of it.

## Repository Structure

```
W8/
├── data/
│   ├── baseline/               # 20-seed DQN baseline curves (.npy)
│   ├── sweep_lr/               # Learning rate sweep curves
│   ├── sweep_exploration/      # Exploration fraction sweep curves
│   ├── sweep_target_update/    # Target update interval sweep curves
│   ├── ablation_no_replay/     # No experience replay ablation curves
│   └── ablation_no_target/     # No target network ablation curves
├── img/                        # Generated figures (PNG)
├── ipynb/
│   ├── development.ipynb       # Training, experiment, and visualization code
│   └── report.ipynb            # Final report
├── environment.yml             # Conda environment specification
└── README.md
```

## Experiments

| Experiment | Description |
|---|---|
| Baseline | Zoo config (lr=6.3e-4, ef=0.12, tui=250, buffer=50k), 20 seeds |
| LR sweep | lr = 2e-4, 6.3e-4, 2e-3 |
| Exploration fraction sweep | ef = 0.06, 0.12, 0.24 |
| Target update interval sweep | tui = 125, 250, 500 |
| Ablation: no experience replay | buffer=1, batch=1 |
| Ablation: no target network | tui=100,000 (frozen) |

All experiments: 100,000 timesteps, 20 random seeds, LunarLander-v3.

## Reproducing Results

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate gym_env
```

### 2. Run the notebook

Open `ipynb/development.ipynb` in JupyterLab and run cells in order. Training results are cached to `data/` by seed; completed runs are skipped automatically.

To run a quick smoke test before committing to a full run, set `SMOKE = True` in the control cell. This runs 2 seeds for 2,000 steps and completes in seconds.

### 3. Generate figures

The visualization cells in `development.ipynb` regenerate all figures to `img/`. They depend only on the saved `.npy` files in `data/` and can be re-run independently of training.

## Key Dependencies

| Package | Version |
|---|---|
| Python | 3.11 |
| stable-baselines3 | 2.8.0 |
| gymnasium | 1.2.3 |
| torch | 2.10.0 |
| numpy | 2.4.3 |
| matplotlib | 3.10.8 |

See `environment.yml` for the full specification.

## References

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. A. (2013). Playing Atari with Deep Reinforcement Learning. *arXiv preprint arXiv:1312.5602*.

Raffin, A. (2020). RL Baselines3 Zoo. GitHub. https://github.com/DLR-RM/rl-baselines3-zoo

Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
