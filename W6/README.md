# Week 6: Policy Gradient Methods

Craig Rudman, Regis University, MSDS684 Reinforcement Learning (Prof. Mike Busch).

A one-step Actor-Critic agent trained on Gymnasium's `LunarLanderContinuous-v3` environment, following the one-step actor-critic method described in Sutton and Barto, Chapter 13. Independent agents were trained across 30 random seeds for 1,000 episodes each; final policy evaluation averages the mean action across all 30 trained actors (ensemble voting).

## Environment

The project targets Python 3.11 and runs inside a conda environment named `gym_env`.

```bash
conda env create -f environment.yml
conda activate gym_env
```

Key dependencies include `torch==2.10.0`, `gymnasium==1.2.3` with `box2d==2.3.10`, `numpy`, `pandas`, `matplotlib`, `tqdm`, and `pytest`.

## Repository Layout

```
W6/
├── src/
│   ├── agent.py        # Actor, Critic, and Agent (one-step TD updates, save/load)
│   ├── trainer.py      # Episode loop, CSV logging, checkpointing
│   └── ensemble.py     # Mean-action voting across trained actors
├── test/               # pytest suites for actor, critic, agent, trainer, ensemble
├── ipynb/
│   ├── development.ipynb   # Exploratory training and diagnostics
│   ├── report.ipynb        # Final report (submission)
│   └── report.html         # Rendered report
├── img/                # Figures and episode replays referenced in the report
├── environment.yml
└── README.md
```

Training CSVs land in `data/` (gitignored for test runs). Episode replay GIFs and result figures live in [img/](img/).

## Running

Use the `gym_env` Python interpreter directly:

```bash
# Run the test suite
/Users/crudman/anaconda3/envs/gym_env/bin/python -m pytest test/

# Launch JupyterLab for the notebooks
/Users/crudman/anaconda3/envs/gym_env/bin/jupyter lab
```

Training is orchestrated from the notebooks. A minimal programmatic invocation:

```python
import gymnasium as gym
from src.agent import Agent
from src.trainer import Trainer

env = gym.make("LunarLanderContinuous-v3")
agent = Agent(
    obs_dim=8, act_dim=2,
    actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
    act_low=env.action_space.low, act_high=env.action_space.high,
    hidden_sizes=(64, 64),
)
Trainer(agent, env, label="run_s0", seed=0).train(num_episodes=1000)
```

## Architecture

- **Actor:** two hidden layers of 64 units with ReLU; outputs mean μ(s) and a state-independent log σ clamped to [-20, 2]. Actions are sampled from a Gaussian and clipped to the environment bounds.
- **Critic:** matching MLP trunk with a scalar value head V(s).
- **Update:** one-step TD target `r + γ · V(s') · (1 - done)` with the bootstrap value detached. The critic minimizes squared TD error; the actor loss is `-log π(a|s) · δ.detach()`.
- **Ensemble:** [src/ensemble.py](src/ensemble.py) averages μ(s) across actors for deterministic rollouts.

## Final Hyperparameters

| Parameter     | Value         |
|---------------|---------------|
| actor_lr      | 1e-4          |
| critic_lr     | 1e-3          |
| gamma         | 0.99          |
| hidden_sizes  | (64, 64)      |
| episodes/seed | 1,000         |
| seeds         | 30            |

## Results Summary

Mean return across 30 seeds climbed through roughly the first 200 episodes and then plateaued well below the solve threshold of 200 points, with a bimodal per-seed distribution (cross-seed mean 24.8, median 27.5). Policy entropy collapsed consistently across all seeds regardless of outcome, indicating the learning mechanism commits actors to whatever policy they find. The ensemble policy demonstrated controlled descent with partial lateral correction but did not reliably land. See [ipynb/report.ipynb](ipynb/report.ipynb) for the full write-up and figures.

## Deliverables

- Report notebook: [ipynb/report.ipynb](ipynb/report.ipynb) (rendered: [ipynb/report.html](ipynb/report.html))
- Development notebook: [ipynb/development.ipynb](ipynb/development.ipynb)
- Source: [src/](src/)
- Tests: [test/](test/)
- Figures and replays: [img/](img/)
