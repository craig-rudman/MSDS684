# Working Agreement

## Workflow Principles

1. **Backlog in Jupyter notebook** — TODO items as comments in Python code blocks (in `ipynb/`)
2. **Iterative development** — each iteration we:
   - Select backlog item(s) together
   - Implement only those items
   - Inspect and demonstrate the code
   - Write commit headline and body together (user handles actual commit/push)
3. **Human in the loop** — discuss the intended operation before coding
4. **Test-first** — write failing tests that define expected behavior (pytest in `test/`)
5. **Object-oriented** — encapsulate in classes with separation of concerns and clean interfaces (in `src/`)
6. **Persistent checkpoints** — save state to disk (in `data/`) at key points so we can restore after kernel restarts

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `ipynb/` | Jupyter notebooks (backlog lives here) |
| `src/` | Implementation classes |
| `test/` | pytest test files |
| `data/` | Data files / checkpoints |
| `img/` | Images |

---

# Lab Assignment: Planning and Learning Integration (Week 7)

**Environment:** Gymnasium Taxi-v3

**Reading:** Sutton & Barto Chapter 8

## Core Components

1. **Direct RL** — Q-learning updates from real experience using NumPy Q-table
2. **Model Learning** — Deterministic table-based model using Python dictionary: `model[(s,a)] = (r, s')`
3. **Planning** — Simulated Q-learning steps by sampling from model

## Deliverables

1. **Dyna-Q** — Compare pure Q-learning (n=0) vs Dyna-Q with n ∈ {5, 10, 50} planning steps
2. **Visualizations** — Cumulative reward over real time steps, episodes until optimal, sample efficiency
3. **Dynamic environment** — After 1000 steps, modify environment structure (custom wrapper)
4. **Dyna-Q+** — Exploration bonus κ√τ for state-action pairs not tried in τ steps; track with `time_since[(s,a)]`
5. **Prioritized sweeping** — Use `heapq` with priority = |TD error|; update predecessors above threshold
6. **Synthesis document** — When model-based vs model-free, sample complexity, computational trade-offs, model errors
7. **Optional** — Neural network dynamics model with PyTorch for continuous state space (e.g., MountainCar-v0)
