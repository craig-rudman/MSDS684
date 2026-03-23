"""Shared fixtures for GridWorld DP tests."""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import GridWorld, PolicyIteration, ValueIteration, GridWorldVisualizer, ExperimentRunner


# ------------------------------------------------------------------
# GridWorld fixtures
# ------------------------------------------------------------------


@pytest.fixture
def env_4x4_deterministic():
    """A simple 4x4 deterministic GridWorld with one obstacle and one terminal state."""
    return GridWorld(
        size=4,
        obstacles=[(1, 1)],
        rewards={(3, 3): 1.0},
        default_reward=-0.04,
        terminal_states=[(3, 3)],
        stochastic=False,
    )


@pytest.fixture
def env_4x4_stochastic():
    """A 4x4 stochastic GridWorld (80% intended, 10% each perpendicular)."""
    return GridWorld(
        size=4,
        obstacles=[(1, 1)],
        rewards={(3, 3): 1.0},
        default_reward=-0.04,
        terminal_states=[(3, 3)],
        stochastic=True,
        intended_prob=0.8,
    )


@pytest.fixture
def env_2x2_simple():
    """Minimal 2x2 grid for analytical verification.

    Layout:
        S  .
        .  G

    No obstacles, terminal at (1,1) with reward 1.0, default reward -1.0.
    """
    return GridWorld(
        size=2,
        obstacles=[],
        rewards={(1, 1): 1.0},
        default_reward=-1.0,
        terminal_states=[(1, 1)],
        stochastic=False,
    )


# ------------------------------------------------------------------
# Solver fixtures
# ------------------------------------------------------------------


@pytest.fixture
def pi_deterministic(env_4x4_deterministic):
    return PolicyIteration(env_4x4_deterministic, gamma=0.99, theta=1e-8)


@pytest.fixture
def vi_deterministic(env_4x4_deterministic):
    return ValueIteration(env_4x4_deterministic, gamma=0.99, theta=1e-8)


@pytest.fixture
def pi_stochastic(env_4x4_stochastic):
    return PolicyIteration(env_4x4_stochastic, gamma=0.99, theta=1e-8)


@pytest.fixture
def vi_stochastic(env_4x4_stochastic):
    return ValueIteration(env_4x4_stochastic, gamma=0.99, theta=1e-8)


# ------------------------------------------------------------------
# Visualizer fixture
# ------------------------------------------------------------------


@pytest.fixture
def visualizer(env_4x4_deterministic):
    return GridWorldVisualizer(env_4x4_deterministic)
