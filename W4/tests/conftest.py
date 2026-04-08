import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from schedules import ConstantSchedule, LinearDecaySchedule, ExponentialDecaySchedule
from agents import SARSAAgent, QLearningAgent
from environment import EnvironmentManager


N_STATES = 48
N_ACTIONS = 4
ALPHA = 0.1
GAMMA = 1.0


@pytest.fixture
def constant_schedule():
    return ConstantSchedule(epsilon=0.1)


@pytest.fixture
def linear_schedule():
    return LinearDecaySchedule(epsilon_start=1.0, epsilon_end=0.01, n_episodes=500)


@pytest.fixture
def exp_schedule():
    return ExponentialDecaySchedule(epsilon_start=1.0, epsilon_end=0.01, n_episodes=500)


@pytest.fixture
def sarsa_agent(constant_schedule):
    return SARSAAgent(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        schedule=constant_schedule,
    )


@pytest.fixture
def qlearning_agent(constant_schedule):
    return QLearningAgent(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        schedule=constant_schedule,
    )


@pytest.fixture
def env_manager():
    return EnvironmentManager('CliffWalking-v1')
