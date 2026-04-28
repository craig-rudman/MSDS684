import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Agent


def test_cannot_instantiate_abstract_agent():
    with pytest.raises(TypeError):
        Agent()


def test_concrete_subclass_implementing_both_methods_instantiates():
    class ConcreteAgent(Agent):
        def act(self, state):
            return 0

        def learn(self, state, action, reward, next_state, terminated, truncated):
            return {}

    agent = ConcreteAgent()
    assert agent is not None
    assert agent.act(0) == 0
    assert agent.learn(0, 0, 0.0, 0, False, False) == {}


def test_subclass_missing_act_cannot_instantiate():
    class IncompleteAgent(Agent):
        def learn(self, state, action, reward, next_state, terminated, truncated):
            return {}

    with pytest.raises(TypeError):
        IncompleteAgent()


def test_subclass_missing_learn_cannot_instantiate():
    class IncompleteAgent(Agent):
        def act(self, state):
            return 0

    with pytest.raises(TypeError):
        IncompleteAgent()
