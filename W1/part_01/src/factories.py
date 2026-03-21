"""Factory functions for creating environments and agents."""

from typing import Optional

from .bandit_env import MultiArmedBanditEnv
from .agents import BanditAgent, EpsilonGreedyAgent, UCBAgent


_AGENT_REGISTRY = {
    "epsilon_greedy": EpsilonGreedyAgent,
    "ucb": UCBAgent,
}


def make_bandit_env(
    k: int = 10,
    arms: Optional[list[dict]] = None,
    randomize: bool = True,
    max_steps: int = 1000,
    seed: Optional[int] = None,
) -> MultiArmedBanditEnv:
    """
    Create a multi-armed bandit environment.

    Shorthand usage:
        env = make_bandit_env(k=10, randomize=True, max_steps=2000)

    Custom arm configs:
        env = make_bandit_env(arms=[
            {"mu": 1.0, "sigma": 0.5},
            {"mu": 2.0, "sigma": 1.0},
        ], max_steps=2000)

    Args:
        k: Number of arms (used when arms is None).
        arms: Explicit arm configurations as a list of dicts.
        randomize: If True, re-sample arm means from N(0,1) on each reset().
        max_steps: Number of steps per episode.
        seed: Random seed.

    Returns:
        A MultiArmedBanditEnv instance.
    """
    if arms is not None:
        k = len(arms)
        randomize = False  # explicit configs override randomization

    return MultiArmedBanditEnv(
        k=k,
        arms=arms,
        randomize=randomize,
        max_steps=max_steps,
        seed=seed,
    )


def make_agent(
    name: str,
    k: int,
    seed: Optional[int] = None,
    **kwargs,
) -> BanditAgent:
    """
    Create a bandit agent by name.

    Auto-generates a descriptive name for plot legends.

    Args:
        name: Agent type ("epsilon_greedy" or "ucb").
        k: Number of arms.
        seed: Random seed.
        **kwargs: Agent-specific parameters (epsilon, c, etc.).

    Returns:
        A BanditAgent instance with auto-generated agent_name attribute.
    """
    if name not in _AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent '{name}'. Options: {list(_AGENT_REGISTRY)}"
        )

    agent = _AGENT_REGISTRY[name](k=k, seed=seed, **kwargs)

    # Auto-generate descriptive name for plot legends
    if name == "epsilon_greedy":
        eps = kwargs.get("epsilon", 0.1)
        agent.agent_name = f"ε-greedy (ε={eps})"
    elif name == "ucb":
        c = kwargs.get("c", 2.0)
        agent.agent_name = f"UCB (c={c})"
    else:
        agent.agent_name = name

    return agent
