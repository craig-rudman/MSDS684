"""Utilities for inspecting Gymnasium environment spaces and MDP structure."""

import gymnasium as gym
from gymnasium import spaces


def describe_space(space: gym.Space) -> dict:
    """
    Return a structured summary of a Gymnasium space.

    Handles Discrete, Box, MultiDiscrete, and MultiBinary spaces.

    Args:
        space: A Gymnasium space instance.

    Returns:
        Dict with keys: type, dimensions, and type-specific details
        (n for Discrete, low/high/shape/dtype for Box, etc.).
    """
    info: dict = {"type": type(space).__name__}

    if isinstance(space, spaces.Discrete):
        info["n"] = int(space.n)
        info["start"] = int(space.start)
        info["dimensions"] = 1
    elif isinstance(space, spaces.Box):
        info["shape"] = space.shape
        info["low"] = space.low.tolist()
        info["high"] = space.high.tolist()
        info["dtype"] = str(space.dtype)
        info["dimensions"] = int(space.shape[0]) if len(space.shape) == 1 else space.shape
    elif isinstance(space, spaces.MultiDiscrete):
        info["nvec"] = space.nvec.tolist()
        info["dimensions"] = len(space.nvec)
    elif isinstance(space, spaces.MultiBinary):
        info["n"] = space.n
        info["dimensions"] = space.n
    else:
        info["repr"] = repr(space)

    return info


def inspect_env(env: gym.Env) -> dict:
    """
    Inspect an environment's observation and action spaces.

    Args:
        env: A Gymnasium environment instance.

    Returns:
        Dict with observation_space and action_space summaries,
        plus environment metadata.
    """
    return {
        "env_id": env.spec.id if env.spec else type(env).__name__,
        "observation_space": describe_space(env.observation_space),
        "action_space": describe_space(env.action_space),
        "reward_range": getattr(env, "reward_range", None),
        "max_episode_steps": (
            env.spec.max_episode_steps if env.spec else None
        ),
    }


def get_transition_table(env: gym.Env) -> dict | None:
    """
    Extract the transition probability table P(s'|s,a) if available.

    Works for environments like FrozenLake-v1 and Taxi-v3 that expose
    env.unwrapped.P as a dict of {state: {action: [(prob, next_state, reward, done)]}}.

    Args:
        env: A Gymnasium environment instance.

    Returns:
        The transition dict, or None if not available.
    """
    return getattr(env.unwrapped, "P", None)


def format_inspection(info: dict) -> str:
    """
    Format an inspection dict as a readable string.

    Args:
        info: Output from inspect_env().

    Returns:
        A human-readable multi-line summary.
    """
    lines = [f"Environment: {info['env_id']}", ""]

    obs = info["observation_space"]
    lines.append(f"Observation Space ({obs['type']}):")
    if obs["type"] == "Discrete":
        lines.append(f"  States: {obs['n']} (indexed {obs['start']}..{obs['start'] + obs['n'] - 1})")
    elif obs["type"] == "Box":
        lines.append(f"  Shape: {obs['shape']}, dtype: {obs['dtype']}")
        lines.append(f"  Bounds: [{obs['low']}, {obs['high']}]")
    lines.append(f"  Dimensions: {obs['dimensions']}")

    lines.append("")
    act = info["action_space"]
    lines.append(f"Action Space ({act['type']}):")
    if act["type"] == "Discrete":
        lines.append(f"  Actions: {act['n']} (indexed {act['start']}..{act['start'] + act['n'] - 1})")
    elif act["type"] == "Box":
        lines.append(f"  Shape: {act['shape']}, dtype: {act['dtype']}")
        lines.append(f"  Bounds: [{act['low']}, {act['high']}]")
    lines.append(f"  Dimensions: {act['dimensions']}")

    if info["max_episode_steps"] is not None:
        lines.append(f"\nMax episode steps: {info['max_episode_steps']}")

    return "\n".join(lines)
