"""Custom GridWorld environment built on the Gymnasium API."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorld(gym.Env):
    """A configurable GridWorld environment.

    Supports configurable rewards, obstacles, and stochastic transitions.
    Exposes the full transition model P(s', r | s, a) for use by DP solvers.

    Parameters
    ----------
    size : int
        Side length of the square grid (default 4, i.e. 4x4).
    obstacles : list[tuple[int, int]] | None
        Grid cells that are impassable.
    rewards : dict[tuple[int, int], float] | None
        Mapping of (row, col) to reward received upon entering that cell.
        All other transitions yield ``default_reward``.
    default_reward : float
        Reward for transitions not specified in ``rewards`` (default -1).
    terminal_states : list[tuple[int, int]] | None
        Cells that end the episode. Defaults to bottom-right corner.
    stochastic : bool
        Whether transitions are stochastic (default False).
    intended_prob : float
        Probability of moving in the intended direction when stochastic
        (default 0.8). Remaining probability is split equally among the
        two perpendicular directions.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Action constants
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

    def __init__(
        self,
        size: int = 4,
        obstacles: list[tuple[int, int]] | None = None,
        rewards: dict[tuple[int, int], float] | None = None,
        default_reward: float = -1.0,
        terminal_states: list[tuple[int, int]] | None = None,
        stochastic: bool = False,
        intended_prob: float = 0.8,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.obstacles = set(obstacles or [])
        self.reward_map = rewards or {}
        self.default_reward = default_reward
        self.terminal_states = set(
            terminal_states if terminal_states is not None else [(size - 1, size - 1)]
        )
        self.stochastic = stochastic
        self.intended_prob = intended_prob
        self.render_mode = render_mode

        self.n_states = size * size
        self.n_actions = 4

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        # Will hold the full transition model: P[s][a] -> list of (prob, s', reward, done)
        self.P: dict[int, dict[int, list[tuple[float, int, float, bool]]]] = {}

        self._build_transition_model()
        self._state: int | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Reset the environment to the start state."""
        raise NotImplementedError

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Take an action and return (obs, reward, terminated, truncated, info)."""
        raise NotImplementedError

    def render(self) -> str | None:
        """Render the current grid state."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Coordinate / state helpers
    # ------------------------------------------------------------------

    def state_to_rc(self, state: int) -> tuple[int, int]:
        """Convert a flat state index to (row, col)."""
        raise NotImplementedError

    def rc_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) to a flat state index."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Transition model
    # ------------------------------------------------------------------

    def _build_transition_model(self) -> None:
        """Construct the full transition model ``self.P``.

        For every (state, action) pair, populates a list of
        (probability, next_state, reward, done) tuples.
        """
        raise NotImplementedError

    def _get_next_state(self, state: int, action: int) -> int:
        """Return the next state after taking ``action`` from ``state``.

        Handles boundaries and obstacles (agent stays in place).
        """
        raise NotImplementedError

    def get_transition_model(
        self,
    ) -> dict[int, dict[int, list[tuple[float, int, float, bool]]]]:
        """Return the full transition model P(s', r | s, a).

        Returns
        -------
        dict
            ``P[s][a]`` is a list of ``(prob, next_state, reward, done)`` tuples.
        """
        raise NotImplementedError
