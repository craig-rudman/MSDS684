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
        super().reset(seed=seed)
        self._state = 0
        return self._state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Take an action and return (obs, reward, terminated, truncated, info)."""
        assert self._state is not None, "Call reset() before step()"
        transitions = self.P[self._state][action]
        probs = [t[0] for t in transitions]
        idx = self.np_random.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[idx]
        self._state = next_state
        return next_state, reward, done, False, {}

    def render(self) -> str | None:
        """Render the current grid state."""
        if self.render_mode != "ansi":
            return None
        grid = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                s = self.rc_to_state(r, c)
                if s == self._state:
                    row.append("A")
                elif (r, c) in self.obstacles:
                    row.append("X")
                elif (r, c) in self.terminal_states:
                    row.append("G")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        return "\n".join(grid)

    # ------------------------------------------------------------------
    # Coordinate / state helpers
    # ------------------------------------------------------------------

    def state_to_rc(self, state: int) -> tuple[int, int]:
        """Convert a flat state index to (row, col)."""
        return divmod(state, self.size)

    def rc_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) to a flat state index."""
        return row * self.size + col

    # ------------------------------------------------------------------
    # Transition model
    # ------------------------------------------------------------------

    def _build_transition_model(self) -> None:
        """Construct the full transition model ``self.P``.

        For every (state, action) pair, populates a list of
        (probability, next_state, reward, done) tuples.
        """
        # Direction deltas: UP, RIGHT, DOWN, LEFT
        deltas = {self.UP: (-1, 0), self.RIGHT: (0, 1),
                  self.DOWN: (1, 0), self.LEFT: (0, -1)}
        # Perpendicular actions for stochastic transitions
        perpendicular = {
            self.UP: [self.LEFT, self.RIGHT],
            self.RIGHT: [self.UP, self.DOWN],
            self.DOWN: [self.LEFT, self.RIGHT],
            self.LEFT: [self.UP, self.DOWN],
        }

        for s in range(self.n_states):
            self.P[s] = {}
            for a in range(self.n_actions):
                if self.stochastic:
                    side_prob = (1.0 - self.intended_prob) / 2.0
                    action_probs = [(self.intended_prob, a)]
                    for perp_a in perpendicular[a]:
                        action_probs.append((side_prob, perp_a))
                else:
                    action_probs = [(1.0, a)]

                # Build transitions, combining probabilities for duplicate next states
                combined: dict[int, tuple[float, float, bool]] = {}
                for prob, actual_action in action_probs:
                    ns = self._get_next_state(s, actual_action)
                    rc = self.state_to_rc(ns)
                    reward = self.reward_map.get(rc, self.default_reward)
                    done = rc in self.terminal_states
                    if ns in combined:
                        old_p, old_r, old_d = combined[ns]
                        combined[ns] = (old_p + prob, old_r, old_d)
                    else:
                        combined[ns] = (prob, reward, done)

                self.P[s][a] = [(p, ns, r, d) for ns, (p, r, d) in combined.items()]

    def _get_next_state(self, state: int, action: int) -> int:
        """Return the next state after taking ``action`` from ``state``.

        Handles boundaries and obstacles (agent stays in place).
        """
        deltas = {self.UP: (-1, 0), self.RIGHT: (0, 1),
                  self.DOWN: (1, 0), self.LEFT: (0, -1)}
        r, c = self.state_to_rc(state)
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc
        # Stay in place if out of bounds or hitting an obstacle
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            return state
        if (nr, nc) in self.obstacles:
            return state
        return self.rc_to_state(nr, nc)

    def get_transition_model(
        self,
    ) -> dict[int, dict[int, list[tuple[float, int, float, bool]]]]:
        """Return the full transition model P(s', r | s, a).

        Returns
        -------
        dict
            ``P[s][a]`` is a list of ``(prob, next_state, reward, done)`` tuples.
        """
        return self.P
