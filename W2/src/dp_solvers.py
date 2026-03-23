"""Dynamic programming solvers: Policy Iteration and Value Iteration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .gridworld import GridWorld


class PolicyIteration:
    """Policy Iteration with both synchronous and in-place evaluation.

    Parameters
    ----------
    env : GridWorld
        Environment exposing ``.n_states``, ``.n_actions``, and ``.P``.
    gamma : float
        Discount factor (default 0.99).
    theta : float
        Convergence threshold for policy evaluation (default 1e-8).
    """

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.99,
        theta: float = 1e-8,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta

        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)

        # History for visualization / analysis
        self.value_history: list[np.ndarray] = []
        self.policy_history: list[np.ndarray] = []
        self.eval_iterations: list[int] = []
        self.wall_clock_times: list[float] = []

    # ------------------------------------------------------------------
    # Synchronous policy evaluation
    # ------------------------------------------------------------------

    def evaluate_sync(self) -> int:
        """Evaluate the current policy using synchronous (two-array) updates.

        Returns
        -------
        int
            Number of sweeps until convergence.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # In-place policy evaluation
    # ------------------------------------------------------------------

    def evaluate_inplace(self) -> int:
        """Evaluate the current policy using in-place (single-array) updates.

        Returns
        -------
        int
            Number of sweeps until convergence.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Policy improvement
    # ------------------------------------------------------------------

    def improve(self) -> bool:
        """Improve the policy greedily with respect to the current value function.

        Returns
        -------
        bool
            True if the policy is stable (no changes), False otherwise.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Full algorithm
    # ------------------------------------------------------------------

    def solve(self, mode: str = "sync") -> tuple[np.ndarray, np.ndarray]:
        """Run policy iteration until the policy is stable.

        Parameters
        ----------
        mode : str
            ``"sync"`` for synchronous evaluation, ``"inplace"`` for in-place.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Final (value_function, policy).
        """
        raise NotImplementedError


class ValueIteration:
    """Value Iteration with both synchronous and in-place updates.

    Parameters
    ----------
    env : GridWorld
        Environment exposing ``.n_states``, ``.n_actions``, and ``.P``.
    gamma : float
        Discount factor (default 0.99).
    theta : float
        Convergence threshold (default 1e-8).
    """

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.99,
        theta: float = 1e-8,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta

        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)

        # History for visualization / analysis
        self.value_history: list[np.ndarray] = []
        self.policy_history: list[np.ndarray] = []
        self.iterations: int = 0
        self.wall_clock_times: list[float] = []

    # ------------------------------------------------------------------
    # Synchronous value iteration
    # ------------------------------------------------------------------

    def solve_sync(self) -> tuple[np.ndarray, np.ndarray]:
        """Run value iteration with synchronous (two-array) updates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Final (value_function, policy).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # In-place value iteration
    # ------------------------------------------------------------------

    def solve_inplace(self) -> tuple[np.ndarray, np.ndarray]:
        """Run value iteration with in-place (single-array) updates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Final (value_function, policy).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def extract_policy(self) -> np.ndarray:
        """Extract greedy policy from the converged value function.

        Returns
        -------
        np.ndarray
            Policy array of shape ``(n_states,)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    def solve(self, mode: str = "sync") -> tuple[np.ndarray, np.ndarray]:
        """Run value iteration in the specified mode.

        Parameters
        ----------
        mode : str
            ``"sync"`` or ``"inplace"``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Final (value_function, policy).
        """
        raise NotImplementedError
