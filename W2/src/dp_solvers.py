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

        # Pre-compute terminal state indices for fast lookup
        self._terminal_states = {
            env.rc_to_state(*t) for t in env.terminal_states
        }

        # History for visualization / analysis
        self.value_history: list[np.ndarray] = [self.V.copy()]
        self.policy_history: list[np.ndarray] = [self.policy.copy()]
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
        P = self.env.P
        sweeps = 0
        while True:
            V_new = np.zeros_like(self.V)
            for s in range(self.env.n_states):
                if s in self._terminal_states:
                    continue
                a = self.policy[s]
                for prob, s_prime, reward, done in P[s][a]:
                    if done:
                        V_new[s] += prob * reward
                    else:
                        V_new[s] += prob * (reward + self.gamma * self.V[s_prime])
            sweeps += 1
            delta = np.max(np.abs(V_new - self.V))
            self.V = V_new
            if delta < self.theta:
                break
        return sweeps

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
        P = self.env.P
        sweeps = 0
        while True:
            delta = 0.0
            for s in range(self.env.n_states):
                if s in self._terminal_states:
                    continue
                v_old = self.V[s]
                a = self.policy[s]
                new_val = 0.0
                for prob, s_prime, reward, done in P[s][a]:
                    if done:
                        new_val += prob * reward
                    else:
                        new_val += prob * (reward + self.gamma * self.V[s_prime])
                self.V[s] = new_val
                delta = max(delta, abs(v_old - new_val))
            sweeps += 1
            if delta < self.theta:
                break
        return sweeps

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
        P = self.env.P
        stable = True
        for s in range(self.env.n_states):
            if s in self._terminal_states:
                continue
            old_action = self.policy[s]
            q_values = np.zeros(self.env.n_actions)
            for a in range(self.env.n_actions):
                for prob, s_prime, reward, done in P[s][a]:
                    if done:
                        q_values[a] += prob * reward
                    else:
                        q_values[a] += prob * (reward + self.gamma * self.V[s_prime])
            self.policy[s] = np.argmax(q_values)
            if old_action != self.policy[s]:
                stable = False
        return stable

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
        import time

        evaluate = self.evaluate_sync if mode == "sync" else self.evaluate_inplace

        while True:
            t0 = time.perf_counter()
            eval_sweeps = evaluate()
            self.eval_iterations.append(eval_sweeps)
            stable = self.improve()
            elapsed = time.perf_counter() - t0
            self.wall_clock_times.append(elapsed)
            self.value_history.append(self.V.copy())
            self.policy_history.append(self.policy.copy())
            if stable:
                break

        return self.V, self.policy


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
        self.value_history: list[np.ndarray] = [self.V.copy()]
        self.policy_history: list[np.ndarray] = [self.policy.copy()]
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
        import time

        P = self.env.P
        terminal = {self.env.rc_to_state(*t) for t in self.env.terminal_states}

        while True:
            t0 = time.perf_counter()
            V_new = np.zeros_like(self.V)
            for s in range(self.env.n_states):
                if s in terminal:
                    continue
                q_values = np.zeros(self.env.n_actions)
                for a in range(self.env.n_actions):
                    for prob, s_prime, reward, done in P[s][a]:
                        if done:
                            q_values[a] += prob * reward
                        else:
                            q_values[a] += prob * (reward + self.gamma * self.V[s_prime])
                V_new[s] = np.max(q_values)
            delta = np.max(np.abs(V_new - self.V))
            self.V = V_new
            self.iterations += 1
            elapsed = time.perf_counter() - t0
            self.wall_clock_times.append(elapsed)
            self.value_history.append(self.V.copy())
            if delta < self.theta:
                break

        self.policy = self.extract_policy()
        self.policy_history.append(self.policy.copy())
        return self.V, self.policy

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
        import time

        P = self.env.P
        terminal = {self.env.rc_to_state(*t) for t in self.env.terminal_states}

        while True:
            t0 = time.perf_counter()
            delta = 0.0
            for s in range(self.env.n_states):
                if s in terminal:
                    continue
                v_old = self.V[s]
                q_values = np.zeros(self.env.n_actions)
                for a in range(self.env.n_actions):
                    for prob, s_prime, reward, done in P[s][a]:
                        if done:
                            q_values[a] += prob * reward
                        else:
                            q_values[a] += prob * (reward + self.gamma * self.V[s_prime])
                self.V[s] = np.max(q_values)
                delta = max(delta, abs(v_old - self.V[s]))
            self.iterations += 1
            elapsed = time.perf_counter() - t0
            self.wall_clock_times.append(elapsed)
            self.value_history.append(self.V.copy())
            if delta < self.theta:
                break

        self.policy = self.extract_policy()
        self.policy_history.append(self.policy.copy())
        return self.V, self.policy

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
        P = self.env.P
        terminal = {self.env.rc_to_state(*t) for t in self.env.terminal_states}
        policy = np.zeros(self.env.n_states, dtype=int)
        for s in range(self.env.n_states):
            if s in terminal:
                continue
            q_values = np.zeros(self.env.n_actions)
            for a in range(self.env.n_actions):
                for prob, s_prime, reward, done in P[s][a]:
                    if done:
                        q_values[a] += prob * reward
                    else:
                        q_values[a] += prob * (reward + self.gamma * self.V[s_prime])
            policy[s] = np.argmax(q_values)
        return policy

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
        if mode == "sync":
            return self.solve_sync()
        return self.solve_inplace()
