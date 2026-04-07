import gymnasium as gym


class EnvironmentManager:
    def __init__(self, env_name: str = 'CliffWalking-v1'):
        self._env = gym.make(env_name)

    @property
    def n_states(self) -> int:
        return self._env.observation_space.n

    @property
    def n_actions(self) -> int:
        return self._env.action_space.n

    def reset(self, seed: int = None) -> int:
        state, _ = self._env.reset(seed=seed)
        return int(state)

    def step(self, action: int) -> tuple:
        state, reward, terminated, truncated, _ = self._env.step(action)
        return int(state), float(reward), bool(terminated), bool(truncated)
