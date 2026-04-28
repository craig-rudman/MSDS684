import gymnasium as gym


class TaxiEnv:
    def __init__(self, gym_env: gym.Env | None = None):
        self._env = gym_env if gym_env is not None else gym.make("Taxi-v4")

    @property
    def num_actions(self) -> int:
        return int(self._env.action_space.n)

    @property
    def num_states(self) -> int:
        return int(self._env.observation_space.n)

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            obs, _info = self._env.reset(seed=seed)
        else:
            obs, _info = self._env.reset()
        return int(obs)

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        obs, reward, terminated, truncated, _info = self._env.step(action)
        return int(obs), float(reward), bool(terminated), bool(truncated)

    def decode_state(self, state: int) -> tuple[int, int, int, int]:
        return tuple(self._env.unwrapped.decode(state))
