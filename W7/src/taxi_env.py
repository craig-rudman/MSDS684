import gymnasium as gym


class TaxiEnv:
    def __init__(self, gym_env: gym.Env | None = None, seed: int | None = None):
        self._env = gym_env if gym_env is not None else gym.make("Taxi-v4")
        self._seed = seed
        self._next_reset_seed: int | None = seed

    def reset(self) -> int:
        if self._next_reset_seed is not None:
            obs, _info = self._env.reset(seed=self._next_reset_seed)
            self._next_reset_seed = None
        else:
            obs, _info = self._env.reset()
        return int(obs)

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        obs, reward, terminated, truncated, _info = self._env.step(action)
        return int(obs), float(reward), bool(terminated), bool(truncated)

    def decode_state(self, state: int) -> tuple[int, int, int, int]:
        return tuple(self._env.unwrapped.decode(state))
