import gymnasium as gym


class DynamicTaxiWrapper(gym.Wrapper):
    def __init__(self, mutation_step: int = 1000):
        env = gym.make("Taxi-v4")
        super().__init__(env)
        self.mutation_step = mutation_step
        self.total_steps = 0
        self._mutated = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        if self.total_steps == self.mutation_step and not self._mutated:
            self._mutate()
        return obs, reward, terminated, truncated, info

    def _mutate(self):
        locs = self.env.unwrapped.locs
        locs[0], locs[3] = locs[3], locs[0]
        self._mutated = True
