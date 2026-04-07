import gymnasium as gym


class EnvironmentManager:
    def __init__(self, env_name: str = 'CliffWalking-v1'):
        pass

    @property
    def n_states(self) -> int:
        pass

    @property
    def n_actions(self) -> int:
        pass

    def reset(self, seed: int = None) -> int:
        pass

    def step(self, action: int) -> tuple:
        pass
