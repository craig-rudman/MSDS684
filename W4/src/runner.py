import numpy as np
from agents import TDAgent
from environment import EnvironmentManager


class EpisodeRunner:
    def __init__(self, agent: TDAgent, env_manager: EnvironmentManager):
        self.agent = agent
        self.env = env_manager

    def run_episode(self) -> float:
        state = self.env.reset()
        action = self.agent.select_action(state)
        total_reward = 0.0
        while True:
            next_state, reward, terminated, truncated = self.env.step(action)
            next_action = self.agent.select_action(next_state)
            self.agent.update(state, action, reward, next_state, next_action, terminated)
            total_reward += reward
            state, action = next_state, next_action
            if terminated or truncated:
                break
        self.agent.decay_epsilon()
        return total_reward

    def run_experiment(self, n_seeds: int, n_episodes: int, base_seed: int = 0) -> np.ndarray:
        matrix = np.zeros((n_seeds, n_episodes))
        for seed_idx in range(n_seeds):
            self.agent.schedule.reset()
            self.agent.reset_qtable()
            self.env.reset(seed=base_seed + seed_idx)
            for ep in range(n_episodes):
                matrix[seed_idx, ep] = self.run_episode()
        return matrix
