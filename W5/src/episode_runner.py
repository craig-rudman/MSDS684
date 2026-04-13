from __future__ import annotations


class EpisodeRunner:
    """Runs a single episode and returns the step count."""

    def __init__(self, env, agent, recorder=None, seed=None) -> None:
        self.env = env
        self.agent = agent
        self.recorder = recorder
        self.seed = seed

    def run_episode(self) -> int:
        state, _ = self.env.reset(seed=self.seed)
        action = self.agent.select_action(state)
        steps = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_action = self.agent.select_action(next_state)
            self.agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            steps += 1

        return steps
