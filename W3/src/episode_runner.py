class EpisodeRunner:
    """Runs episodes in a Gymnasium environment using an agent's policy."""

    def __init__(self, env):
        self.env = env

    def run_episode(self, agent) -> list:
        """Generate one complete episode trajectory.

        Returns:
            List of (state, action, reward) tuples.
        """
        episode = []
        state, _ = self.env.reset()
        terminated = False
        while not terminated:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def run_training(self, agent, num_episodes: int) -> list:
        """Run multiple episodes, updating the agent after each.

        Returns:
            List of total rewards per episode.
        """
        rewards = []
        for _ in range(num_episodes):
            episode = self.run_episode(agent)
            agent.update(episode)
            episode_reward = sum(r for _, _, r in episode)
            rewards.append(episode_reward)
        return rewards
