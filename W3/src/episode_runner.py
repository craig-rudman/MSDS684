class EpisodeRunner:
    """Runs episodes in a Gymnasium environment using an agent's policy."""

    def __init__(self, env):
        self.env = env

    def run_episode(self, agent, verbose: bool = False) -> list:
        """Generate one complete episode trajectory.

        Args:
            agent: Agent providing select_action(state).
            verbose: If True, print each step's state, action, and reward.

        Returns:
            List of (state, action, reward) tuples.
        """
        episode = []
        state, _ = self.env.reset()
        terminated = False
        step_num = 0
        while not terminated:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            episode.append((state, action, reward))
            step_num += 1
            if verbose:
                action_name = "Hit" if action == 1 else "Stick"
                player_sum, dealer_card, usable_ace = state
                print(f"Step {step_num}: State=(sum={player_sum}, dealer={dealer_card}, "
                      f"ace={bool(usable_ace)}) Action={action_name} Reward={reward}")
            state = next_state
        if verbose:
            total = sum(r for _, _, r in episode)
            result = "Win" if total > 0 else "Loss" if total < 0 else "Draw"
            print(f"Episode finished: {step_num} steps, Result={result}")
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
