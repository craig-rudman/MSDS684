import os
import torch
import pandas as pd
from datetime import datetime


class Trainer:
    def __init__(self, agent, env, label, seed=42):
        self.agent = agent
        self.env = env
        self.label = label
        self.seed = seed
        self.data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    def train(self, num_episodes):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # act_low/act_high are tensors and don't serialize cleanly as CSV columns
        config = {k: v for k, v in self.agent._config.items()
                  if k not in ("act_low", "act_high")}
        config["label"] = self.label
        if self.seed is not None:
            config["seed"] = self.seed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        records = []

        for episode in range(num_episodes):
            reset_seed = self.seed if (episode == 0 and self.seed is not None) else None
            obs, _ = self.env.reset(seed=reset_seed)
            done = False
            step = 0
            episode_return = 0.0
            episode_records = []

            while not done:
                obs_tensor = torch.FloatTensor(obs)
                action, log_prob, value, entropy = self.agent.select_action(obs_tensor)

                next_obs, reward, terminated, truncated, _ = self.env.step(
                    action.detach().numpy()
                )
                done = terminated or truncated
                episode_return += float(reward)

                next_value = (torch.tensor(0.0) if done
                              else self.agent.get_value(torch.FloatTensor(next_obs)))

                update = self.agent.update(log_prob, value, next_value, reward, done)

                episode_records.append({
                    "episode": episode,
                    "step": step,
                    "reward": float(reward),
                    "td_error": update["td_error"],
                    "entropy": entropy.item(),
                    "x": float(obs[0]),
                    "y": float(obs[1]),
                    "angle": float(obs[4]),
                })

                obs = next_obs
                step += 1

            for r in episode_records:
                r["episode_return"] = episode_return
                r["episode_length"] = step
                r.update(config)

            records.extend(episode_records)

        df = pd.DataFrame(records)
        os.makedirs(self.data_dir, exist_ok=True)
        csv_path = os.path.join(self.data_dir, f"{self.label}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)

        return df
