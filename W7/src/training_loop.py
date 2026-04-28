import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent import Agent
from taxi_env import TaxiEnv


TRACE_COLUMNS = [
    "agent_id",
    "seed",
    "episode",
    "total_reward",
    "num_steps",
    "cumulative_steps",
    "terminated",
    "truncated",
    "wall_time_episode",
    "wall_time_planning_episode",
    "n_planning_updates_episode",
    "cumulative_planning_updates",
]

RESERVED_HYPERPARAM_KEYS = {"agent_id", "seed", "num_episodes"}


class TrainingLoop:
    def __init__(
        self,
        env: TaxiEnv,
        agent: Agent,
        agent_id: str,
        output_dir: Path | str,
        hyperparams: dict | None = None,
        record_per_step: bool = False,
    ):
        self.env = env
        self.agent = agent
        self.agent_id = agent_id
        self.output_dir = Path(output_dir)
        self.hyperparams = hyperparams or {}
        self.record_per_step = record_per_step

        reserved_collisions = RESERVED_HYPERPARAM_KEYS & self.hyperparams.keys()
        if reserved_collisions:
            raise ValueError(
                f"Reserved keys {reserved_collisions} cannot appear in hyperparams; "
                f"the driver records these from run() args."
            )

    def run(self, num_episodes: int, seed: int) -> pd.DataFrame:
        if self.record_per_step:
            raise NotImplementedError(
                "Forensic per-step recording is not yet implemented."
            )

        episode_rows: list[dict] = []
        cumulative_steps = 0
        cumulative_planning_updates = 0
        state = self.env.reset(seed=seed)

        for episode in range(num_episodes):
            if episode > 0:
                state = self.env.reset()

            ep_total_reward = 0.0
            ep_num_steps = 0
            ep_wall_time = 0.0
            ep_wall_time_planning = 0.0
            ep_n_planning_updates = 0
            ep_terminated = False
            ep_truncated = False

            while True:
                action = self.agent.act(state)

                start = time.perf_counter()
                next_state, reward, terminated, truncated = self.env.step(action)
                stats = self.agent.learn(state, action, reward, next_state, terminated, truncated)
                wall_time_step = time.perf_counter() - start

                ep_total_reward += reward
                ep_num_steps += 1
                ep_wall_time += wall_time_step
                ep_wall_time_planning += stats.get("wall_time_planning", 0.0)
                ep_n_planning_updates += stats.get("n_planning_updates", 0)
                state = next_state

                if terminated or truncated:
                    ep_terminated = terminated
                    ep_truncated = truncated
                    break

            cumulative_steps += ep_num_steps
            cumulative_planning_updates += ep_n_planning_updates
            episode_rows.append({
                "agent_id": self.agent_id,
                "seed": seed,
                "episode": episode,
                "total_reward": ep_total_reward,
                "num_steps": ep_num_steps,
                "cumulative_steps": cumulative_steps,
                "terminated": ep_terminated,
                "truncated": ep_truncated,
                "wall_time_episode": ep_wall_time,
                "wall_time_planning_episode": ep_wall_time_planning,
                "n_planning_updates_episode": ep_n_planning_updates,
                "cumulative_planning_updates": cumulative_planning_updates,
            })

        trace = pd.DataFrame(episode_rows, columns=TRACE_COLUMNS)
        self._persist(trace, seed, num_episodes)
        return trace

    def _persist(self, trace: pd.DataFrame, seed: int, num_episodes: int) -> None:
        agent_dir = self.output_dir / self.agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        trace_path = agent_dir / f"trace_seed_{seed}.csv"
        trace.to_csv(trace_path, index=False)

        config = {
            "agent_id": self.agent_id,
            "seed": seed,
            "num_episodes": num_episodes,
            "hyperparams": self.hyperparams,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        config_path = agent_dir / f"config_seed_{seed}.json"
        config_path.write_text(json.dumps(config, indent=2))
