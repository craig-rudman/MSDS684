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
    "step",
    "episode",
    "step_in_episode",
    "state",
    "action",
    "reward",
    "next_state",
    "terminated",
    "truncated",
    "wall_time_step",
    "wall_time_planning",
    "n_planning_updates",
]


class TrainingLoop:
    def __init__(
        self,
        env: TaxiEnv,
        agent: Agent,
        agent_id: str,
        output_dir: Path | str,
        hyperparams: dict | None = None,
    ):
        self.env = env
        self.agent = agent
        self.agent_id = agent_id
        self.output_dir = Path(output_dir)
        self.hyperparams = hyperparams or {}

    def run(self, num_episodes: int, seed: int) -> pd.DataFrame:
        rows: list[dict] = []
        global_step = 0
        state = self.env.reset(seed=seed)

        for episode in range(num_episodes):
            if episode > 0:
                state = self.env.reset()

            step_in_episode = 0
            while True:
                action = self.agent.act(state)

                start = time.perf_counter()
                next_state, reward, terminated, truncated = self.env.step(action)
                stats = self.agent.learn(state, action, reward, next_state, terminated, truncated)
                wall_time_step = time.perf_counter() - start

                rows.append({
                    "agent_id": self.agent_id,
                    "seed": seed,
                    "step": global_step,
                    "episode": episode,
                    "step_in_episode": step_in_episode,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminated": terminated,
                    "truncated": truncated,
                    "wall_time_step": wall_time_step,
                    "wall_time_planning": stats.get("wall_time_planning", 0.0),
                    "n_planning_updates": stats.get("n_planning_updates", 0),
                })

                global_step += 1
                step_in_episode += 1
                state = next_state

                if terminated or truncated:
                    break

        trace = pd.DataFrame(rows, columns=TRACE_COLUMNS)
        self._persist(trace, seed)
        return trace

    def _persist(self, trace: pd.DataFrame, seed: int) -> None:
        agent_dir = self.output_dir / self.agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        trace_path = agent_dir / f"trace_seed_{seed}.csv"
        trace.to_csv(trace_path, index=False)

        config = {
            "agent_id": self.agent_id,
            "seed": seed,
            "hyperparams": self.hyperparams,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        config_path = agent_dir / f"config_seed_{seed}.json"
        config_path.write_text(json.dumps(config, indent=2))
