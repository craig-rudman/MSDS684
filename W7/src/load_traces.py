from pathlib import Path

import pandas as pd


def load_traces(*agent_dirs: Path | str) -> pd.DataFrame:
    frames = []
    for agent_dir in agent_dirs:
        agent_dir = Path(agent_dir)
        if not agent_dir.exists():
            raise FileNotFoundError(f"Agent directory does not exist: {agent_dir}")
        trace_files = sorted(agent_dir.glob("trace_seed_*.csv"))
        if not trace_files:
            raise FileNotFoundError(
                f"No trace_seed_*.csv files found in agent directory: {agent_dir}"
            )
        for trace_file in trace_files:
            frames.append(pd.read_csv(trace_file))
    return pd.concat(frames, ignore_index=True)
