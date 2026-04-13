from __future__ import annotations

import csv
import os


class EpisodeRecorder:
    """Records episode trajectories to a CSV file, one row per step."""

    def __init__(self, filepath: str, reset: bool = False) -> None:
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        is_new = reset or not os.path.exists(filepath)
        self._file = open(filepath, 'w' if reset else 'a', newline='')
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(['episode', 'step', 'position', 'velocity'])

    def record_step(self, episode: int, step: int, position: float, velocity: float) -> None:
        self._writer.writerow([episode,step,position,velocity])

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
