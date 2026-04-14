from __future__ import annotations

import csv
import os


class EpisodeRecorder:
    """Records episode trajectories to a CSV file, one row per step."""

    def __init__(self, filepath: str, reset: bool = False, experiment: str | None = None) -> None:
        self.filepath = filepath
        self.experiment = experiment
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        is_new = reset or not os.path.exists(filepath)
        self._file = open(filepath, 'w' if reset else 'a', newline='')
        self._writer = csv.writer(self._file)
        if is_new:
            header = ['episode', 'step', 'position', 'velocity']
            if experiment is not None:
                header = ['experiment'] + header
            self._writer.writerow(header)

    def record_step(self, episode: int, step: int, position: float, velocity: float) -> None:
        row = [episode, step, position, velocity]
        if self.experiment is not None:
            row = [self.experiment] + row
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
