import csv
import os
import pytest

from src.episode_recorder import EpisodeRecorder


@pytest.fixture
def csv_path(tmp_path):
    return str(tmp_path / "trajectories.csv")


class TestEpisodeRecorderConstruction:
    def test_creates_file_on_construction(self, csv_path):
        rec = EpisodeRecorder(csv_path)
        rec.close()
        assert os.path.exists(csv_path)

    def test_stores_filepath(self, csv_path):
        rec = EpisodeRecorder(csv_path)
        rec.close()
        assert rec.filepath == csv_path

    def test_writes_header_row(self, csv_path):
        rec = EpisodeRecorder(csv_path)
        rec.close()
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["episode", "step", "position", "velocity"]


class TestEpisodeRecorderRecordStep:
    def test_writes_one_row_per_step(self, csv_path):
        rec = EpisodeRecorder(csv_path)
        rec.record_step(1, 0, -0.5, 0.0)
        rec.record_step(1, 1, -0.48, 0.01)
        rec.close()
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 steps

    def test_row_values_are_correct(self, csv_path):
        rec = EpisodeRecorder(csv_path)
        rec.record_step(2, 5, -0.3, 0.02)
        rec.close()
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert rows[1] == ["2", "5", "-0.3", "0.02"]


class TestEpisodeRecorderAppend:
    def test_appends_to_existing_file(self, csv_path):
        with EpisodeRecorder(csv_path) as rec:
            rec.record_step(1, 0, -0.5, 0.0)
        with EpisodeRecorder(csv_path) as rec:
            rec.record_step(2, 0, -0.4, 0.01)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 steps

    def test_header_written_only_once(self, csv_path):
        with EpisodeRecorder(csv_path) as rec:
            rec.record_step(1, 0, -0.5, 0.0)
        with EpisodeRecorder(csv_path) as rec:
            rec.record_step(2, 0, -0.4, 0.01)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["episode", "step", "position", "velocity"]
        assert rows[1][0] != "episode"


class TestEpisodeRecorderReset:
    def test_reset_overwrites_existing_file(self, csv_path):
        with EpisodeRecorder(csv_path) as rec:
            rec.record_step(1, 0, -0.5, 0.0)
        with EpisodeRecorder(csv_path, reset=True) as rec:
            rec.record_step(2, 0, -0.4, 0.01)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 step (not 3)

    def test_reset_writes_fresh_header(self, csv_path):
        with EpisodeRecorder(csv_path) as rec:
            rec.record_step(1, 0, -0.5, 0.0)
        with EpisodeRecorder(csv_path, reset=True) as rec:
            pass
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["episode", "step", "position", "velocity"]
        assert len(rows) == 1


class TestEpisodeRecorderContextManager:
    def test_context_manager_closes_file(self, csv_path):
        with EpisodeRecorder(csv_path) as rec:
            rec.record_step(1, 0, -0.5, 0.0)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 step

    def test_context_manager_returns_recorder(self, csv_path):
        with EpisodeRecorder(csv_path) as rec:
            assert isinstance(rec, EpisodeRecorder)
