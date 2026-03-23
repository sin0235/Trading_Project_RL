import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.zeppelin_checkpoint_replay_helpers import (
    find_replay_start_t,
    select_replay_checkpoints,
    select_replay_checkpoints_best_and_second,
)


class ZeppelinCheckpointReplayHelperTests(unittest.TestCase):
    def test_select_replay_checkpoints_keeps_progression_and_special_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "ppo_run"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            for step in (5120, 10240, 20480, 40960, 81920):
                (ckpt_dir / f"checkpoint_{step}.pt").write_bytes(b"x")
            (ckpt_dir / "best_model.pt").write_bytes(b"x")
            (ckpt_dir / "final_model.pt").write_bytes(b"x")

            checkpoints = select_replay_checkpoints(run_dir, numeric_count=3)

            self.assertEqual(
                [item["checkpoint_id"] for item in checkpoints],
                ["checkpoint_5120", "checkpoint_20480", "checkpoint_81920", "best_model", "final_model"],
            )

    def test_find_replay_start_t_keeps_window_buffer_before_display_start(self):
        dates = pd.date_range("2026-01-01", periods=90, freq="B")

        start_t = find_replay_start_t(
            dates=dates,
            display_start="2026-03-02",
            window_size=30,
        )

        self.assertEqual(start_t, 41)
        self.assertEqual(str(dates[start_t + 1].date()), "2026-03-02")

    def test_select_replay_checkpoints_best_and_second_prefers_best_then_second_numeric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "ppo_run"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "best_model.pt").write_bytes(b"x")
            for step in (1000, 2000, 3000):
                (ckpt_dir / f"checkpoint_{step}.pt").write_bytes(b"x")

            checkpoints = select_replay_checkpoints_best_and_second(run_dir)

            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(checkpoints[0]["checkpoint_id"], "best_model")
            self.assertEqual(checkpoints[1]["checkpoint_id"], "checkpoint_2000")

    def test_find_replay_start_t_respects_environment_start_cap(self):
        dates = pd.date_range("2026-01-01", periods=90, freq="B")

        start_t = find_replay_start_t(
            dates=dates,
            display_start="2026-03-02",
            window_size=30,
            max_start_t=29,
        )

        self.assertEqual(start_t, 29)


if __name__ == "__main__":
    unittest.main()
