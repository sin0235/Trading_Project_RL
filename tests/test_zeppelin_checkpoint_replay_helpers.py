import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.zeppelin_checkpoint_replay_helpers import (
    find_replay_start_t,
    select_replay_checkpoints,
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


if __name__ == "__main__":
    unittest.main()
