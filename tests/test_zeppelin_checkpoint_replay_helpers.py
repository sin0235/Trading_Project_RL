import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.zeppelin_checkpoint_replay_helpers import (
    _ensure_replay_fallback_root,
    _candidate_vnstock_sources,
    _build_recent_dataset,
    _drawdown_pct_series,
    _risk_summary,
    _rolling_quality_series,
    find_replay_start_t,
    select_replay_checkpoints,
    select_replay_checkpoints_best_and_second,
)

TEST_TMP_ROOT = (Path(__file__).resolve().parents[1] / ".pytest_tmp").resolve()
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class ZeppelinCheckpointReplayHelperTests(unittest.TestCase):
    def test_candidate_vnstock_sources_keeps_primary_then_online_fallbacks(self):
        self.assertEqual(_candidate_vnstock_sources("VCI"), ["VCI", "KBS", "MSN", "FMP"])
        self.assertEqual(_candidate_vnstock_sources("KBS"), ["KBS", "VCI", "MSN", "FMP"])

    def test_ensure_replay_fallback_root_seeds_from_first_available_data_root(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmpdir:
            project_root = Path(tmpdir) / "project-demo"
            processed_v2 = project_root / "data" / "processed_v2"
            processed_v2.mkdir(parents=True)
            (processed_v2 / "ACB.csv").write_text("time,open,high,low,close,volume\n2025-01-02,1,1,1,1,100\n", encoding="utf-8")

            fallback_root = _ensure_replay_fallback_root(project_root, [processed_v2])

            self.assertEqual(fallback_root, (project_root / "data" / "replay_fallback").resolve())
            self.assertTrue((fallback_root / "ACB.csv").exists())

    def test_select_replay_checkpoints_keeps_progression_and_special_models(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmpdir:
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
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmpdir:
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

    def test_drawdown_and_rolling_quality_series_align_with_values(self):
        values = [100.0, 110.0, 105.0, 120.0, 114.0]

        drawdown = _drawdown_pct_series(values, initial_balance=100.0)
        rolling_sharpe, rolling_sortino = _rolling_quality_series(values, initial_balance=100.0, window=3)

        self.assertEqual(len(drawdown), len(values))
        self.assertEqual(len(rolling_sharpe), len(values))
        self.assertEqual(len(rolling_sortino), len(values))
        self.assertAlmostEqual(drawdown[0], 0.0, places=4)
        self.assertLess(drawdown[2], 0.0)

    def test_risk_summary_exposes_core_dashboard_metrics(self):
        values = [100.0, 104.0, 101.0, 108.0, 112.0, 109.0]

        summary = _risk_summary(values, initial_balance=100.0)

        self.assertIn("sharpe_ratio", summary)
        self.assertIn("sortino_ratio", summary)
        self.assertIn("calmar_ratio", summary)
        self.assertIn("max_drawdown_pct", summary)
        self.assertIn("profit_factor", summary)
        self.assertIn("win_rate_pct", summary)

    def test_build_recent_dataset_rejects_local_source_for_demo(self):
        with self.assertRaisesRegex(RuntimeError, "Không còn fallback sang dữ liệu cục bộ"):
            _build_recent_dataset(
                project_root="d:/HCMUTE/RL/Project",
                tickers=["FPT"],
                features=["close"],
                window_size=60,
                data_roots=[],
                recent_months=12,
                warmup_months=4,
                vnstock_source="local",
                end_date="2026-02-28",
            )


if __name__ == "__main__":
    unittest.main()
