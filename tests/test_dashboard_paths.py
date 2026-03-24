import unittest
import tempfile
from pathlib import Path

from scripts.dashboard_paths import DashboardProjectPaths

TEST_TMP_ROOT = (Path(__file__).resolve().parents[1] / ".pytest_tmp").resolve()
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class DashboardPathsTests(unittest.TestCase):
    def test_core_paths_are_resolved_from_project_root(self):
        paths = DashboardProjectPaths.from_project_root("D:/tmp/project-demo")

        self.assertEqual(paths.cache_dir, Path("D:/tmp/project-demo/.zeppelin_cache").resolve())
        self.assertEqual(paths.train_cache_path, Path("D:/tmp/project-demo/.zeppelin_cache/dashboard_train_payload.json").resolve())
        self.assertEqual(paths.replay_cache_path, Path("D:/tmp/project-demo/.zeppelin_cache/dashboard_replay_payload.json").resolve())
        self.assertEqual(paths.dashboard_cache_path, Path("D:/tmp/project-demo/.zeppelin_cache/dashboard_payload.json").resolve())
        self.assertEqual(paths.processed_data_root, Path("D:/tmp/project-demo/data/processed").resolve())
        self.assertEqual(paths.processed_v2_root, Path("D:/tmp/project-demo/data/processed_v2").resolve())
        self.assertEqual(paths.replay_fallback_root, Path("D:/tmp/project-demo/data/replay_fallback").resolve())

    def test_ddq_compare_candidates_follow_fixed_structure(self):
        paths = DashboardProjectPaths.from_project_root("D:/tmp/project-demo")

        self.assertEqual(
            paths.ddq_checkpoint_candidates,
            [
                Path("D:/tmp/project-demo/results/compare/ddq_best/checkpoints/best_model.pt").resolve(),
                Path("D:/tmp/project-demo/results/compare/ddq_best/best_model.pt").resolve(),
                Path("D:/tmp/project-demo/results/compare/ddq_best/checkpoints/final_model.pt").resolve(),
                Path("D:/tmp/project-demo/results/compare/ddq_best/final_model.pt").resolve(),
            ],
        )

    def test_ddq_compare_candidates_include_nested_run_directory(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as tmpdir:
            project_root = Path(tmpdir) / "project-demo"
            nested_run = project_root / "results" / "compare" / "ddq_best" / "ddq_20260323_100158"
            (nested_run / "checkpoints").mkdir(parents=True)

            paths = DashboardProjectPaths.from_project_root(project_root)

            self.assertIn(nested_run.resolve(), paths.ddq_compare_artifact_roots)
            self.assertIn(
                (nested_run / "checkpoints" / "best_model.pt").resolve(),
                paths.ddq_checkpoint_candidates,
            )
            self.assertIn(
                (nested_run / "config.json").resolve(),
                paths.ddq_config_json_candidates,
            )


if __name__ == "__main__":
    unittest.main()
