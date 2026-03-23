import unittest
from pathlib import Path

from scripts.dashboard_paths import DashboardProjectPaths


class DashboardPathsTests(unittest.TestCase):
    def test_core_paths_are_resolved_from_project_root(self):
        paths = DashboardProjectPaths.from_project_root("D:/tmp/project-demo")

        self.assertEqual(paths.cache_dir, Path("D:/tmp/project-demo/.zeppelin_cache").resolve())
        self.assertEqual(paths.train_cache_path, Path("D:/tmp/project-demo/.zeppelin_cache/dashboard_train_payload.json").resolve())
        self.assertEqual(paths.replay_cache_path, Path("D:/tmp/project-demo/.zeppelin_cache/dashboard_replay_payload.json").resolve())
        self.assertEqual(paths.dashboard_cache_path, Path("D:/tmp/project-demo/.zeppelin_cache/dashboard_payload.json").resolve())
        self.assertEqual(paths.processed_data_root, Path("D:/tmp/project-demo/data/processed").resolve())
        self.assertEqual(paths.processed_v2_root, Path("D:/tmp/project-demo/data/processed_v2").resolve())

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


if __name__ == "__main__":
    unittest.main()
