import tempfile
import textwrap
import unittest
from pathlib import Path

from src.training.PPO import (
    DEFAULT_CONFIG,
    compute_learning_rate,
    load_ppo_config,
    resolve_eval_checkpoint,
    resolve_ppo_config,
)


class PPOConfigLogicTests(unittest.TestCase):
    def test_load_ppo_config_reads_yaml_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ppo.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    fee_rate: 0.002
                    total_timesteps: 12345
                    n_eval_episodes: 1
                    reward_name: tmp
                    lr_schedule: linear
                    min_learning_rate: 0.00002
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_ppo_config(config_path)

            self.assertEqual(cfg["fee_rate"], 0.002)
            self.assertEqual(cfg["total_timesteps"], 12345)
            self.assertEqual(cfg["n_eval_episodes"], 1)
            self.assertEqual(cfg["reward_name"], "tmp")
            self.assertEqual(cfg["lr_schedule"], "linear")
            self.assertEqual(cfg["min_learning_rate"], 0.00002)

    def test_resolve_ppo_config_merge_priority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ppo.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    fee_rate: 0.002
                    total_timesteps: 12345
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = resolve_ppo_config(
                config={"total_timesteps": 22222},
                config_path=config_path,
            )

            self.assertEqual(cfg["fee_rate"], 0.002)
            self.assertEqual(cfg["total_timesteps"], 22222)
            self.assertEqual(cfg["batch_size"], DEFAULT_CONFIG["batch_size"])

    def test_load_ppo_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ppo.yaml"
            config_path.write_text("unknown_key: 1\n", encoding="utf-8")

            with self.assertRaisesRegex(KeyError, "unknown_key"):
                load_ppo_config(config_path)

    def test_resolve_ppo_config_maps_legacy_lr_decay_to_constant_schedule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "ppo.yaml"
            config_path.write_text("lr_decay: false\n", encoding="utf-8")

            cfg = resolve_ppo_config(config_path=config_path)

            self.assertEqual(cfg["lr_schedule"], "constant")

    def test_compute_learning_rate_linear_uses_floor(self):
        self.assertAlmostEqual(compute_learning_rate(1e-4, 2e-5, 0.0, "linear"), 1e-4)
        self.assertAlmostEqual(compute_learning_rate(1e-4, 2e-5, 0.5, "linear"), 6e-5)
        self.assertAlmostEqual(compute_learning_rate(1e-4, 2e-5, 1.0, "linear"), 2e-5)

    def test_compute_learning_rate_cosine_and_constant(self):
        self.assertAlmostEqual(compute_learning_rate(1e-4, 2e-5, 0.25, "cosine"), 8.82842712474619e-05)
        self.assertAlmostEqual(compute_learning_rate(1e-4, 2e-5, 0.75, "constant"), 1e-4)

    def test_resolve_eval_checkpoint_prefers_best_then_final_then_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            (ckpt_dir / "checkpoint_100.pt").write_bytes(b"x")
            (ckpt_dir / "checkpoint_20.pt").write_bytes(b"x")

            path, source = resolve_eval_checkpoint(ckpt_dir)
            self.assertEqual(path.name, "checkpoint_100.pt")
            self.assertEqual(source, "latest_checkpoint")

            (ckpt_dir / "final_model.pt").write_bytes(b"x")
            path, source = resolve_eval_checkpoint(ckpt_dir)
            self.assertEqual(path.name, "final_model.pt")
            self.assertEqual(source, "final_model")

            (ckpt_dir / "best_model.pt").write_bytes(b"x")
            path, source = resolve_eval_checkpoint(ckpt_dir)
            self.assertEqual(path.name, "best_model.pt")
            self.assertEqual(source, "best_model")


if __name__ == "__main__":
    unittest.main()
