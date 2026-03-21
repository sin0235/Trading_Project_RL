import tempfile
import textwrap
import unittest
from pathlib import Path

from src.training.PPO import DEFAULT_CONFIG, load_ppo_config, resolve_ppo_config


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
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_ppo_config(config_path)

            self.assertEqual(cfg["fee_rate"], 0.002)
            self.assertEqual(cfg["total_timesteps"], 12345)
            self.assertEqual(cfg["n_eval_episodes"], 1)
            self.assertEqual(cfg["reward_name"], "tmp")

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


if __name__ == "__main__":
    unittest.main()
