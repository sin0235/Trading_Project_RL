import torch
import tempfile
import textwrap
import unittest
from pathlib import Path

from src.models.lstm import PPOLSTMActorCritic
from src.training.PPO import (
    DEFAULT_CONFIG,
    build_reward_kwargs_from_config,
    compute_rollout_steps_to_next_milestone,
    compute_periodic_trigger_interval,
    compute_learning_rate,
    get_results_root_candidates,
    infer_run_config_from_checkpoint,
    is_periodic_trigger_step,
    load_ppo_config,
    normalize_early_stop_baseline,
    normalize_checkpoint_milestones,
    next_periodic_trigger_step,
    next_checkpoint_milestone,
    resolve_eval_run_across_roots,
    resolve_eval_run,
    resolve_eval_checkpoint,
    load_run_config,
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
                    reward_excess_scale: 40.0
                    reward_turnover_scale: 0.5
                    early_stop_patience_evals: 5
                    early_stop_baseline: equal_weight
                    lr_schedule: linear
                    min_learning_rate: 0.00002
                    trade_deadband: 0.03
                    max_weight_change_per_step: 0.15
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_ppo_config(config_path)

            self.assertEqual(cfg["fee_rate"], 0.002)
            self.assertEqual(cfg["total_timesteps"], 12345)
            self.assertEqual(cfg["n_eval_episodes"], 1)
            self.assertEqual(cfg["reward_name"], "tmp")
            self.assertEqual(cfg["reward_excess_scale"], 40.0)
            self.assertEqual(cfg["reward_turnover_scale"], 0.5)
            self.assertEqual(cfg["early_stop_patience_evals"], 5)
            self.assertEqual(cfg["early_stop_baseline"], "equal_weight")
            self.assertEqual(cfg["lr_schedule"], "linear")
            self.assertEqual(cfg["min_learning_rate"], 0.00002)
            self.assertEqual(cfg["trade_deadband"], 0.03)
            self.assertEqual(cfg["max_weight_change_per_step"], 0.15)

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

    def test_resolve_ppo_config_validates_execution_filters(self):
        with self.assertRaisesRegex(ValueError, "reward_excess_scale"):
            resolve_ppo_config(config={"reward_excess_scale": -1.0})
        with self.assertRaisesRegex(ValueError, "early_stop_patience_evals"):
            resolve_ppo_config(config={"early_stop_patience_evals": -1})
        with self.assertRaisesRegex(ValueError, "early_stop_baseline"):
            resolve_ppo_config(config={"early_stop_baseline": "foobar"})
        with self.assertRaisesRegex(ValueError, "trade_deadband"):
            resolve_ppo_config(config={"trade_deadband": -0.01})
        with self.assertRaisesRegex(ValueError, "max_weight_change_per_step"):
            resolve_ppo_config(config={"max_weight_change_per_step": 0.0})
        with self.assertRaisesRegex(ValueError, "dirichlet_total_concentration"):
            resolve_ppo_config(config={"dirichlet_total_concentration": -1.0})

    def test_normalize_early_stop_baseline_supports_off_and_known_baselines(self):
        self.assertIsNone(normalize_early_stop_baseline("off"))
        self.assertIsNone(normalize_early_stop_baseline(None))
        self.assertEqual(normalize_early_stop_baseline("equal_weight"), "equal_weight")

    def test_build_reward_kwargs_from_config_uses_sharpe_scales(self):
        cfg = resolve_ppo_config(
            config={
                "reward_name": "sharpe",
                "reward_window": 45,
                "reward_sharpe_scale": 1.5,
                "reward_excess_scale": 35.0,
                "reward_drawdown_scale": 2.5,
                "reward_turnover_scale": 0.75,
            }
        )

        reward_kwargs = build_reward_kwargs_from_config(cfg)

        self.assertEqual(
            reward_kwargs,
            {
                "window": 45,
                "sharpe_scale": 1.5,
                "excess_scale": 35.0,
                "drawdown_scale": 2.5,
                "turnover_scale": 0.75,
            },
        )

    def test_resolve_ppo_config_normalizes_milestone_checkpoint_steps(self):
        cfg = resolve_ppo_config(
            config={
                "total_timesteps": 500,
                "milestone_checkpoint_steps": [100, "200", 100, 999, 300.0],
            }
        )

        self.assertEqual(cfg["milestone_checkpoint_steps"], [100, 200, 300])

    def test_normalize_checkpoint_milestones_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "số nguyên dương"):
            normalize_checkpoint_milestones([100.5], total_timesteps=1_000)
        with self.assertRaisesRegex(ValueError, "không hợp lệ"):
            normalize_checkpoint_milestones(["abc"], total_timesteps=1_000)

    def test_compute_rollout_steps_to_next_milestone_splits_early_rollout(self):
        milestones = [100, 1_000, 5_000]

        self.assertEqual(next_checkpoint_milestone(0, milestones), 100)
        self.assertEqual(
            compute_rollout_steps_to_next_milestone(
                current_step=0,
                total_timesteps=10_000,
                n_steps=2_048,
                milestone_steps=milestones,
                saved_steps=set(),
            ),
            100,
        )
        self.assertEqual(
            compute_rollout_steps_to_next_milestone(
                current_step=3_048,
                total_timesteps=10_000,
                n_steps=2_048,
                milestone_steps=milestones,
                saved_steps={100, 1_000},
            ),
            1_952,
        )

    def test_periodic_checkpoint_boundaries_remain_aligned_after_early_milestone(self):
        current_step = 0
        saved_steps = set()
        milestone_steps = [100]
        observed_boundaries = []

        for _ in range(6):
            rollout_steps = compute_rollout_steps_to_next_milestone(
                current_step=current_step,
                total_timesteps=500_000,
                n_steps=2_048,
                milestone_steps=milestone_steps,
                saved_steps=saved_steps,
                periodic_frequencies=[5],
            )
            current_step += rollout_steps
            if current_step in milestone_steps:
                saved_steps.add(current_step)
                observed_boundaries.append(current_step)
            if is_periodic_trigger_step(current_step, frequency=5, n_steps=2_048):
                observed_boundaries.append(current_step)

        self.assertEqual(current_step, 10_240)
        self.assertEqual(observed_boundaries, [100, 10_240])
        self.assertEqual(
            next_periodic_trigger_step(
                current_step=100,
                frequency=5,
                n_steps=2_048,
                total_timesteps=500_000,
            ),
            10_240,
        )
        self.assertEqual(compute_periodic_trigger_interval(5, 2_048), 10_240)

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

    def test_resolve_eval_checkpoint_can_read_legacy_checkpoint_directly_in_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "final_model.pt").write_bytes(b"x")

            path, source = resolve_eval_checkpoint(run_dir)

            self.assertEqual(path.name, "final_model.pt")
            self.assertEqual(source, "final_model")

    def test_load_run_config_uses_run_specific_model_shape_and_fills_new_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "run_id": "ppo_old",
                      "agent": "PPO_LSTM",
                      "hidden_size": 32,
                      "num_layers": 1,
                      "dropout": 0.0,
                      "window_size": 30,
                      "learning_rate": 0.0003,
                      "n_steps": 16,
                      "batch_size": 8,
                      "n_epochs": 1,
                      "total_timesteps": 32,
                      "device": "cpu"
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_run_config(run_dir, overrides={"device": "auto"})

            self.assertEqual(cfg["hidden_size"], 32)
            self.assertEqual(cfg["num_layers"], 1)
            self.assertEqual(cfg["device"], "auto")
            self.assertEqual(cfg["reward_name"], DEFAULT_CONFIG["reward_name"])

    def test_load_run_config_does_not_inherit_yaml_defaults_from_current_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text(
                textwrap.dedent(
                    """
                    {
                      "run_id": "ppo_old",
                      "agent": "PPO_LSTM",
                      "reward_name": "sharpe",
                      "hidden_size": 64,
                      "num_layers": 1,
                      "dropout": 0.0,
                      "window_size": 30,
                      "learning_rate": 0.0003,
                      "n_steps": 16,
                      "batch_size": 8,
                      "n_epochs": 1,
                      "total_timesteps": 32
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_run_config(run_dir)

            self.assertEqual(cfg["reward_name"], "sharpe")
            self.assertEqual(cfg["trade_deadband"], DEFAULT_CONFIG["trade_deadband"])
            self.assertEqual(
                cfg["max_weight_change_per_step"],
                DEFAULT_CONFIG["max_weight_change_per_step"],
            )

    def test_infer_run_config_from_checkpoint_recovers_model_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best_model.pt"
            model = PPOLSTMActorCritic(
                n_stocks=len(DEFAULT_CONFIG["tickers"]),
                n_features=len(DEFAULT_CONFIG["features"]),
                seq_len=DEFAULT_CONFIG["window_size"],
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
            )
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            cfg = infer_run_config_from_checkpoint(ckpt_path, base_config=DEFAULT_CONFIG, overrides={"device": "auto"})

            self.assertEqual(cfg["hidden_size"], 32)
            self.assertEqual(cfg["num_layers"], 1)
            self.assertEqual(cfg["device"], "auto")

    def test_resolve_eval_run_can_fallback_to_checkpoint_inference_when_config_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir)
            run_dir = results_root / "ppo_test_run"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            model = PPOLSTMActorCritic(
                n_stocks=len(DEFAULT_CONFIG["tickers"]),
                n_features=len(DEFAULT_CONFIG["features"]),
                seq_len=DEFAULT_CONFIG["window_size"],
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
            )
            torch.save({"model_state_dict": model.state_dict()}, ckpt_dir / "best_model.pt")

            resolved = resolve_eval_run(
                results_root,
                base_config=DEFAULT_CONFIG,
                overrides={"device": "auto"},
            )

            self.assertEqual(resolved["run_dir"], run_dir)
            self.assertEqual(resolved["config_source"], "checkpoint_inferred")
            self.assertEqual(resolved["config"]["hidden_size"], 32)

    def test_get_results_root_candidates_includes_repo_and_cwd_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "repo"
            cwd = Path(tmpdir) / "working"
            candidates = get_results_root_candidates(project_root=project_root, cwd=cwd)

            self.assertIn(project_root / "results" / "runs", candidates)
            self.assertIn(cwd / "results" / "runs", candidates)

    def test_resolve_eval_run_across_roots_can_find_legacy_cwd_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "repo"
            legacy_root = Path(tmpdir) / "working" / "results" / "runs"
            project_root.mkdir(parents=True)
            legacy_root.mkdir(parents=True)

            run_dir = legacy_root / "ppo_test_run"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            model = PPOLSTMActorCritic(
                n_stocks=len(DEFAULT_CONFIG["tickers"]),
                n_features=len(DEFAULT_CONFIG["features"]),
                seq_len=DEFAULT_CONFIG["window_size"],
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
            )
            torch.save({"model_state_dict": model.state_dict()}, ckpt_dir / "best_model.pt")

            resolved = resolve_eval_run_across_roots(
                [
                    project_root / "results" / "runs",
                    legacy_root,
                ],
                base_config=DEFAULT_CONFIG,
                overrides={"device": "auto"},
            )

            self.assertEqual(resolved["run_dir"], run_dir)
            self.assertEqual(resolved["results_root"], legacy_root)
            self.assertEqual(resolved["config_source"], "checkpoint_inferred")


if __name__ == "__main__":
    unittest.main()
