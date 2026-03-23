"""
Training script cho PPO + LSTM trên TradingEnv.

Luồng chính:
    1. Load data, chia train/val/test
    2. Tạo env (train + val)
    3. Tạo model PPOLSTMActorCritic + PPOAgent
    4. Training loop: collect rollout -> PPO update -> eval -> checkpoint
    5. Final eval trên test set, lưu summary

Chạy:
    python -m src.training.PPO
"""

import os
import sys
import math
import json
import random
from collections.abc import Iterable
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.constants import TICKERS, FEATURES, WINDOW_SIZE, DATA_PATH
from src.models.lstm import PPOLSTMActorCritic
from src.agents.ppo_agent import PPOAgent
from src.environment.trading_env import TradingEnv
from src.utils.data_splitter import load_data, split_by_ratio
from src.utils.metrics import compute_all, format_report
from src.utils.logger import TrainingLogger, make_run_id


DEFAULT_CONFIG = {
    # --- Data ---
    "tickers": TICKERS,
    "features": FEATURES,
    "window_size": WINDOW_SIZE,
    "data_path": DATA_PATH,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # --- Environment ---
    "initial_balance": 1_000_000_000,
    "fee_rate": 0.001,
    "max_steps_train": 756,
    "max_steps_eval": 9999,
    "reward_scaling": 1.0,
    "reward_name": "sharpe",
    "reward_window": 30,
    "trade_deadband": 0.01,
    "max_weight_change_per_step": 0.15,

    # --- Model (LSTM) ---
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "log_std_init": -0.5,
    "dirichlet_total_concentration": 0.0,

    # --- PPO ---
    "learning_rate": 2e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.025,

    # --- Schedule ---
    "total_timesteps": 600_000,
    "lr_decay": True,
    "lr_schedule": "cosine",
    "min_learning_rate": 2e-5,
    "eval_freq": 5,
    "save_freq": 5,
    "milestone_checkpoint_steps": [100],
    "n_eval_episodes": 1,

    # --- Misc ---
    "seed": 42,
    "device": "auto",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "Conf" / "ppo_conf.yaml"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_config_path(config_path: str | os.PathLike | None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    path = Path(config_path)
    if path.is_absolute():
        return path

    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path

    project_path = (PROJECT_ROOT / path).resolve()
    return project_path


def load_ppo_config(config_path: str | os.PathLike | None = None) -> dict:
    path = _resolve_config_path(config_path)
    if not path.exists():
        if config_path is None:
            return {}
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Cần cài PyYAML để đọc file config PPO.") from exc

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config PPO phải là mapping/dict, nhận được: {type(cfg).__name__}")

    unknown_keys = sorted(set(cfg) - set(DEFAULT_CONFIG))
    if unknown_keys:
        raise KeyError(f"Config PPO chứa key không hợp lệ: {unknown_keys}")

    return cfg


def resolve_ppo_config(config: dict | None = None,
                       config_path: str | os.PathLike | None = None) -> dict:
    yaml_cfg = load_ppo_config(config_path)
    runtime_cfg = config or {}
    resolved = {**DEFAULT_CONFIG, **yaml_cfg, **runtime_cfg}

    # Tương thích ngược với config cũ chỉ có lr_decay.
    schedule_explicit = "lr_schedule" in yaml_cfg or "lr_schedule" in runtime_cfg
    legacy_decay_explicit = "lr_decay" in yaml_cfg or "lr_decay" in runtime_cfg
    if not schedule_explicit and legacy_decay_explicit:
        resolved["lr_schedule"] = "linear" if resolved["lr_decay"] else "constant"

    resolved["lr_schedule"] = str(resolved["lr_schedule"]).strip().lower()
    valid_schedules = {"constant", "linear", "cosine"}
    if resolved["lr_schedule"] not in valid_schedules:
        raise ValueError(
            f"lr_schedule không hợp lệ: {resolved['lr_schedule']}. "
            f"Hỗ trợ: {sorted(valid_schedules)}"
        )

    if resolved["learning_rate"] <= 0:
        raise ValueError("learning_rate phải > 0.")
    if resolved["min_learning_rate"] <= 0:
        raise ValueError("min_learning_rate phải > 0.")
    if resolved["min_learning_rate"] > resolved["learning_rate"]:
        raise ValueError("min_learning_rate không được lớn hơn learning_rate.")
    if resolved["trade_deadband"] < 0:
        raise ValueError("trade_deadband phải >= 0.")
    if not (0 < resolved["max_weight_change_per_step"] <= 1.0):
        raise ValueError("max_weight_change_per_step phải trong khoảng (0, 1].")
    if resolved["dirichlet_total_concentration"] < 0:
        raise ValueError("dirichlet_total_concentration phải >= 0.")
    if resolved["total_timesteps"] <= 0:
        raise ValueError("total_timesteps phải > 0.")
    resolved["milestone_checkpoint_steps"] = normalize_checkpoint_milestones(
        resolved.get("milestone_checkpoint_steps"),
        total_timesteps=resolved["total_timesteps"],
    )

    return resolved


def compute_learning_rate(
    base_lr: float,
    min_lr: float,
    progress: float,
    schedule: str,
) -> float:
    progress = min(max(float(progress), 0.0), 1.0)

    if schedule == "constant":
        return float(base_lr)

    if schedule == "linear":
        return float(min_lr + (base_lr - min_lr) * (1.0 - progress))

    if schedule == "cosine":
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr + (base_lr - min_lr) * cosine_decay)

    raise ValueError(f"Unsupported lr_schedule: {schedule}")


def normalize_checkpoint_milestones(
    raw_steps,
    total_timesteps: int | None = None,
) -> list[int]:
    if raw_steps is None:
        return []

    if isinstance(raw_steps, (str, bytes, int, float, np.integer, np.floating)):
        values = [raw_steps]
    elif isinstance(raw_steps, Iterable):
        values = list(raw_steps)
    else:
        raise ValueError("milestone_checkpoint_steps phải là số nguyên hoặc danh sách số nguyên.")

    normalized: list[int] = []
    max_steps = int(total_timesteps) if total_timesteps is not None else None

    for value in values:
        if isinstance(value, bool):
            raise ValueError("milestone_checkpoint_steps không được chứa giá trị boolean.")

        if isinstance(value, str):
            cleaned = value.strip().replace("_", "")
            if not cleaned:
                continue
            try:
                step = int(cleaned)
            except ValueError as exc:
                raise ValueError(
                    f"milestone_checkpoint_steps chứa giá trị không hợp lệ: {value!r}"
                ) from exc
        elif isinstance(value, (int, np.integer)):
            step = int(value)
        elif isinstance(value, (float, np.floating)):
            if not float(value).is_integer():
                raise ValueError(
                    f"milestone_checkpoint_steps phải là số nguyên dương, nhận được {value!r}"
                )
            step = int(value)
        else:
            raise ValueError(
                f"milestone_checkpoint_steps chứa kiểu không hỗ trợ: {type(value).__name__}"
            )

        if step <= 0:
            raise ValueError("milestone_checkpoint_steps phải chứa các mốc > 0.")
        if max_steps is not None and step > max_steps:
            continue
        normalized.append(step)

    return sorted(set(normalized))


def next_checkpoint_milestone(
    current_step: int,
    milestone_steps: list[int],
    saved_steps: set[int] | None = None,
) -> int | None:
    saved = saved_steps or set()
    for milestone in milestone_steps:
        if milestone > current_step and milestone not in saved:
            return milestone
    return None


def compute_rollout_steps_to_next_milestone(
    current_step: int,
    total_timesteps: int,
    n_steps: int,
    milestone_steps: list[int],
    saved_steps: set[int] | None = None,
    periodic_frequencies: Iterable[int] | None = None,
) -> int:
    remaining_steps = int(total_timesteps) - int(current_step)
    if remaining_steps <= 0:
        return 0

    rollout_steps = min(int(n_steps), remaining_steps)
    stop_candidates: list[int] = []

    milestone = next_checkpoint_milestone(
        current_step=int(current_step),
        milestone_steps=milestone_steps,
        saved_steps=saved_steps,
    )
    if milestone is not None:
        stop_candidates.append(int(milestone))

    for frequency in periodic_frequencies or ():
        boundary = next_periodic_trigger_step(
            current_step=int(current_step),
            frequency=int(frequency),
            n_steps=int(n_steps),
            total_timesteps=int(total_timesteps),
        )
        if boundary is not None:
            stop_candidates.append(boundary)

    if not stop_candidates:
        return rollout_steps

    next_stop_step = min(stop_candidates)
    steps_until_stop = int(next_stop_step) - int(current_step)
    if steps_until_stop <= 0:
        return rollout_steps

    return min(rollout_steps, steps_until_stop)


def compute_periodic_trigger_interval(frequency: int, n_steps: int) -> int:
    frequency = int(frequency)
    n_steps = int(n_steps)

    if frequency <= 0:
        raise ValueError("Periodic frequency phải > 0.")
    if n_steps <= 0:
        raise ValueError("n_steps phải > 0.")

    return frequency * n_steps


def next_periodic_trigger_step(
    current_step: int,
    frequency: int,
    n_steps: int,
    total_timesteps: int | None = None,
) -> int | None:
    interval = compute_periodic_trigger_interval(frequency=frequency, n_steps=n_steps)
    next_step = ((int(current_step) // interval) + 1) * interval
    if total_timesteps is not None and next_step > int(total_timesteps):
        return None
    return next_step


def is_periodic_trigger_step(step: int, frequency: int, n_steps: int) -> bool:
    interval = compute_periodic_trigger_interval(frequency=frequency, n_steps=n_steps)
    step = int(step)
    return step > 0 and step % interval == 0


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("checkpoint_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _candidate_checkpoint_dirs(path: str | os.PathLike | Path) -> list[Path]:
    path = Path(path)
    if path.name == "checkpoints":
        return [path]

    candidates = [path / "checkpoints", path]
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_eval_checkpoint(path: str | os.PathLike | Path) -> tuple[Path | None, str]:
    checkpoint_dirs = _candidate_checkpoint_dirs(path)
    if all(not ckpt_dir.exists() for ckpt_dir in checkpoint_dirs):
        return None, "missing_dir"

    for ckpt_dir in checkpoint_dirs:
        if not ckpt_dir.exists():
            continue

        best_path = ckpt_dir / "best_model.pt"
        if best_path.exists():
            return best_path, "best_model"

        final_path = ckpt_dir / "final_model.pt"
        if final_path.exists():
            return final_path, "final_model"

        checkpoint_paths = sorted(
            ckpt_dir.glob("checkpoint_*.pt"),
            key=lambda candidate: (_checkpoint_step(candidate), candidate.stat().st_mtime),
        )
        if checkpoint_paths:
            return checkpoint_paths[-1], "latest_checkpoint"

    return None, "no_checkpoint"


def load_run_config(
    run_dir: str | os.PathLike | Path,
    overrides: dict | None = None,
) -> dict:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config.json trong run: {run_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f) or {}

    if not isinstance(raw_cfg, dict):
        raise ValueError(f"config.json của run phải là dict, nhận được: {type(raw_cfg).__name__}")

    # Khi load config của một run đã train, ưu tiên tuyệt đối config đã được lưu
    # cùng DEFAULT_CONFIG trong code để lấp đầy các key mới.
    # Không để YAML hiện tại của project vô tình ghi đè metadata run cũ.
    run_cfg = {**DEFAULT_CONFIG, **{k: v for k, v in raw_cfg.items() if k in DEFAULT_CONFIG}}
    if overrides:
        run_cfg.update(overrides)

    return resolve_ppo_config(config=run_cfg)


def infer_run_config_from_checkpoint(
    ckpt_path: str | os.PathLike | Path,
    base_config: dict,
    overrides: dict | None = None,
) -> dict:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint không chứa model_state_dict hợp lệ: {ckpt_path}")

    ih_key = "feature_extractor.lstm.weight_ih_l0"
    hh_key = "feature_extractor.lstm.weight_hh_l0"
    actor_out_key = "actor_head.5.bias"
    if ih_key not in state_dict or hh_key not in state_dict or actor_out_key not in state_dict:
        raise ValueError(f"Checkpoint không đúng định dạng PPO-LSTM mong đợi: {ckpt_path}")

    hidden_size = int(state_dict[hh_key].shape[1])
    num_layers = len([k for k in state_dict if k.startswith("feature_extractor.lstm.weight_hh_l")])
    n_assets = int(state_dict[actor_out_key].shape[0])
    n_stocks = n_assets - 1
    input_size = int(state_dict[ih_key].shape[1])

    if n_stocks <= 0:
        raise ValueError(f"Checkpoint có n_stocks không hợp lệ: {ckpt_path}")
    if input_size % n_stocks != 0:
        raise ValueError(
            f"Không suy luận được số features từ checkpoint {ckpt_path}: "
            f"input_size={input_size}, n_stocks={n_stocks}"
        )

    inferred_n_features = input_size // n_stocks
    tickers = list(base_config.get("tickers", []))
    features = list(base_config.get("features", []))
    if len(tickers) != n_stocks:
        raise ValueError(
            f"Checkpoint cần {n_stocks} tickers nhưng base_config có {len(tickers)}. "
            "Không thể dựng env/model tương thích."
        )
    if len(features) != inferred_n_features:
        raise ValueError(
            f"Checkpoint cần {inferred_n_features} features nhưng base_config có {len(features)}. "
            "Không thể dựng env/model tương thích."
        )

    inferred_cfg = {
        **base_config,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }
    if overrides:
        inferred_cfg.update(overrides)

    return resolve_ppo_config(config=inferred_cfg)


def resolve_eval_run(
    results_root: str | os.PathLike | Path,
    base_config: dict | None = None,
    overrides: dict | None = None,
) -> dict:
    results_root = Path(results_root)
    if not results_root.exists():
        return {
            "run_dir": None,
            "ckpt_path": None,
            "ckpt_source": "missing_results_root",
            "config": None,
            "config_source": "none",
            "skipped_runs": [],
        }

    skipped_runs = []
    runs = sorted([p for p in results_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in runs:
        ckpt_path, ckpt_source = resolve_eval_checkpoint(run_dir)
        if ckpt_path is None:
            skipped_runs.append({"run_id": run_dir.name, "reason": ckpt_source})
            continue

        try:
            cfg = load_run_config(run_dir, overrides=overrides)
            config_source = "run_config"
        except FileNotFoundError:
            if base_config is None:
                skipped_runs.append({"run_id": run_dir.name, "reason": "missing_config"})
                continue
            try:
                cfg = infer_run_config_from_checkpoint(ckpt_path, base_config=base_config, overrides=overrides)
                config_source = "checkpoint_inferred"
            except Exception as exc:
                skipped_runs.append({"run_id": run_dir.name, "reason": f"infer_failed: {exc}"})
                continue
        except Exception as exc:
            skipped_runs.append({"run_id": run_dir.name, "reason": f"config_failed: {exc}"})
            continue

        return {
            "run_dir": run_dir,
            "ckpt_path": ckpt_path,
            "ckpt_source": ckpt_source,
            "config": cfg,
            "config_source": config_source,
            "skipped_runs": skipped_runs,
        }

    return {
        "run_dir": None,
        "ckpt_path": None,
        "ckpt_source": "no_compatible_run",
        "config": None,
        "config_source": "none",
        "skipped_runs": skipped_runs,
    }


def get_results_root_candidates(
    project_root: str | os.PathLike | Path | None = None,
    cwd: str | os.PathLike | Path | None = None,
) -> list[Path]:
    project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
    cwd = Path(cwd) if cwd is not None else Path.cwd()

    candidates = [
        project_root / "results" / "runs",
        cwd / "results" / "runs",
        cwd.parent / "results" / "runs",
        project_root.parent / "results" / "runs",
        Path("/kaggle/working/repo/results/runs"),
        Path("/kaggle/working/results/runs"),
        Path("/kaggle/results/runs"),
    ]

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)

    return deduped


def resolve_eval_run_across_roots(
    results_roots: list[str | os.PathLike | Path],
    base_config: dict | None = None,
    overrides: dict | None = None,
) -> dict:
    checked_results_roots = []
    missing_results_roots = []
    all_skipped_runs = []
    candidates = []

    for results_root in results_roots:
        root = Path(results_root)
        checked_results_roots.append(root)
        resolved = resolve_eval_run(root, base_config=base_config, overrides=overrides)

        for item in resolved["skipped_runs"]:
            all_skipped_runs.append({"results_root": str(root), **item})

        if resolved["run_dir"] is None:
            if resolved["ckpt_source"] == "missing_results_root":
                missing_results_roots.append(root)
            continue

        candidates.append({
            **resolved,
            "results_root": root,
        })

    if candidates:
        best = max(candidates, key=lambda item: item["run_dir"].stat().st_mtime)
        best["checked_results_roots"] = checked_results_roots
        best["missing_results_roots"] = missing_results_roots
        best["all_skipped_runs"] = all_skipped_runs
        return best

    return {
        "run_dir": None,
        "ckpt_path": None,
        "ckpt_source": "no_compatible_run",
        "config": None,
        "config_source": "none",
        "results_root": None,
        "checked_results_roots": checked_results_roots,
        "missing_results_roots": missing_results_roots,
        "skipped_runs": [],
        "all_skipped_runs": all_skipped_runs,
    }


def make_env(tickers, data_dict, config, for_eval=False):
    return TradingEnv(
        tickers=tickers,
        mode="continuous",
        initial_balance=config["initial_balance"],
        fee_rate=config["fee_rate"],
        window_size=config["window_size"],
        data_dict=data_dict,
        features=config["features"],
        max_steps=config["max_steps_eval"] if for_eval else config["max_steps_train"],
        random_start=not for_eval,
        reward_scaling=config["reward_scaling"],
        reward_name=config["reward_name"],
        reward_kwargs={"window": config["reward_window"]},
        trade_deadband=config["trade_deadband"],
        max_weight_change_per_step=config["max_weight_change_per_step"],
        print_verbosity=999999,
    )


def average_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}

    avg_metrics = {}
    keys = set().union(*(m.keys() for m in metrics_list))
    for key in keys:
        vals = [m[key] for m in metrics_list if key in m]
        if vals:
            avg_metrics[key] = float(np.mean(vals))
    return avg_metrics


def run_rule_based_episode(env: TradingEnv, policy_fn) -> list[float]:
    obs, _ = env.reset()
    done = False
    values = [env.portfolio_value]
    step_idx = 0

    while not done:
        action = np.asarray(policy_fn(step_idx, obs, env), dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        values.append(env.portfolio_value)
        step_idx += 1

    return values


def equal_weight_baseline_values(env: TradingEnv) -> list[float]:
    target = np.full(env.n_stocks + 1, 1.0 / (env.n_stocks + 1), dtype=np.float32)
    return run_rule_based_episode(env, lambda _step_idx, _obs, _env: target)


def buy_and_hold_baseline_values(env: TradingEnv) -> list[float]:
    initial_target = np.full(env.n_stocks + 1, 1.0 / (env.n_stocks + 1), dtype=np.float32)

    def policy_fn(step_idx, _obs, rollout_env: TradingEnv):
        if step_idx == 0:
            return initial_target

        # Dung ty trong tai gia mo session ke tiep de giu nguyen holdings, tranh tai can bang gia tao.
        next_trade_prices = rollout_env.get_trade_prices()
        return rollout_env.state_space.get_portfolio_state(
            rollout_env.cash, rollout_env.holdings, next_trade_prices
        ).astype(np.float32)

    return run_rule_based_episode(env, policy_fn)


def evaluate_baselines(env: TradingEnv, initial_balance: float) -> dict[str, dict]:
    baselines = {
        "equal_weight": compute_all(equal_weight_baseline_values(env), initial_balance),
        "buy_and_hold_equal_weight": compute_all(buy_and_hold_baseline_values(env), initial_balance),
    }
    return baselines


def build_baseline_comparison(model_metrics: dict, baseline_metrics: dict) -> dict:
    return {
        "baseline_metrics": baseline_metrics,
        "delta_total_return": float(
            model_metrics.get("total_return", 0.0) - baseline_metrics.get("total_return", 0.0)
        ),
        "delta_sharpe_ratio": float(
            model_metrics.get("sharpe_ratio", 0.0) - baseline_metrics.get("sharpe_ratio", 0.0)
        ),
        "delta_max_drawdown": float(
            model_metrics.get("max_drawdown", 0.0) - baseline_metrics.get("max_drawdown", 0.0)
        ),
    }


def format_baseline_comparison(label: str, model_metrics: dict, baseline_metrics: dict) -> str:
    delta_return = model_metrics.get("total_return", 0.0) - baseline_metrics.get("total_return", 0.0)
    delta_sharpe = model_metrics.get("sharpe_ratio", 0.0) - baseline_metrics.get("sharpe_ratio", 0.0)
    return (
        f"[BASELINE/{label}] "
        f"baseline_return={baseline_metrics.get('total_return', 0):.2%} | "
        f"baseline_sharpe={baseline_metrics.get('sharpe_ratio', 0):.4f} | "
        f"delta_return={delta_return:.2%} | "
        f"delta_sharpe={delta_sharpe:.4f}"
    )


def train_ppo(config: dict = None, config_path: str | os.PathLike | None = None):
    cfg = resolve_ppo_config(config=config, config_path=config_path)
    set_seed(cfg["seed"])

    # ----------------------------------------------------------------
    # 1. Load & split data
    # ----------------------------------------------------------------
    data_dict = load_data(tickers=cfg["tickers"], data_path=cfg["data_path"])
    split = split_by_ratio(
        data_dict,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
    )
    print(f"Data split: {split.summary()}")

    # ----------------------------------------------------------------
    # 2. Environments
    # ----------------------------------------------------------------
    train_env = make_env(cfg["tickers"], split.train, cfg, for_eval=False)
    val_env   = make_env(cfg["tickers"], split.val,   cfg, for_eval=True)
    test_env  = make_env(cfg["tickers"], split.test,  cfg, for_eval=True)

    state_space = train_env.state_space
    n_stocks  = state_space.n_stocks
    n_features = state_space.n_features

    # ----------------------------------------------------------------
    # 3. Model + Agent
    # ----------------------------------------------------------------
    model = PPOLSTMActorCritic(
        n_stocks=n_stocks,
        n_features=n_features,
        seq_len=cfg["window_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        log_std_init=cfg["log_std_init"],
        dirichlet_total_concentration=cfg["dirichlet_total_concentration"],
    )

    agent = PPOAgent(
        model=model,
        lr=cfg["learning_rate"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg["target_kl"],
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        device=cfg["device"],
    )

    print(f"Device: {agent.device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # ----------------------------------------------------------------
    # 4. Logger
    # ----------------------------------------------------------------
    run_id = make_run_id("ppo")
    logger = TrainingLogger(
        run_id=run_id,
        agent="PPO_LSTM",
        config=cfg,
        results_dir=str(PROJECT_ROOT / "results" / "runs"),
    )
    logger.info(f"Data split: {split.summary()}")
    logger.info(f"Device: {agent.device} | Params: {total_params:,}")
    logger.info(
        f"Reward function: {cfg['reward_name']} | "
        f"reward_window={cfg['reward_window']} | reward_scaling={cfg['reward_scaling']}"
    )
    logger.info(
        f"Execution filters: trade_deadband={cfg['trade_deadband']:.3f} | "
        f"max_weight_change_per_step={cfg['max_weight_change_per_step']:.3f}"
    )
    logger.info(
        f"LR schedule: {cfg['lr_schedule']} | "
        f"base_lr={cfg['learning_rate']:.2e} | min_lr={cfg['min_learning_rate']:.2e}"
    )
    logger.info(
        "Policy exploration: "
        + (
            f"fixed_dirichlet_total_concentration={cfg['dirichlet_total_concentration']:.2f}"
            if cfg["dirichlet_total_concentration"] > 0
            else "legacy_free_dirichlet_concentration"
        )
        + f" | ent_coef={cfg['ent_coef']:.4f}"
    )
    if cfg["milestone_checkpoint_steps"]:
        logger.info(f"Milestone checkpoints: {cfg['milestone_checkpoint_steps']}")

    # ----------------------------------------------------------------
    # 5. Training loop
    # ----------------------------------------------------------------
    episode_counter = 0
    best_val_sharpe = -np.inf
    obs = None
    latest_val_baselines = None
    rollout = 0

    save_dir = logger.get_run_dir() / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    train_reset_seed = cfg["seed"]
    saved_checkpoint_steps: set[int] = set()

    def save_step_checkpoint(step: int, reason: str) -> bool:
        step = int(step)
        if step <= 0 or step in saved_checkpoint_steps:
            return False
        agent.save(str(save_dir / f"checkpoint_{step}.pt"))
        saved_checkpoint_steps.add(step)
        logger.info(f"Saved checkpoint_{step}.pt ({reason})")
        return True

    while agent.total_steps < cfg["total_timesteps"]:
        rollout += 1
        rollout_steps = compute_rollout_steps_to_next_milestone(
            current_step=agent.total_steps,
            total_timesteps=cfg["total_timesteps"],
            n_steps=cfg["n_steps"],
            milestone_steps=cfg["milestone_checkpoint_steps"],
            saved_steps=saved_checkpoint_steps,
            periodic_frequencies=(cfg["eval_freq"], cfg["save_freq"]),
        )
        if rollout_steps <= 0:
            break

        progress = agent.total_steps / cfg["total_timesteps"]
        agent.set_lr(
            compute_learning_rate(
                base_lr=cfg["learning_rate"],
                min_lr=cfg["min_learning_rate"],
                progress=progress,
                schedule=cfg["lr_schedule"],
            )
        )

        # Collect
        obs, ep_infos = agent.collect_rollout(
            train_env,
            state_space,
            rollout_steps,
            obs,
            reset_seed=train_reset_seed,
        )
        train_reset_seed = None
        rollout_diag = dict(agent.last_rollout_stats or {})

        # Update
        update_stats = agent.update()

        # Log episodes
        for ep in ep_infos:
            episode_counter += 1
            logger.log_episode(
                episode=episode_counter,
                total_reward=ep["total_reward"],
                portfolio_value=ep["portfolio_value"],
                total_return=ep["total_return"],
                n_trades=ep["n_trades"],
                total_cost=ep["total_cost"],
                steps=ep["steps"],
                extra={
                    "avg_turnover": ep.get("avg_turnover", 0.0),
                    "steps_with_trades": ep.get("steps_with_trades", 0),
                    "steps_with_trades_pct": ep.get("steps_with_trades_pct", 0.0),
                    "avg_concentration_sum": ep.get("avg_concentration_sum", 0.0),
                },
            )

        # Log update
        logger.log_train_step(
            step=agent.total_steps,
            policy_loss=update_stats["policy_loss"],
            value_loss=update_stats["value_loss"],
            entropy=update_stats["entropy"],
            approx_kl=update_stats["approx_kl"],
            clip_fraction=update_stats["clip_fraction"],
            learning_rate=agent.get_lr(),
            extra=rollout_diag,
        )

        # Periodic info
        if rollout % 5 == 0 or rollout == 1:
            logger.info(
                f"[Rollout {rollout}] "
                f"steps={agent.total_steps:,} | "
                f"pi_loss={update_stats['policy_loss']:.5f} | "
                f"v_loss={update_stats['value_loss']:.5f} | "
                f"kl={update_stats['approx_kl']:.5f} | "
                f"grad={update_stats.get('grad_norm', 0):.4f} | "
                f"lr={agent.get_lr():.2e} | "
                f"turnover={rollout_diag.get('avg_turnover', 0.0):.2%} | "
                f"trade_steps={int(rollout_diag.get('steps_with_trades', 0))}/{rollout_steps} | "
                f"conc_sum={rollout_diag.get('avg_concentration_sum', 0.0):.2f}"
            )

        # Periodic eval on val set
        if is_periodic_trigger_step(agent.total_steps, cfg["eval_freq"], cfg["n_steps"]):
            val_values = agent.evaluate(val_env, val_env.state_space, cfg["n_eval_episodes"])
            val_diag = dict(agent.last_eval_stats or {})
            val_metrics_list = [compute_all(pv, cfg["initial_balance"]) for pv in val_values]
            avg_val_metrics = average_metrics(val_metrics_list)
            val_baselines = evaluate_baselines(val_env, cfg["initial_balance"])
            latest_val_baselines = {
                name: build_baseline_comparison(avg_val_metrics, metrics)
                for name, metrics in val_baselines.items()
            }
            logger.log_eval(episode=episode_counter, metrics=avg_val_metrics, split="val", extra=val_diag)
            logger.info(f"\n{format_report(avg_val_metrics)}")
            logger.info(
                "[DIAG/VAL] "
                f"avg_turnover={val_diag.get('avg_turnover', 0.0):.2%} | "
                f"steps_with_trades={val_diag.get('steps_with_trades', 0):.1f} | "
                f"steps_with_trades_pct={val_diag.get('steps_with_trades_pct', 0.0):.2%} | "
                f"avg_concentration_sum={val_diag.get('avg_concentration_sum', 0.0):.2f}"
            )
            for name, metrics in val_baselines.items():
                logger.info(format_baseline_comparison(name.upper(), avg_val_metrics, metrics))

            if avg_val_metrics.get("sharpe_ratio", -np.inf) > best_val_sharpe:
                best_val_sharpe = avg_val_metrics["sharpe_ratio"]
                agent.save(str(save_dir / "best_model.pt"))
                logger.info(f"Best val Sharpe: {best_val_sharpe:.4f} -> saved best_model.pt")

        if agent.total_steps in cfg["milestone_checkpoint_steps"]:
            save_step_checkpoint(agent.total_steps, "milestone_checkpoint")

        # Periodic checkpoint
        if is_periodic_trigger_step(agent.total_steps, cfg["save_freq"], cfg["n_steps"]):
            save_step_checkpoint(agent.total_steps, "periodic_checkpoint")

    # ----------------------------------------------------------------
    # 6. Final eval on test set (load best model)
    # ----------------------------------------------------------------
    best_path = save_dir / "best_model.pt"
    if best_path.exists():
        agent.load(str(best_path))
        logger.info("Loaded best_model.pt for final test evaluation")
    else:
        logger.info("best_model.pt chưa tồn tại, dùng model hiện tại ở cuối training để final eval")

    test_values = agent.evaluate(test_env, test_env.state_space, n_episodes=cfg["n_eval_episodes"])
    test_diag = dict(agent.last_eval_stats or {})
    test_metrics_list = [compute_all(pv, cfg["initial_balance"]) for pv in test_values]
    avg_test = average_metrics(test_metrics_list)
    test_baselines = evaluate_baselines(test_env, cfg["initial_balance"])

    logger.log_eval(episode=episode_counter, metrics=avg_test, split="test", extra=test_diag)
    logger.info(f"\n=== FINAL TEST RESULTS (avg {len(test_values)} episodes) ===")
    logger.info(f"\n{format_report(avg_test)}")
    logger.info(
        "[DIAG/TEST] "
        f"avg_turnover={test_diag.get('avg_turnover', 0.0):.2%} | "
        f"steps_with_trades={test_diag.get('steps_with_trades', 0):.1f} | "
        f"steps_with_trades_pct={test_diag.get('steps_with_trades_pct', 0.0):.2%} | "
        f"avg_concentration_sum={test_diag.get('avg_concentration_sum', 0.0):.2f}"
    )
    for name, metrics in test_baselines.items():
        logger.info(format_baseline_comparison(name.upper(), avg_test, metrics))

    # ----------------------------------------------------------------
    # 7. Summary
    # ----------------------------------------------------------------
    agent.save(str(save_dir / "final_model.pt"))
    logger.save_summary(
        metrics=avg_test,
        extra={
            "data_split": split.summary(),
            "total_episodes": episode_counter,
            "best_val_sharpe": best_val_sharpe,
            "baseline_comparisons": {
                "val": latest_val_baselines,
                "test": {
                    name: build_baseline_comparison(avg_test, metrics)
                    for name, metrics in test_baselines.items()
                },
            },
            "diagnostics": {
                "last_rollout": rollout_diag,
                "test": test_diag,
            },
        },
    )

    return agent, avg_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO-LSTM trading agent.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path tới file YAML config. Mặc định dùng Conf/ppo_conf.yaml nếu tồn tại.",
    )
    args = parser.parse_args()

    train_ppo(config_path=args.config)
