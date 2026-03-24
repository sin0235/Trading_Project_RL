"""
Training script cho Branching Double DQN (DDQ) + Branching DRQN (LSTM) tren TradingEnv (MultiDiscrete).

Luồng chính:
    1. Load data, chia train/val/test
    2. Tao env mode=MultiDiscrete
    3. Tao BranchingDRQNNetwork + BranchingDDQAgent (online/target)
    4. Vòng env: epsilon-greedy -> replay -> TD update (Double DQN) -> eval/checkpoint

Chạy:
    python -m src.training.DDQ
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agents.branchingddn_agent import BranchingDDQAgent
from src.constants import DATA_PATH, FEATURES, TICKERS, WINDOW_SIZE
from src.environment.trading_env import TradingEnv
from src.models.lstm import BranchingDRQNNetwork
from src.training.PPO import (
    average_metrics,
    build_baseline_comparison,
    compute_learning_rate,
    evaluate_baselines,
    format_baseline_comparison,
    set_seed,
)
from src.utils.data_splitter import load_data, split_by_ratio
from src.utils.logger import TrainingLogger, make_run_id
from src.utils.metrics import compute_all, format_report


DEFAULT_CONFIG = {
    "tickers": TICKERS,
    "features": FEATURES,
    "window_size": WINDOW_SIZE,
    "data_path": DATA_PATH,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    "initial_balance": 1_000_000_000,
    "fee_rate": 0.001,
    "max_steps_train": 512,
    "max_steps_eval": 9999,
    "reward_scaling": 1.0,
    "reward_name": "advanced",
    "reward_window": 30,

    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "k": 3,

    "learning_rate": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "replay_buffer_size": 100_000,
    "learning_starts": 5_000,
    "train_freq": 4,
    "gradient_steps": 1,
    "max_grad_norm": 0.5,

    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 200_000,

    "total_timesteps": 500_000,
    "lr_schedule": "cosine",
    "min_learning_rate": 1e-5,
    "train_metrics_log_every": 200,

    "eval_freq": 10_000,
    "save_freq": 20_000,
    "n_eval_episodes": 1,
    "eval_account_report": True,
    "eval_account_report_every": 1,
    "eval_account_report_top_k": 0,

    "seed": 42,
    "device": "auto",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "Conf" / "branchingddq_conf.yaml"


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

    run_cfg = {**DEFAULT_CONFIG, **{k: v for k, v in raw_cfg.items() if k in DEFAULT_CONFIG}}
    if overrides:
        run_cfg.update(overrides)

    return resolve_ddq_config(config=run_cfg)


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
    shared_fc_key = "shared_fc.0.weight"
    branch_bias_keys = sorted(
        [
            key
            for key in state_dict
            if key.startswith("advantage_streams.") and key.endswith(".2.bias")
        ],
        key=lambda k: int(k.split(".")[1]),
    )
    has_legacy_adv_head = "advantage_stream.2.bias" in state_dict
    if (
        ih_key not in state_dict
        or hh_key not in state_dict
        or shared_fc_key not in state_dict
        or (not branch_bias_keys and not has_legacy_adv_head)
    ):
        raise ValueError(f"Checkpoint không đúng định dạng DDQ-LSTM mong đợi: {ckpt_path}")

    hidden_size = int(state_dict[hh_key].shape[1])
    num_layers = len([k for k in state_dict if k.startswith("feature_extractor.lstm.weight_hh_l")])
    input_size = int(state_dict[ih_key].shape[1])
    combined_dim = int(state_dict[shared_fc_key].shape[1])
    n_stocks = combined_dim - hidden_size - 1
    if n_stocks <= 0:
        raise ValueError(
            f"Không suy luận được n_stocks từ checkpoint {ckpt_path}: "
            f"combined_dim={combined_dim}, hidden_size={hidden_size}"
        )
    if input_size % n_stocks != 0:
        raise ValueError(
            f"Không suy luận được số features từ checkpoint {ckpt_path}: "
            f"input_size={input_size}, n_stocks={n_stocks}"
        )
    if branch_bias_keys:
        if len(branch_bias_keys) != n_stocks:
            raise ValueError(
                f"Checkpoint có {len(branch_bias_keys)} advantage branches nhưng "
                f"suy luận được n_stocks={n_stocks}. Không thể dựng model tương thích."
            )
        inferred_k = int(state_dict[branch_bias_keys[0]].shape[0])
    else:
        n_actions = int(state_dict["advantage_stream.2.bias"].shape[0])
        if n_actions % n_stocks != 0:
            raise ValueError(
                f"Không suy luận được k từ checkpoint {ckpt_path}: "
                f"n_actions={n_actions}, n_stocks={n_stocks}"
            )
        inferred_k = n_actions // n_stocks

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
        "k": inferred_k,
    }
    if overrides:
        inferred_cfg.update(overrides)

    return resolve_ddq_config(config=inferred_cfg)


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


def _resolve_config_path(config_path: str | os.PathLike | None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    path = Path(config_path)
    if path.is_absolute():
        return path

    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path

    return (PROJECT_ROOT / path).resolve()


def load_ddq_config(config_path: str | os.PathLike | None = None) -> dict:
    path = _resolve_config_path(config_path)
    if not path.exists():
        if config_path is None:
            return {}
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Cần cài PyYAML để đọc file config DDQ.") from exc

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config DDQ phải là mapping/dict, nhận được: {type(cfg).__name__}")

    unknown_keys = sorted(set(cfg) - set(DEFAULT_CONFIG))
    if unknown_keys:
        raise KeyError(f"Config DDQ chứa key không hợp lệ: {unknown_keys}")

    return cfg


def resolve_ddq_config(
    config: dict | None = None,
    config_path: str | os.PathLike | None = None,
) -> dict:
    yaml_cfg = load_ddq_config(config_path)
    runtime_cfg = config or {}
    resolved = {**DEFAULT_CONFIG, **yaml_cfg, **runtime_cfg}

    # YAML co the parse mot so gia tri dang chuoi (vi du: "1e-4").
    # Ep kieu som de toan bo pipeline luon nhan dung numeric types.
    float_keys = [
        "learning_rate",
        "min_learning_rate",
        "gamma",
        "tau",
        "max_grad_norm",
        "epsilon_start",
        "epsilon_end",
        "reward_scaling",
        "fee_rate",
        "dropout",
        "train_ratio",
        "val_ratio",
        "test_ratio",
    ]
    int_keys = [
        "window_size",
        "max_steps_train",
        "max_steps_eval",
        "hidden_size",
        "num_layers",
        "k",
        "batch_size",
        "replay_buffer_size",
        "learning_starts",
        "train_freq",
        "gradient_steps",
        "epsilon_decay_steps",
        "total_timesteps",
        "train_metrics_log_every",
        "eval_freq",
        "save_freq",
        "n_eval_episodes",
        "eval_account_report_every",
        "eval_account_report_top_k",
        "seed",
    ]

    for key in float_keys:
        try:
            resolved[key] = float(resolved[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config key '{key}' phải là số thực hợp lệ, nhận được: {resolved[key]!r}") from exc

    for key in int_keys:
        try:
            resolved[key] = int(resolved[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config key '{key}' phải là số nguyên hợp lệ, nhận được: {resolved[key]!r}") from exc

    eval_report_raw = resolved.get("eval_account_report", False)
    if isinstance(eval_report_raw, bool):
        resolved["eval_account_report"] = eval_report_raw
    elif isinstance(eval_report_raw, (int, float)):
        resolved["eval_account_report"] = bool(eval_report_raw)
    elif isinstance(eval_report_raw, str):
        normalized = eval_report_raw.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            resolved["eval_account_report"] = True
        elif normalized in {"0", "false", "no", "n", "off"}:
            resolved["eval_account_report"] = False
        else:
            raise ValueError(
                "Config key 'eval_account_report' phải là bool hợp lệ "
                f"(true/false), nhận được: {eval_report_raw!r}"
            )
    else:
        raise ValueError(
            "Config key 'eval_account_report' phải là bool/int/str hợp lệ, "
            f"nhận được: {type(eval_report_raw).__name__}"
        )

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
    if resolved["total_timesteps"] <= 0:
        raise ValueError("total_timesteps phải > 0.")
    if resolved["train_metrics_log_every"] <= 0:
        raise ValueError("train_metrics_log_every phải > 0.")
    if resolved["epsilon_decay_steps"] < 0:
        raise ValueError("epsilon_decay_steps phải >= 0.")

    return resolved


def compute_epsilon(step: int, cfg: dict) -> float:
    start = float(cfg["epsilon_start"])
    end = float(cfg["epsilon_end"])
    decay = int(cfg["epsilon_decay_steps"])
    if decay <= 0:
        return end
    t = min(int(step), decay)
    frac = t / decay
    return float(start + (end - start) * frac)


def format_train_metrics_line(step: int, epsilon: float, lr: float, stats: dict) -> str:
    """Format train metrics thành 1 dòng ngắn gọn cho notebook/terminal."""
    n_gs = int(stats.get("n_gradient_steps", 0))
    q_mean = float(stats.get("q_mean", 0.0))
    q_std = float(stats.get("q_std", 0.0))
    td_abs = float(stats.get("td_abs_mean", 0.0))
    grad_norm = float(stats.get("grad_norm", 0.0))
    done_ratio = float(stats.get("done_ratio", 0.0))
    ap0 = float(stats.get("action_prob_0", 0.0))
    ap1 = float(stats.get("action_prob_1", 0.0))
    ap2 = float(stats.get("action_prob_2", 0.0))
    loss = float(stats.get("loss", 0.0))

    return (
        f"[train] step={step:,} | loss={loss:.6f} | eps={epsilon:.4f} | lr={lr:.2e} | "
        f"q={q_mean:.4f}+/-{q_std:.4f} | td_abs={td_abs:.5f} | grad={grad_norm:.4f} | "
        f"done={done_ratio:.3f} | act(s/h/b)=({ap0:.2f}/{ap1:.2f}/{ap2:.2f}) | gs={n_gs}"
    )


def make_env(tickers, data_dict, config, for_eval=False):
    return TradingEnv(
        tickers=tickers,
        mode="MultiDiscrete",
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
        print_verbosity=999999,
    )


def collect_account_history(
    agent: BranchingDDQAgent,
    env: TradingEnv,
    state_space,
    n_episodes: int,
    deterministic: bool = True,
) -> pd.DataFrame:
    records = []
    effective = int(n_episodes)
    if deterministic and not getattr(env, "random_start", True):
        effective = 1

    eps = 0.0 if deterministic else 0.05
    agent.model.eval()

    with torch.no_grad():
        for ep in range(1, effective + 1):
            obs, info = env.reset()
            done = False
            step_idx = 0

            while True:
                holdings = np.asarray(info["holdings"], dtype=np.int64)
                row = {
                    "episode": ep,
                    "step": step_idx,
                    "date": str(info["date"]),
                    "cash": float(info["cash"]),
                    "total_shares": int(np.sum(holdings)),
                    "total_asset": float(info["portfolio_value"]),
                }
                for ticker, qty in zip(env.state_space.tickers, holdings):
                    row[f"holding_{ticker}"] = int(qty)
                records.append(row)

                if done:
                    break

                ms, ps = state_space.flat_obs_to_sequential(obs)
                action = agent.select_action(ms, ps, epsilon=eps)
                obs, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_idx += 1

    account_df = pd.DataFrame(records)
    if account_df.empty:
        return account_df

    account_df["growth_pct_vs_prev"] = (
        account_df.groupby("episode")["total_asset"].pct_change().fillna(0.0) * 100.0
    )
    return account_df


def save_account_history_csv(
    account_df: pd.DataFrame,
    ckpt_path: Path,
    run_dir: str | os.PathLike | None = None,
) -> Path:
    if run_dir is not None:
        output_dir = Path(run_dir) / "eval"
    elif ckpt_path.parent.name == "checkpoints":
        output_dir = ckpt_path.parent.parent / "eval"
    else:
        output_dir = ckpt_path.parent / "eval"

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"account_history_{timestamp}.csv"
    account_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def train_branchingddq(config: dict | None = None, config_path: str | os.PathLike | None = None):
    cfg = resolve_ddq_config(config=config, config_path=config_path)
    set_seed(cfg["seed"])

    data_dict = load_data(tickers=cfg["tickers"], data_path=cfg["data_path"])
    split = split_by_ratio(
        data_dict,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
    )
    print(f"Data split: {split.summary()}")

    train_env = make_env(cfg["tickers"], split.train, cfg, for_eval=False)
    val_env = make_env(cfg["tickers"], split.val, cfg, for_eval=True)
    test_env = make_env(cfg["tickers"], split.test, cfg, for_eval=True)

    state_space = train_env.state_space
    n_stocks = state_space.n_stocks
    n_features = state_space.n_features

    model = BranchingDRQNNetwork(
        n_stocks=n_stocks,
        n_features=n_features,
        seq_len=cfg["window_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        k=cfg["k"],
    )

    agent = BranchingDDQAgent(
        model=model,
        lr=cfg["learning_rate"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        batch_size=cfg["batch_size"],
        replay_buffer_size=cfg["replay_buffer_size"],
        learning_starts=cfg["learning_starts"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        max_grad_norm=cfg["max_grad_norm"],
        device=cfg["device"],
    )

    print(f"Device: {agent.device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    run_id = make_run_id("branchingddq")
    logger = TrainingLogger(
        run_id=run_id,
        agent="BranchingDDQ_LSTM",
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
        f"LR schedule: {cfg['lr_schedule']} | "
        f"base_lr={cfg['learning_rate']:.2e} | min_lr={cfg['min_learning_rate']:.2e}"
    )

    save_dir = logger.get_run_dir() / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = train_env.reset(seed=cfg["seed"])
    episode_reward = 0.0
    episode_counter = 0
    best_val_sharpe = -np.inf
    latest_val_baselines = None

    for step in range(1, cfg["total_timesteps"] + 1):
        progress = step / cfg["total_timesteps"]
        agent.set_lr(
            compute_learning_rate(
                base_lr=cfg["learning_rate"],
                min_lr=cfg["min_learning_rate"],
                progress=progress,
                schedule=cfg["lr_schedule"],
            )
        )
        epsilon = compute_epsilon(step, cfg)

        ms, ps = state_space.flat_obs_to_sequential(obs)
        action = agent.select_action(ms, ps, epsilon)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated

        next_ms, next_ps = state_space.flat_obs_to_sequential(next_obs)
        agent.store_transition(ms, ps, action, reward, next_ms, next_ps, done)
        agent.total_steps = step

        episode_reward += float(reward)

        train_stats = agent.maybe_train()
        if train_stats is not None:
            logger.log_train_step(
                step=step,
                policy_loss=0.0,
                value_loss=train_stats["loss"],
                entropy=0.0,
                approx_kl=0.0,
                clip_fraction=0.0,
                learning_rate=agent.get_lr(),
                extra={"epsilon": epsilon, "n_gradient_steps": train_stats.get("n_gradient_steps", 0)},
            )
            if step % int(cfg["train_metrics_log_every"]) == 0:
                logger.info(
                    format_train_metrics_line(
                        step=step,
                        epsilon=epsilon,
                        lr=agent.get_lr(),
                        stats=train_stats,
                    )
                )

        if done:
            episode_counter += 1
            logger.log_episode(
                episode=episode_counter,
                total_reward=episode_reward,
                portfolio_value=info.get("portfolio_value", 0),
                total_return=(info.get("portfolio_value", 0) - train_env.initial_balance)
                / train_env.initial_balance,
                n_trades=train_env.trades,
                total_cost=train_env.cost,
                steps=train_env.current_step,
            )
            episode_reward = 0.0
            obs, _ = train_env.reset()
        else:
            obs = next_obs

        if step % cfg["eval_freq"] == 0:
            val_values = agent.evaluate(val_env, val_env.state_space, cfg["n_eval_episodes"])
            val_metrics_list = [compute_all(pv, cfg["initial_balance"]) for pv in val_values]
            avg_val_metrics = average_metrics(val_metrics_list)
            val_baselines = evaluate_baselines(val_env, cfg["initial_balance"])
            latest_val_baselines = {
                name: build_baseline_comparison(avg_val_metrics, metrics)
                for name, metrics in val_baselines.items()
            }
            logger.log_eval(episode=episode_counter, metrics=avg_val_metrics, split="val")
            logger.info(f"\n{format_report(avg_val_metrics)}")
            for name, metrics in val_baselines.items():
                logger.info(format_baseline_comparison(name.upper(), avg_val_metrics, metrics))

            if avg_val_metrics.get("sharpe_ratio", -np.inf) > best_val_sharpe:
                best_val_sharpe = avg_val_metrics["sharpe_ratio"]
                agent.save(str(save_dir / "best_model.pt"))
                logger.info(f"Best val Sharpe: {best_val_sharpe:.4f} -> saved best_model.pt")

        if step % cfg["save_freq"] == 0:
            agent.save(str(save_dir / f"checkpoint_{step}.pt"))

    best_path = save_dir / "best_model.pt"
    if best_path.exists():
        agent.load(str(best_path))
        logger.info("Loaded best_model.pt for final test evaluation")
    else:
        logger.info("best_model.pt chưa tồn tại, dùng model hiện tại cho final eval")

    test_values = agent.evaluate(
        test_env,
        test_env.state_space,
        n_episodes=cfg["n_eval_episodes"],
        account_report=cfg["eval_account_report"],
        report_every=cfg["eval_account_report_every"],
        top_k_holdings=cfg["eval_account_report_top_k"],
    )
    test_metrics_list = [compute_all(pv, cfg["initial_balance"]) for pv in test_values]
    avg_test = average_metrics(test_metrics_list)
    test_baselines = evaluate_baselines(test_env, cfg["initial_balance"])

    logger.log_eval(episode=episode_counter, metrics=avg_test, split="test")
    logger.info(f"\n=== FINAL TEST RESULTS (avg {len(test_values)} episodes) ===")
    logger.info(f"\n{format_report(avg_test)}")
    for name, metrics in test_baselines.items():
        logger.info(format_baseline_comparison(name.upper(), avg_test, metrics))

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
        },
    )

    return agent, avg_test


def evaluate_branchingddq(
    config: dict | None = None,
    config_path: str | os.PathLike | None = None,
    run_dir: str | os.PathLike | None = None,
    checkpoint_path: str | os.PathLike | None = None,
    n_eval_episodes: int | None = None,
) -> dict:
    overrides = dict(config or {})
    base_cfg = resolve_ddq_config(config=None, config_path=config_path)

    eval_cfg: dict
    resolved: dict | None = None
    config_source = "runtime_config"
    ckpt_source = "manual"

    selected_run_dir: Path | None = None

    if run_dir is not None:
        run_path = Path(run_dir)
        selected_run_dir = run_path
        eval_cfg = load_run_config(run_path, overrides=overrides)
        config_source = "run_config"

        if checkpoint_path is None:
            ckpt_path, ckpt_source = resolve_eval_checkpoint(run_path)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"Không tìm thấy checkpoint trong run_dir={run_path}. reason={ckpt_source}"
                )
        else:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")
    elif checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")
        eval_cfg = infer_run_config_from_checkpoint(ckpt_path, base_config=base_cfg, overrides=overrides)
        config_source = "checkpoint_inferred"
    else:
        roots = get_results_root_candidates(project_root=PROJECT_ROOT, cwd=Path.cwd())
        resolved = resolve_eval_run_across_roots(roots, base_config=base_cfg, overrides=overrides)
        if resolved["run_dir"] is None:
            checked = [str(p) for p in resolved.get("checked_results_roots", [])]
            raise FileNotFoundError(
                "Không tìm thấy run/checkpoint DDQ hợp lệ để evaluate. "
                f"Đã kiểm tra các thư mục: {checked}"
            )

        ckpt_path = resolved["ckpt_path"]
        selected_run_dir = resolved["run_dir"]
        ckpt_source = resolved["ckpt_source"]
        eval_cfg = resolved["config"]
        config_source = resolved["config_source"]

    if n_eval_episodes is not None:
        eval_cfg["n_eval_episodes"] = int(n_eval_episodes)

    set_seed(eval_cfg["seed"])
    data_dict = load_data(tickers=eval_cfg["tickers"], data_path=eval_cfg["data_path"])
    split = split_by_ratio(
        data_dict,
        train_ratio=eval_cfg["train_ratio"],
        val_ratio=eval_cfg["val_ratio"],
        test_ratio=eval_cfg["test_ratio"],
    )
    test_env = make_env(eval_cfg["tickers"], split.test, eval_cfg, for_eval=True)
    state_space = test_env.state_space

    model = BranchingDRQNNetwork(
        n_stocks=state_space.n_stocks,
        n_features=state_space.n_features,
        seq_len=eval_cfg["window_size"],
        hidden_size=eval_cfg["hidden_size"],
        num_layers=eval_cfg["num_layers"],
        dropout=eval_cfg["dropout"],
        k=eval_cfg["k"],
    )
    agent = BranchingDDQAgent(
        model=model,
        lr=eval_cfg["learning_rate"],
        gamma=eval_cfg["gamma"],
        tau=eval_cfg["tau"],
        batch_size=eval_cfg["batch_size"],
        replay_buffer_size=eval_cfg["replay_buffer_size"],
        learning_starts=eval_cfg["learning_starts"],
        train_freq=eval_cfg["train_freq"],
        gradient_steps=eval_cfg["gradient_steps"],
        max_grad_norm=eval_cfg["max_grad_norm"],
        device=eval_cfg["device"],
    )
    agent.load(str(ckpt_path))

    test_values = agent.evaluate(
        test_env,
        test_env.state_space,
        n_episodes=eval_cfg["n_eval_episodes"],
        account_report=eval_cfg["eval_account_report"],
        report_every=eval_cfg["eval_account_report_every"],
        top_k_holdings=eval_cfg["eval_account_report_top_k"],
    )
    account_df = collect_account_history(
        agent,
        test_env,
        test_env.state_space,
        n_episodes=eval_cfg["n_eval_episodes"],
        deterministic=True,
    )
    account_csv_path = save_account_history_csv(account_df, ckpt_path=Path(ckpt_path), run_dir=selected_run_dir)
    test_metrics_list = [compute_all(pv, eval_cfg["initial_balance"]) for pv in test_values]
    avg_test = average_metrics(test_metrics_list)
    test_baselines = evaluate_baselines(test_env, eval_cfg["initial_balance"])

    print("\n=== BRANCHING DDQ EVALUATION ===")
    print(f"checkpoint: {ckpt_path} ({ckpt_source})")
    print(f"config_source: {config_source}")
    if resolved is not None and resolved.get("run_dir") is not None:
        print(f"run_dir: {resolved['run_dir']}")
        print(f"results_root: {resolved.get('results_root')}")
    print(f"episodes: {eval_cfg['n_eval_episodes']}")
    print(f"account_csv: {account_csv_path}")
    print(f"\n{format_report(avg_test)}")
    for name, metrics in test_baselines.items():
        print(format_baseline_comparison(name.upper(), avg_test, metrics))

    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_source": ckpt_source,
        "config_source": config_source,
        "account_csv": str(account_csv_path),
        "metrics": avg_test,
        "baseline_comparisons": {
            name: build_baseline_comparison(avg_test, metrics)
            for name, metrics in test_baselines.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Branching DDQ-LSTM trading agent.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path tới file YAML config. Mặc định dùng Conf/branchingddq_conf.yaml nếu tồn tại.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Chạy evaluate-only thay vì train.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory để load config/checkpoint cho evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path checkpoint .pt để evaluate.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Override số episode khi evaluate.",
    )
    args = parser.parse_args()

    if args.eval:
        evaluate_branchingddq(
            config_path=args.config,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            n_eval_episodes=args.n_eval_episodes,
        )
    else:
        train_branchingddq(config_path=args.config)
