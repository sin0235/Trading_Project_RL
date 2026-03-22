"""
Training script cho Double DQN (DDQ) + DRQN (LSTM) trên TradingEnv (discrete).

Luồng chính:
    1. Load data, chia train/val/test
    2. Tạo env mode=discrete (DRQN)
    3. Tạo DRQNNetwork + DDQAgent (online/target)
    4. Vòng env: epsilon-greedy -> replay -> TD update (Double DQN) -> eval/checkpoint

Chạy:
    python -m src.training.DDQ
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agents.ddq_agent import DDQAgent
from src.constants import DATA_PATH, FEATURES, TICKERS, WINDOW_SIZE
from src.environment.trading_env import TradingEnv
from src.models.lstm import DRQNNetwork
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
    "reward_name": "sharpe",
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

    "eval_freq": 10_000,
    "save_freq": 20_000,
    "n_eval_episodes": 1,

    "seed": 42,
    "device": "auto",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "Conf" / "ddq_conf.yaml"


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


def make_env(tickers, data_dict, config, for_eval=False):
    return TradingEnv(
        tickers=tickers,
        mode="discrete",
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


def train_ddq(config: dict | None = None, config_path: str | os.PathLike | None = None):
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

    model = DRQNNetwork(
        n_stocks=n_stocks,
        n_features=n_features,
        seq_len=cfg["window_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        k=cfg["k"],
    )

    agent = DDQAgent(
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

    run_id = make_run_id("ddq")
    logger = TrainingLogger(
        run_id=run_id,
        agent="DDQ_LSTM",
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

    test_values = agent.evaluate(test_env, test_env.state_space, n_episodes=cfg["n_eval_episodes"])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDQ-LSTM trading agent.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path tới file YAML config. Mặc định dùng Conf/ddq_conf.yaml nếu tồn tại.",
    )
    args = parser.parse_args()

    train_ddq(config_path=args.config)
