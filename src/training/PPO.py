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
import random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.constants import TICKERS, FEATURES, WINDOW_SIZE
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
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # --- Environment ---
    "initial_balance": 1_000_000_000,
    "fee_rate": 0.0015,
    "max_steps_train": 200,
    "max_steps_eval": 9999,
    "reward_scaling": 1e-4,

    # --- Model (LSTM) ---
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "log_std_init": -0.5,

    # --- PPO ---
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,

    # --- Schedule ---
    "total_timesteps": 500_000,
    "lr_decay": True,
    "eval_freq": 10,
    "save_freq": 50,
    "n_eval_episodes": 3,

    # --- Misc ---
    "seed": 42,
    "device": "auto",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        print_verbosity=999999,
    )


def train_ppo(config: dict = None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])

    # ----------------------------------------------------------------
    # 1. Load & split data
    # ----------------------------------------------------------------
    data_dict = load_data(tickers=cfg["tickers"])
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
    )
    logger.info(f"Data split: {split.summary()}")
    logger.info(f"Device: {agent.device} | Params: {total_params:,}")

    # ----------------------------------------------------------------
    # 5. Training loop
    # ----------------------------------------------------------------
    n_rollouts = cfg["total_timesteps"] // cfg["n_steps"]
    episode_counter = 0
    best_val_sharpe = -np.inf
    obs = None

    save_dir = logger.get_run_dir() / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    for rollout in range(1, n_rollouts + 1):
        # LR linear decay
        if cfg["lr_decay"]:
            progress = agent.total_steps / cfg["total_timesteps"]
            new_lr = cfg["learning_rate"] * (1.0 - progress)
            new_lr = max(new_lr, 1e-6)
            agent.set_lr(new_lr)

        # Collect
        obs, ep_infos = agent.collect_rollout(train_env, state_space, cfg["n_steps"], obs)

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
        )

        # Periodic info
        if rollout % 5 == 0 or rollout == 1:
            logger.info(
                f"[Rollout {rollout}/{n_rollouts}] "
                f"steps={agent.total_steps:,} | "
                f"pi_loss={update_stats['policy_loss']:.5f} | "
                f"v_loss={update_stats['value_loss']:.5f} | "
                f"kl={update_stats['approx_kl']:.5f} | "
                f"lr={agent.get_lr():.2e}"
            )

        # Periodic eval on val set
        if rollout % cfg["eval_freq"] == 0:
            val_values = agent.evaluate(val_env, val_env.state_space, cfg["n_eval_episodes"])
            for i, pv in enumerate(val_values):
                metrics = compute_all(pv, cfg["initial_balance"])
                logger.log_eval(episode=episode_counter, metrics=metrics, split="val")

                if i == 0:
                    logger.info(f"\n{format_report(metrics)}")

                    if metrics.get("sharpe_ratio", -999) > best_val_sharpe:
                        best_val_sharpe = metrics["sharpe_ratio"]
                        agent.save(str(save_dir / "best_model.pt"))
                        logger.info(f"Best val Sharpe: {best_val_sharpe:.4f} -> saved best_model.pt")

        # Periodic checkpoint
        if rollout % cfg["save_freq"] == 0:
            agent.save(str(save_dir / f"checkpoint_{agent.total_steps}.pt"))

    # ----------------------------------------------------------------
    # 6. Final eval on test set (load best model)
    # ----------------------------------------------------------------
    best_path = save_dir / "best_model.pt"
    if best_path.exists():
        agent.load(str(best_path))
        logger.info("Loaded best_model.pt for final test evaluation")

    test_values = agent.evaluate(test_env, test_env.state_space, n_episodes=cfg["n_eval_episodes"])
    test_metrics_list = [compute_all(pv, cfg["initial_balance"]) for pv in test_values]

    avg_test = {}
    for key in test_metrics_list[0]:
        vals = [m[key] for m in test_metrics_list if key in m]
        avg_test[key] = float(np.mean(vals))

    logger.log_eval(episode=episode_counter, metrics=avg_test, split="test")
    logger.info(f"\n=== FINAL TEST RESULTS (avg {len(test_values)} episodes) ===")
    logger.info(f"\n{format_report(avg_test)}")

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
        },
    )

    return agent, avg_test


if __name__ == "__main__":
    train_ppo()
