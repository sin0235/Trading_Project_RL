"""
Logger thống nhất cho toàn bộ project.

Cấu trúc log:
    results/
        runs/
            <run_id>/
                training.log      - log text đầy đủ
                metrics.csv       - metrics theo từng episode (dùng để vẽ đồ thị)
                eval_metrics.csv  - metrics đánh giá cuối mỗi checkpoint
                config.json       - hyperparameter và cấu hình thí nghiệm
                summary.json      - kết quả tổng hợp cuối cùng

Cách dùng:
    logger = TrainingLogger(run_id="ppo_run_01", agent="PPO", config={...})
    logger.log_episode(episode=1, reward=..., portfolio_value=..., ...)
    logger.log_eval(episode=100, metrics={...})
    logger.save_summary(metrics={...})
"""

import os
import json
import csv
import logging
import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


RESULTS_DIR = "results/runs"


class TrainingLogger:
    """
    Logger tập trung cho một lần chạy thí nghiệm (run).
    Ghi đồng thời ra console, file text và CSV.
    """

    def __init__(
        self,
        run_id: str,
        agent: str,
        config: Dict[str, Any],
        results_dir: str = RESULTS_DIR,
        console_level: int = logging.INFO,
    ):
        """
        Args:
            run_id: Tên định danh run, ví dụ "ppo_lstm_v1" hoặc tự động theo timestamp.
            agent: Tên agent, ví dụ "PPO" hoặc "DQN".
            config: Dict hyperparameter và cấu hình thí nghiệm.
            results_dir: Thư mục gốc chứa kết quả.
            console_level: Mức log ra console (logging.INFO / logging.DEBUG).
        """
        self.run_id = run_id
        self.agent = agent
        self.config = config
        self.start_time = datetime.datetime.now()

        self.run_dir = Path(results_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._setup_text_logger(console_level)
        self._setup_csv_writers()
        self._save_config()

        self.logger.info(f"Run bắt đầu: {run_id} | Agent: {agent} | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        portfolio_value: float,
        total_return: float,
        n_trades: int,
        total_cost: float,
        steps: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi log sau mỗi episode training.

        Args:
            episode: Số thứ tự episode.
            total_reward: Tổng reward tích lũy trong episode.
            portfolio_value: Giá trị tài khoản cuối episode.
            total_return: Tỷ suất lợi nhuận (0.05 = 5%).
            n_trades: Số lệnh giao dịch đã thực hiện.
            total_cost: Tổng phí giao dịch.
            steps: Số bước trong episode.
            extra: Các trường bổ sung tuỳ chọn (ví dụ loss, entropy...).
        """
        row = {
            "episode": episode,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_reward": round(total_reward, 6),
            "portfolio_value": round(portfolio_value, 2),
            "total_return_pct": round(total_return * 100, 4),
            "n_trades": n_trades,
            "total_cost": round(total_cost, 2),
            "steps": steps,
        }
        if extra:
            row.update({k: _safe_round(v) for k, v in extra.items()})

        self._write_csv(self._episode_writer, self._episode_file, row)

        diagnostics = []
        if extra:
            avg_turnover = extra.get("avg_turnover")
            steps_with_trades = extra.get("steps_with_trades")
            avg_concentration_sum = extra.get("avg_concentration_sum")
            if avg_turnover is not None:
                diagnostics.append(f"turnover={float(avg_turnover):.2%}")
            if steps_with_trades is not None:
                diagnostics.append(f"trade_steps={int(round(float(steps_with_trades))):>3}/{steps}")
            if avg_concentration_sum is not None:
                diagnostics.append(f"conc_sum={float(avg_concentration_sum):.2f}")

        self.logger.info(
            f"[Ep {episode:>6}] reward={total_reward:>9.4f} | "
            f"value={portfolio_value:>16,.0f} | "
            f"return={total_return:>7.2%} | "
            f"trades={n_trades:>4} | cost={total_cost:>10,.0f}"
            + (f" | {' | '.join(diagnostics)}" if diagnostics else "")
        )

    def log_eval(
        self,
        episode: int,
        metrics: Dict[str, float],
        split: str = "val",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi kết quả đánh giá tại một checkpoint.

        Args:
            episode: Episode tại thời điểm đánh giá.
            metrics: Dict từ metrics.compute_all().
            split: "val" hoặc "test".
        """
        row = {"episode": episode, "split": split, "timestamp": datetime.datetime.now().isoformat()}
        row.update({k: _safe_round(v) for k, v in metrics.items()})
        if extra:
            row.update({k: _safe_round(v) for k, v in extra.items()})

        self._write_csv(self._eval_writer, self._eval_file, row)

        diagnostics = []
        if extra:
            avg_turnover = extra.get("avg_turnover")
            steps_with_trades = extra.get("steps_with_trades")
            avg_concentration_sum = extra.get("avg_concentration_sum")
            if avg_turnover is not None:
                diagnostics.append(f"turnover={float(avg_turnover):.2%}")
            if steps_with_trades is not None:
                diagnostics.append(f"trade_steps={float(steps_with_trades):.1f}")
            if avg_concentration_sum is not None:
                diagnostics.append(f"conc_sum={float(avg_concentration_sum):.2f}")

        self.logger.info(
            f"[EVAL/{split.upper()} Ep {episode}] "
            f"total_return={metrics.get('total_return', 0):.2%} | "
            f"sharpe={metrics.get('sharpe_ratio', 0):.4f} | "
            f"max_dd={metrics.get('max_drawdown', 0):.2%} | "
            f"win_rate={metrics.get('win_rate', 0):.2%}"
            + (f" | {' | '.join(diagnostics)}" if diagnostics else "")
        )

    def log_train_step(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        approx_kl: float,
        clip_fraction: float,
        learning_rate: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi log sau mỗi update bước training (PPO update / DQN gradient step).
        """
        row = {
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            "policy_loss": round(policy_loss, 6),
            "value_loss": round(value_loss, 6),
            "entropy": round(entropy, 6),
            "approx_kl": round(approx_kl, 6),
            "clip_fraction": round(clip_fraction, 6),
            "learning_rate": learning_rate,
        }
        if extra:
            row.update({k: _safe_round(v) for k, v in extra.items()})

        self._write_csv(self._step_writer, self._step_file, row)

        self.logger.debug(
            f"[Step {step:>8}] "
            f"pi_loss={policy_loss:>8.5f} | "
            f"v_loss={value_loss:>8.5f} | "
            f"entropy={entropy:>7.5f} | "
            f"kl={approx_kl:>7.5f} | "
            f"clip={clip_fraction:>5.3f} | "
            f"lr={learning_rate:.2e}"
        )

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def save_summary(self, metrics: Dict[str, float], extra: Optional[Dict] = None) -> None:
        """
        Lưu kết quả tổng hợp cuối cùng vào summary.json.
        Đây là file chính để đưa vào báo cáo.
        """
        elapsed = datetime.datetime.now() - self.start_time
        summary = {
            "run_id": self.run_id,
            "agent": self.agent,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "elapsed_seconds": int(elapsed.total_seconds()),
            "final_metrics": {k: _safe_round(v) for k, v in metrics.items()},
        }
        if extra:
            summary["extra"] = extra

        path = self.run_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Summary đã lưu: {path}")
        self.logger.info(
            f"Run kết thúc sau {elapsed}. "
            f"total_return={metrics.get('total_return', 0):.2%} | "
            f"sharpe={metrics.get('sharpe_ratio', 0):.4f}"
        )

    def get_run_dir(self) -> Path:
        return self.run_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_text_logger(self, console_level: int) -> None:
        self.logger = logging.getLogger(self.run_id)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        # File handler — ghi toàn bộ DEBUG trở lên
        log_path = self.run_dir / "training.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

    def _setup_csv_writers(self) -> None:
        episode_cols = [
            "episode", "timestamp", "total_reward", "portfolio_value",
            "total_return_pct", "n_trades", "total_cost", "steps",
            "avg_turnover", "steps_with_trades", "steps_with_trades_pct",
            "avg_concentration_sum",
        ]
        eval_cols = [
            "episode", "split", "timestamp",
            "total_return", "annualized_return", "annualized_vol",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "max_drawdown_duration",
            "win_rate", "profit_factor",
            "skewness", "kurtosis", "var_95", "cvar_95",
            "final_value", "initial_value",
            "alpha", "beta", "information_ratio",
            "avg_turnover", "steps_with_trades", "steps_with_trades_pct",
            "avg_concentration_sum",
        ]
        step_cols = [
            "step", "timestamp", "policy_loss", "value_loss",
            "entropy", "approx_kl", "clip_fraction", "learning_rate",
            "avg_turnover", "steps_with_trades", "steps_with_trades_pct",
            "avg_concentration_sum",
        ]

        self._episode_file, self._episode_writer = self._open_csv("metrics.csv", episode_cols)
        self._eval_file,    self._eval_writer    = self._open_csv("eval_metrics.csv", eval_cols)
        self._step_file,    self._step_writer    = self._open_csv("train_steps.csv", step_cols)

    def _open_csv(self, filename: str, fieldnames: List[str]):
        path = self.run_dir / filename
        f = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        return f, writer

    def _write_csv(self, writer: csv.DictWriter, file, row: dict) -> None:
        writer.writerow(row)
        file.flush()

    def _save_config(self) -> None:
        path = self.run_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"run_id": self.run_id, "agent": self.agent, **self.config},
                f, ensure_ascii=False, indent=2, default=str,
            )

    def __del__(self):
        for f in [self._episode_file, self._eval_file, self._step_file]:
            try:
                f.close()
            except Exception:
                pass


def make_run_id(agent: str, suffix: str = "") -> str:
    """
    Tạo run_id tự động theo định dạng: <agent>_<YYYYMMDD_HHMMSS>[_suffix].
    Ví dụ: PPO_20260313_143022_v1
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [agent.lower(), ts]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def _safe_round(v: Any, ndigits: int = 6) -> Any:
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v
