"""
Branching DDQ (Double Deep Q-Learning) Agent với Branching DRQN cho MultiDiscrete trading.

Thiết kế:
    - ReplayBuffer: lưu (s, a_vec, r, s', done), trong đó a_vec có shape (n_stocks,)
    - Online + target network (Double DQN theo từng branch)
    - Soft-update target (polyak tau) sau mỗi gradient step
    - Epsilon-greedy qua BranchingDRQNNetwork.select_action
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lstm import BranchingDRQNNetwork
np.random.seed(42)
torch.manual_seed(42)


class ReplayBuffer:
    """Ring buffer numpy cho transitions đã flatten thành chuỗi LSTM."""

    def __init__(
        self,
        capacity: int,
        seq_len: int,
        market_feat_dim: int,
        portfolio_dim: int,
        n_branches: int,
    ):
        self.capacity = int(capacity)
        self.seq_len = seq_len
        self.market_feat_dim = market_feat_dim
        self.portfolio_dim = portfolio_dim
        self.n_branches = int(n_branches)

        self._ptr = 0
        self.size = 0

        self.market = np.zeros(
            (self.capacity, seq_len, market_feat_dim), dtype=np.float32
        )
        self.portfolio = np.zeros((self.capacity, portfolio_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.n_branches), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_market = np.zeros(
            (self.capacity, seq_len, market_feat_dim), dtype=np.float32
        )
        self.next_portfolio = np.zeros((self.capacity, portfolio_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def push(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_market_state: np.ndarray,
        next_portfolio_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._ptr
        self.market[i] = market_state
        self.portfolio[i] = portfolio_state
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.shape != (self.n_branches,):
            raise ValueError(
                f"Action shape {action_arr.shape} != ({self.n_branches},)"
            )
        self.actions[i] = action_arr
        self.rewards[i] = float(reward)
        self.next_market[i] = next_market_state
        self.next_portfolio[i] = next_portfolio_state
        self.dones[i] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "market_states": torch.tensor(self.market[idx], device=device, dtype=torch.float32),
            "portfolio_states": torch.tensor(
                self.portfolio[idx], device=device, dtype=torch.float32
            ),
            "actions": torch.tensor(self.actions[idx], device=device, dtype=torch.long),
            "rewards": torch.tensor(self.rewards[idx], device=device, dtype=torch.float32),
            "next_market_states": torch.tensor(
                self.next_market[idx], device=device, dtype=torch.float32
            ),
            "next_portfolio_states": torch.tensor(
                self.next_portfolio[idx], device=device, dtype=torch.float32
            ),
            "dones": torch.tensor(self.dones[idx], device=device, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return self.size


class BranchingDDQAgent:
    def __init__(
        self,
        model: BranchingDRQNNetwork,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 64,
        replay_buffer_size: int = 100_000,
        learning_starts: int = 5_000,
        train_freq: int = 4,
        gradient_steps: int = 1,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.max_grad_norm = max_grad_norm

        seq_len = model.seq_len
        market_feat_dim = model.n_stocks * model.n_features
        portfolio_dim = 1 + model.n_stocks

        self.buffer = ReplayBuffer(
            capacity=replay_buffer_size,
            seq_len=seq_len,
            market_feat_dim=market_feat_dim,
            portfolio_dim=portfolio_dim,
            n_branches=model.n_stocks,
        )

        self.total_steps = 0
        self.n_updates = 0
        self._train_step_counter = 0

    @staticmethod
    def _format_account_snapshot(
        info: Dict[str, Any],
        tickers: List[str],
        prev_snapshot: Optional[Dict[str, Any]],
        top_k_holdings: int,
    ) -> tuple[str, Dict[str, Any]]:
        cash = float(info.get("cash", 0.0))
        holdings = np.asarray(info.get("holdings", []), dtype=np.int64)
        prices = np.asarray(info.get("prices", []), dtype=np.float64)
        holding_values = holdings.astype(np.float64) * prices

        total_asset = float(info.get("portfolio_value", cash + float(np.sum(holding_values))))
        prev_total_asset = None if prev_snapshot is None else float(prev_snapshot["total_asset"])
        prev_cash = None if prev_snapshot is None else float(prev_snapshot["cash"])
        prev_holdings = None if prev_snapshot is None else np.asarray(prev_snapshot["holdings"], dtype=np.int64)

        delta_asset = 0.0 if prev_total_asset is None else total_asset - prev_total_asset
        delta_asset_pct = (
            0.0
            if prev_total_asset is None or abs(prev_total_asset) < 1e-12
            else (delta_asset / prev_total_asset) * 100.0
        )
        delta_cash = 0.0 if prev_cash is None else cash - prev_cash

        header = (
            f"[{info.get('date', 'N/A')}] step={info.get('step', -1)} | "
            f"tong_tai_san={total_asset:,.0f} ({delta_asset:+,.0f}, {delta_asset_pct:+.2f}%) | "
            f"tien_mat={cash:,.0f} ({delta_cash:+,.0f})"
        )

        non_zero_idx = np.flatnonzero(holdings)
        positions = []
        if non_zero_idx.size > 0:
            ranked = non_zero_idx[np.argsort(holding_values[non_zero_idx])[::-1]]
            if top_k_holdings > 0:
                ranked = ranked[:top_k_holdings]

            prev_holding_values = None if prev_snapshot is None else prev_snapshot["holding_values"]
            for idx in ranked:
                prev_value = 0.0 if prev_holding_values is None else float(prev_holding_values[idx])
                value_delta = holding_values[idx] - prev_value
                prev_qty = 0 if prev_holdings is None else int(prev_holdings[idx])
                qty_delta = int(holdings[idx]) - prev_qty
                weight = 0.0 if abs(total_asset) < 1e-12 else (holding_values[idx] / total_asset) * 100.0
                positions.append(
                    f"  - {tickers[idx]} | qty={int(holdings[idx])} | "
                    f"price={prices[idx]:,.2f} | value={holding_values[idx]:,.0f} | "
                    f"qty_delta={qty_delta:+,d} | value_delta={value_delta:+,.0f} | w={weight:.2f}%"
                )

            if top_k_holdings > 0 and non_zero_idx.size > len(ranked):
                positions.append(f"  - ... ({non_zero_idx.size - len(ranked)} ma khac)")
        else:
            positions.append("  - Khong nam giu co phieu")

        snapshot_text = "\n".join([header, "Danh muc:", *positions])
        snapshot_state = {
            "total_asset": total_asset,
            "cash": cash,
            "holdings": holdings.copy(),
            "holding_values": holding_values,
        }
        return snapshot_text, snapshot_state

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def select_action(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        self.model.eval()
        ms = torch.tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        ps = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.model.select_action(ms, ps, hidden=None, epsilon=float(epsilon))
        return np.asarray(action, dtype=np.int64)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store_transition(
        self,
        market_state: np.ndarray,
        portfolio_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_market_state: np.ndarray,
        next_portfolio_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(
            market_state,
            portfolio_state,
            action,
            reward,
            next_market_state,
            next_portfolio_state,
            done,
        )

    def maybe_train(self) -> Optional[Dict[str, float]]:
        self._train_step_counter += 1
        if self._train_step_counter % self.train_freq != 0:
            return None
        if len(self.buffer) < max(self.learning_starts, self.batch_size):
            return None

        self.model.train()
        tracked_stats: Dict[str, float] = {}
        n = 0
        for _ in range(self.gradient_steps):
            stats = self._update_once()
            for key, value in stats.items():
                tracked_stats[key] = tracked_stats.get(key, 0.0) + float(value)
            n += 1
        self.model.eval()

        self.n_updates += n
        out = {k: v / max(n, 1) for k, v in tracked_stats.items()}
        out["n_gradient_steps"] = float(n)
        return out

    def _update_once(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.device)

        ms = batch["market_states"]
        ps = batch["portfolio_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_ms = batch["next_market_states"]
        next_ps = batch["next_portfolio_states"]
        dones = batch["dones"]

        with torch.no_grad():
            next_q_online, _ = self.model(next_ms, next_ps, hidden=None)  # (B, N, K)
            next_actions = next_q_online.argmax(dim=-1, keepdim=True)  # (B, N, 1)
            next_q_target, _ = self.target_model(next_ms, next_ps, hidden=None)
            next_q = next_q_target.gather(-1, next_actions).squeeze(-1)  # (B, N)
            target = rewards.unsqueeze(1) + self.gamma * (1.0 - dones.unsqueeze(1)) * next_q

        current_q, _ = self.model(ms, ps, hidden=None)
        current_q = current_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, N)

        td_error = target - current_q
        loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        action_counts = torch.bincount(actions.view(-1), minlength=self.model.k).float()
        action_probs = action_counts / action_counts.sum().clamp_min(1.0)

        stats = {
            "loss": float(loss.item()),
            "q_mean": float(current_q.mean().item()),
            "q_std": float(current_q.std(unbiased=False).item()),
            "target_mean": float(target.mean().item()),
            "target_std": float(target.std(unbiased=False).item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "done_ratio": float(dones.mean().item()),
            "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
        }
        for action_id, prob in enumerate(action_probs.tolist()):
            stats[f"action_prob_{action_id}"] = float(prob)

        return stats

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        env,
        state_space,
        n_episodes: int = 5,
        deterministic: bool = True,
        account_report: bool = False,
        report_every: int = 1,
        top_k_holdings: int = 0,
    ) -> List[List[float]]:
        self.model.eval()
        all_values: List[List[float]] = []
        effective = n_episodes
        if deterministic and not getattr(env, "random_start", True):
            effective = 1

        eps = 0.0 if deterministic else 0.05
        report_every = max(int(report_every), 1)

        tickers = list(getattr(state_space, "tickers", []))
        if not tickers:
            tickers = [f"asset_{i}" for i in range(getattr(env, "n_stocks", 0))]

        with torch.no_grad():
            for episode_idx in range(effective):
                obs, info = env.reset()
                done = False
                pv = [env.portfolio_value]
                prev_snapshot = None
                step_counter = 0

                if account_report:
                    print(f"\n=== ACCOUNT REPORT | episode {episode_idx + 1}/{effective} ===")
                    text, prev_snapshot = self._format_account_snapshot(
                        info=info,
                        tickers=tickers,
                        prev_snapshot=prev_snapshot,
                        top_k_holdings=int(top_k_holdings),
                    )
                    print(text)

                while not done:
                    ms, ps = state_space.flat_obs_to_sequential(obs)
                    action = self.select_action(ms, ps, epsilon=eps)
                    obs, _reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    step_counter += 1

                    if account_report and (done or step_counter % report_every == 0):
                        text, prev_snapshot = self._format_account_snapshot(
                            info=info,
                            tickers=tickers,
                            prev_snapshot=prev_snapshot,
                            top_k_holdings=int(top_k_holdings),
                        )
                        print(text)

                    pv.append(env.portfolio_value)

                all_values.append(pv)

        return all_values

    # ------------------------------------------------------------------
    # LR
    # ------------------------------------------------------------------

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(lr)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "n_updates": self.n_updates,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.target_model.load_state_dict(ckpt["target_model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint không khớp với kiến trúc model hiện tại. "
                "Hãy khởi tạo agent/model từ đúng config của run đã train rồi load lại. "
                f"Checkpoint: {path}\n\nChi tiết gốc:\n{exc}"
            ) from exc
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.n_updates = int(ckpt.get("n_updates", 0))