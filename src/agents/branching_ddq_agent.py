"""
Branching Double Deep Q-Learning Agent cho discrete vector action per-stock.

Thiết kế:
    - ReplayBuffer lưu action dạng vector độ dài n_stocks
    - Online + target network dùng BranchingDRQNNetwork
    - Double DQN target trên từng branch
    - Loss Huber trung bình trên toàn bộ ma trận Q (batch, n_stocks)
"""

import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.lstm import BranchingDRQNNetwork

np.random.seed(42)
torch.manual_seed(42)


class BranchingReplayBuffer:
    """Ring buffer numpy cho transitions của Branching DDQ."""

    def __init__(
        self,
        capacity: int,
        seq_len: int,
        market_feat_dim: int,
        portfolio_dim: int,
        n_stocks: int,
    ):
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self.market_feat_dim = int(market_feat_dim)
        self.portfolio_dim = int(portfolio_dim)
        self.n_stocks = int(n_stocks)

        self._ptr = 0
        self.size = 0

        self.market = np.zeros(
            (self.capacity, self.seq_len, self.market_feat_dim), dtype=np.float32
        )
        self.portfolio = np.zeros((self.capacity, self.portfolio_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.n_stocks), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_market = np.zeros(
            (self.capacity, self.seq_len, self.market_feat_dim), dtype=np.float32
        )
        self.next_portfolio = np.zeros((self.capacity, self.portfolio_dim), dtype=np.float32)
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
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_stocks,):
            raise ValueError(f"Branching action shape {action.shape} != ({self.n_stocks},)")

        i = self._ptr
        self.market[i] = market_state
        self.portfolio[i] = portfolio_state
        self.actions[i] = action
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

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.max_grad_norm = float(max_grad_norm)

        seq_len = model.seq_len
        market_feat_dim = model.n_stocks * model.n_features
        portfolio_dim = 1 + model.n_stocks

        self.buffer = BranchingReplayBuffer(
            capacity=replay_buffer_size,
            seq_len=seq_len,
            market_feat_dim=market_feat_dim,
            portfolio_dim=portfolio_dim,
            n_stocks=model.n_stocks,
        )

        self.total_steps = 0
        self.n_updates = 0
        self._train_step_counter = 0

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
        sum_loss = 0.0
        n = 0
        for _ in range(self.gradient_steps):
            stats = self._update_once()
            sum_loss += stats["loss"]
            n += 1
        self.model.eval()

        self.n_updates += n
        return {"loss": sum_loss / max(n, 1), "n_gradient_steps": n}

    def _update_once(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.device)

        ms = batch["market_states"]
        ps = batch["portfolio_states"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(1)
        next_ms = batch["next_market_states"]
        next_ps = batch["next_portfolio_states"]
        dones = batch["dones"].unsqueeze(1)

        with torch.no_grad():
            next_q_online, _ = self.model(next_ms, next_ps, hidden=None)
            next_actions = next_q_online.argmax(dim=-1, keepdim=True)
            next_q_target, _ = self.target_model(next_ms, next_ps, hidden=None)
            next_q = next_q_target.gather(-1, next_actions).squeeze(-1)
            target = rewards + self.gamma * (1.0 - dones) * next_q

        current_q, _ = self.model(ms, ps, hidden=None)
        current_q = current_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        loss = nn.functional.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {"loss": float(loss.item())}

    def evaluate(
        self,
        env,
        state_space,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> List[List[float]]:
        self.model.eval()
        all_values: List[List[float]] = []
        effective = n_episodes
        if deterministic and not getattr(env, "random_start", True):
            effective = 1

        eps = 0.0 if deterministic else 0.05

        with torch.no_grad():
            for _ in range(effective):
                obs, _ = env.reset()
                done = False
                pv = [env.portfolio_value]

                while not done:
                    ms, ps = state_space.flat_obs_to_sequential(obs)
                    action = self.select_action(ms, ps, epsilon=eps)
                    obs, _reward, terminated, truncated, _info = env.step(action)
                    done = terminated or truncated
                    pv.append(env.portfolio_value)

                all_values.append(pv)

        return all_values

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(lr)

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
                "Checkpoint không khớp với kiến trúc Branching DDQ hiện tại. "
                "Hãy khởi tạo agent/model từ đúng config của run đã train rồi load lại. "
                f"Checkpoint: {path}\n\nChi tiết gốc:\n{exc}"
            ) from exc
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.n_updates = int(ckpt.get("n_updates", 0))

