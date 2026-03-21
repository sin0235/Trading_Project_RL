"""
PPO Agent với LSTM cho continuous trading.

Thiết kế:
    - RolloutBuffer: lưu transitions, tính GAE
    - PPOAgent: thu thập rollout + PPO clipped update + evaluate + save/load

Mỗi observation từ env (mode=flatten) được chuyển qua flat_obs_to_sequential()
thành (market_state, portfolio_state) cho LSTM.

Update dùng stateless hidden (zero-init) để hỗ trợ mini-batch song song.
Rollout collection cũng dùng stateless hidden để khớp hoàn toàn với PPO update.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Any
from pathlib import Path

from src.models.lstm import PPOLSTMActorCritic


class RolloutBuffer:

    def __init__(self):
        self.market_states: List[np.ndarray] = []
        self.portfolio_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(self, market_state, portfolio_state, action, log_prob, reward, done, value):
        self.market_states.append(market_state)
        self.portfolio_states.append(portfolio_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "market_states": torch.tensor(np.array(self.market_states), dtype=torch.float32, device=device),
            "portfolio_states": torch.tensor(np.array(self.portfolio_states), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.float32, device=device),
            "old_log_probs": torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(self.returns, dtype=torch.float32, device=device),
        }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class PPOAgent:

    def __init__(
        self,
        model: PPOLSTMActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.03,
        n_epochs: int = 10,
        batch_size: int = 256,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.buffer = RolloutBuffer()
        self._episode_reward = 0.0

        self.total_steps = 0
        self.n_updates = 0

    # ------------------------------------------------------------------
    # Rollout Collection
    # ------------------------------------------------------------------

    def collect_rollout(self, env, state_space, n_steps: int, obs=None,
                        reset_seed: Optional[int] = None):
        """
        Thu thập n_steps transitions.
        PPO hiện dùng LSTM như một sequence encoder trên từng observation window,
        nên rollout/update đều dùng hidden=None để objective nhất quán.

        Returns:
            obs: observation cuối cùng
            episode_infos: list dict thông tin mỗi episode kết thúc trong rollout
        """
        self.buffer.clear()
        self.model.eval()

        if obs is None:
            obs, _ = env.reset(seed=reset_seed)
            self._episode_reward = 0.0

        episode_infos = []

        with torch.no_grad():
            for _ in range(n_steps):
                market_state, portfolio_state = state_space.flat_obs_to_sequential(obs)

                ms_t = torch.tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                ps_t = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device).unsqueeze(0)

                action, action_for_buffer, log_prob, value, _ = self.model.get_action(
                    ms_t, ps_t, hidden=None
                )

                action_np = action.cpu().numpy().squeeze(0)
                action_buffer_np = action_for_buffer.cpu().numpy().squeeze(0)
                log_prob_val = log_prob.cpu().item()
                value_val = value.cpu().item()

                new_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                self.total_steps += 1
                self._episode_reward += reward

                self.buffer.add(
                    market_state=market_state,
                    portfolio_state=portfolio_state,
                    action=action_buffer_np,
                    log_prob=log_prob_val,
                    reward=reward,
                    done=done,
                    value=value_val,
                )

                if done:
                    episode_infos.append({
                        "total_reward": self._episode_reward,
                        "portfolio_value": info.get("portfolio_value", 0),
                        "total_return": (info.get("portfolio_value", 0) - env.initial_balance) / env.initial_balance,
                        "n_trades": env.trades,
                        "total_cost": env.cost,
                        "steps": env.current_step,
                    })
                    self._episode_reward = 0.0
                    obs, _ = env.reset()
                else:
                    obs = new_obs

            # Bootstrap: V(s_T) cho GAE
            market_state, portfolio_state = state_space.flat_obs_to_sequential(obs)
            ms_t = torch.tensor(market_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            ps_t = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value, _ = self.model.forward(ms_t, ps_t, hidden=None)
            last_value = last_value.cpu().item()

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        return obs, episode_infos

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        PPO clipped update trên buffer. Mini-batch shuffle, stateless hidden.
        Trả về dict training metrics.
        """
        # PPO update phải chạy ở train mode để cuDNN cho phép backward qua LSTM.
        # rollout/evaluate vẫn dùng eval() nên hành vi suy luận không đổi.
        self.model.train()
        data = self.buffer.get_tensors(self.device)

        n = len(self.buffer)
        advantages = data["advantages"]
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        sum_entropy = 0.0
        sum_approx_kl = 0.0
        sum_clip_frac = 0.0
        n_mini_batches = 0
        early_stop = False

        for epoch in range(self.n_epochs):
            if early_stop:
                break

            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                mb_market = data["market_states"][idx]
                mb_portfolio = data["portfolio_states"][idx]
                mb_actions = data["actions"][idx]
                mb_old_lp = data["old_log_probs"][idx]
                mb_adv = advantages[idx]
                mb_ret = data["returns"][idx]

                new_lp, entropy, new_val, _ = self.model.evaluate_actions(
                    mb_market, mb_portfolio, mb_actions, hidden=None
                )
                new_val = new_val.squeeze(-1)

                # PPO clipped surrogate
                log_ratio = new_lp - mb_old_lp
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_val, mb_ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()

                sum_policy_loss += policy_loss.item()
                sum_value_loss += value_loss.item()
                sum_entropy += (-entropy_loss.item())
                sum_approx_kl += approx_kl
                sum_clip_frac += clip_frac
                n_mini_batches += 1

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

        self.n_updates += n_mini_batches
        d = max(n_mini_batches, 1)

        return {
            "policy_loss": sum_policy_loss / d,
            "value_loss": sum_value_loss / d,
            "entropy": sum_entropy / d,
            "approx_kl": sum_approx_kl / d,
            "clip_fraction": sum_clip_frac / d,
            "n_mini_batches": n_mini_batches,
            "early_stop_epoch": epoch if early_stop else self.n_epochs,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env, state_space, n_episodes: int = 5,
                 deterministic: bool = True) -> List[List[float]]:
        """
        Chạy n_episodes đánh giá. Trả về list các chuỗi portfolio_values.
        Nếu env có fixed start và chạy deterministic, nhiều episode sẽ cho cùng
        một trajectory; khi đó chỉ chạy một lần để tránh metric lặp lại.
        """
        self.model.eval()
        all_values = []
        effective_episodes = n_episodes
        if deterministic and not getattr(env, "random_start", True):
            effective_episodes = 1

        with torch.no_grad():
            for _ in range(effective_episodes):
                obs, _ = env.reset()
                done = False
                pv = [env.portfolio_value]

                while not done:
                    ms, ps = state_space.flat_obs_to_sequential(obs)
                    ms_t = torch.tensor(ms, dtype=torch.float32, device=self.device).unsqueeze(0)
                    ps_t = torch.tensor(ps, dtype=torch.float32, device=self.device).unsqueeze(0)

                    if deterministic:
                        concentration, _, _ = self.model.forward(ms_t, ps_t, hidden=None)
                        action = concentration / concentration.sum(dim=-1, keepdim=True)
                    else:
                        action, _, _, _, _ = self.model.get_action(ms_t, ps_t, hidden=None)

                    action_np = action.cpu().numpy().squeeze(0)
                    obs, reward, terminated, truncated, info = env.step(action_np)
                    done = terminated or truncated
                    pv.append(env.portfolio_value)

                all_values.append(pv)

        return all_values

    # ------------------------------------------------------------------
    # LR control
    # ------------------------------------------------------------------

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "n_updates": self.n_updates,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint không khớp với kiến trúc model hiện tại. "
                "Hãy khởi tạo agent/model từ đúng config của run đã train rồi load lại. "
                f"Checkpoint: {path}\n\nChi tiết gốc:\n{exc}"
            ) from exc
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.n_updates = ckpt.get("n_updates", 0)
