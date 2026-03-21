import unittest

import numpy as np
import pandas as pd
import torch

from src.agents.ppo_agent import PPOAgent
from src.environment.trading_env import TradingEnv
from src.models.lstm import DRQNNetwork, LSTMFeatureExtractor, PPOLSTMActorCritic
from src.training.PPO import (
    average_metrics,
    build_baseline_comparison,
    evaluate_baselines,
)


def make_price_frame(
    n_days: int,
    close_start: float = 10.0,
    open_values: np.ndarray | None = None,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = np.linspace(close_start, close_start + n_days - 1, n_days, dtype=np.float32)
    if open_values is None:
        open_values = close.copy()
    return pd.DataFrame(
        {
            "time": dates,
            "open": np.asarray(open_values, dtype=np.float32),
            "close": close,
            "close_norm": np.linspace(-1.0, 1.0, n_days, dtype=np.float32),
            "return_1d": np.zeros(n_days, dtype=np.float32),
            "return_5d": np.zeros(n_days, dtype=np.float32),
            "macd": np.zeros(n_days, dtype=np.float32),
            "rsi": np.zeros(n_days, dtype=np.float32),
            "adx": np.zeros(n_days, dtype=np.float32),
            "volume_norm": np.zeros(n_days, dtype=np.float32),
        }
    )


def make_data_dict(n_days: int, tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {
        ticker: make_price_frame(n_days, close_start=10.0 + idx * 5.0)
        for idx, ticker in enumerate(tickers)
    }


class PPOLSTMLogicTests(unittest.TestCase):
    def test_dirichlet_policy_outputs_simplex_and_recomputes_logprob(self):
        torch.manual_seed(0)
        model = PPOLSTMActorCritic(n_stocks=3, n_features=7, seq_len=30)
        model.eval()

        market = torch.randn(8, 30, 21)
        portfolio = torch.randn(8, 4)

        action, action_for_buffer, log_prob, value, _ = model.get_action(market, portfolio)
        recomputed_log_prob, entropy, recomputed_value, _ = model.evaluate_actions(
            market, portfolio, action_for_buffer
        )

        self.assertTrue(torch.all(action >= 0.0).item())
        self.assertTrue(torch.allclose(action.sum(dim=-1), torch.ones(8), atol=1e-5))
        self.assertTrue(torch.allclose(log_prob, recomputed_log_prob, atol=1e-5))
        self.assertTrue(torch.allclose(value, recomputed_value, atol=1e-5))
        self.assertTrue(torch.isfinite(entropy).all().item())

    def test_non_contiguous_4d_inputs_are_supported(self):
        torch.manual_seed(0)
        ppo_model = PPOLSTMActorCritic(n_stocks=4, n_features=3, seq_len=5)
        drqn_model = DRQNNetwork(n_stocks=4, n_features=3, seq_len=5)

        market = torch.randn(2, 4, 5, 3).transpose(1, 2)
        portfolio = torch.randn(2, 5)

        concentration, value, _ = ppo_model.forward(market, portfolio)
        q_values, _ = drqn_model.forward(market, portfolio)

        self.assertEqual(concentration.shape, (2, 5))
        self.assertEqual(value.shape, (2, 1))
        self.assertEqual(q_values.shape, (2, 12))

    def test_forget_gate_effective_bias_is_one(self):
        extractor = LSTMFeatureExtractor(input_size=12, hidden_size=8, num_layers=2, dropout=0.1)

        bias_ih = extractor.lstm.bias_ih_l0.detach()
        bias_hh = extractor.lstm.bias_hh_l0.detach()
        n = bias_ih.numel()
        effective_forget_bias = bias_ih[n // 4:n // 2] + bias_hh[n // 4:n // 2]

        self.assertTrue(torch.allclose(effective_forget_bias, torch.ones_like(effective_forget_bias), atol=1e-6))

    def test_ppo_agent_collect_rollout_and_update_smoke(self):
        torch.manual_seed(0)
        tickers = ("AAA", "BBB")
        env = TradingEnv(
            tickers=list(tickers),
            mode="continuous",
            data_dict=make_data_dict(50, tickers),
            window_size=30,
            random_start=False,
            max_steps=5,
        )

        model = PPOLSTMActorCritic(n_stocks=2, n_features=7, seq_len=30)
        agent = PPOAgent(
            model=model,
            lr=1e-3,
            n_epochs=1,
            batch_size=4,
            target_kl=None,
            device="cpu",
        )

        obs, episode_infos = agent.collect_rollout(env, env.state_space, n_steps=8)
        update_stats = agent.update()

        self.assertIsNotNone(obs)
        self.assertIsInstance(episode_infos, list)
        self.assertEqual(len(agent.buffer.actions[0]), env.n_stocks + 1)
        self.assertTrue(np.isfinite(update_stats["policy_loss"]))
        self.assertTrue(np.isfinite(update_stats["value_loss"]))
        self.assertTrue(np.isfinite(update_stats["entropy"]))
        self.assertTrue(agent.model.training)

    def test_collect_rollout_preserves_episode_reward_across_calls(self):
        torch.manual_seed(0)
        tickers = ("AAA", "BBB")
        env = TradingEnv(
            tickers=list(tickers),
            mode="continuous",
            data_dict=make_data_dict(60, tickers),
            window_size=30,
            random_start=False,
            max_steps=5,
            reward_scaling=1.0,
        )
        env.reward_fn.calculate = lambda v_old, v_new, trade_amounts=None: 1.0

        model = PPOLSTMActorCritic(n_stocks=2, n_features=7, seq_len=30)
        agent = PPOAgent(
            model=model,
            lr=1e-3,
            n_epochs=1,
            batch_size=4,
            target_kl=None,
            device="cpu",
        )

        obs, ep_infos_first = agent.collect_rollout(env, env.state_space, n_steps=3)
        _, ep_infos_second = agent.collect_rollout(env, env.state_space, n_steps=3, obs=obs)

        self.assertEqual(ep_infos_first, [])
        self.assertEqual(len(ep_infos_second), 1)
        self.assertEqual(ep_infos_second[0]["steps"], 5)
        self.assertEqual(ep_infos_second[0]["total_reward"], 5.0)

    def test_collect_rollout_reset_seed_is_reproducible(self):
        torch.manual_seed(0)
        tickers = ("AAA", "BBB")
        model = PPOLSTMActorCritic(n_stocks=2, n_features=7, seq_len=30)
        agent = PPOAgent(
            model=model,
            lr=1e-3,
            n_epochs=1,
            batch_size=4,
            target_kl=None,
            device="cpu",
        )

        env1 = TradingEnv(
            tickers=list(tickers),
            mode="continuous",
            data_dict=make_data_dict(60, tickers),
            window_size=30,
            random_start=True,
            max_steps=5,
        )
        env2 = TradingEnv(
            tickers=list(tickers),
            mode="continuous",
            data_dict=make_data_dict(60, tickers),
            window_size=30,
            random_start=True,
            max_steps=5,
        )

        agent.collect_rollout(env1, env1.state_space, n_steps=1, reset_seed=123)
        start_date_1 = env1.date_memory[0]
        agent.collect_rollout(env2, env2.state_space, n_steps=1, reset_seed=123)
        start_date_2 = env2.date_memory[0]

        self.assertEqual(start_date_1, start_date_2)

    def test_evaluate_avoids_duplicate_fixed_start_episodes(self):
        torch.manual_seed(0)
        tickers = ("AAA", "BBB")
        env = TradingEnv(
            tickers=list(tickers),
            mode="continuous",
            data_dict=make_data_dict(60, tickers),
            window_size=30,
            random_start=False,
            max_steps=5,
        )
        model = PPOLSTMActorCritic(n_stocks=2, n_features=7, seq_len=30)
        agent = PPOAgent(
            model=model,
            lr=1e-3,
            n_epochs=1,
            batch_size=4,
            target_kl=None,
            device="cpu",
        )

        values = agent.evaluate(env, env.state_space, n_episodes=3, deterministic=True)

        self.assertEqual(len(values), 1)

    def test_average_metrics_aggregates_all_keys(self):
        metrics = average_metrics(
            [
                {"a": 1.0, "b": 3.0},
                {"a": 3.0, "b": 5.0, "c": 7.0},
            ]
        )

        self.assertEqual(metrics["a"], 2.0)
        self.assertEqual(metrics["b"], 4.0)
        self.assertEqual(metrics["c"], 7.0)

    def test_evaluate_baselines_returns_comparable_metrics(self):
        tickers = ("AAA", "BBB")
        env = TradingEnv(
            tickers=list(tickers),
            mode="continuous",
            data_dict=make_data_dict(60, tickers),
            window_size=30,
            random_start=False,
            max_steps=5,
        )

        baselines = evaluate_baselines(env, env.initial_balance)
        comparison = build_baseline_comparison(
            {"total_return": 0.2, "sharpe_ratio": 1.0, "max_drawdown": 0.1},
            baselines["equal_weight"],
        )

        self.assertIn("equal_weight", baselines)
        self.assertIn("buy_and_hold_equal_weight", baselines)
        self.assertIn("delta_total_return", comparison)
        self.assertIn("delta_sharpe_ratio", comparison)


if __name__ == "__main__":
    unittest.main()
