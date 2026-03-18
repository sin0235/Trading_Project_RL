import unittest

import numpy as np
import pandas as pd
import torch

from src.agents.ppo_agent import PPOAgent
from src.environment.trading_env import TradingEnv
from src.models.lstm import DRQNNetwork, LSTMFeatureExtractor, PPOLSTMActorCritic


def make_price_frame(n_days: int, close_start: float = 10.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    close = np.linspace(close_start, close_start + n_days - 1, n_days, dtype=np.float32)
    return pd.DataFrame(
        {
            "time": dates,
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


if __name__ == "__main__":
    unittest.main()
