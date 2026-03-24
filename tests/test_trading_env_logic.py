import unittest

import numpy as np
import pandas as pd

from src.environment.reward_function import AdvancedRewardFunction, SharpeRewardFunction, SharpePlusRewardFunction
from src.environment.action_space import decode_discrete_action
from src.environment.trading_env import TradingEnv


def make_price_frame(
    n_days: int,
    close_start: float = 10.0,
    open_values: np.ndarray | None = None,
    close_values: np.ndarray | None = None,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    if close_values is None:
        close = np.linspace(close_start, close_start + n_days - 1, n_days, dtype=np.float32)
    else:
        close = np.asarray(close_values, dtype=np.float32)
    if open_values is None:
        open_values = close.copy()
    df = pd.DataFrame(
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
    return df


def make_data_dict(n_days: int, tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {
        ticker: make_price_frame(n_days, close_start=10.0 + idx)
        for idx, ticker in enumerate(tickers)
    }


class DummyReward:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.clear()

    def calculate(self, v_old, v_new, trade_amounts=None):
        stored = None if trade_amounts is None else np.array(trade_amounts, copy=True)
        self.calls.append((v_old, v_new, stored))
        return 1.0


class TradingEnvLogicTests(unittest.TestCase):
    def test_decode_discrete_action_large_holdings_no_int32_overflow(self):
        action = np.array([0], dtype=np.int64)
        holdings = np.array([2_467_816_000], dtype=np.int64)
        prices = np.array([10.0], dtype=np.float64)

        trade_amounts = decode_discrete_action(
            action=action,
            n_stocks=1,
            min_shares=100,
            cash=0.0,
            holdings=holdings,
            prices=prices,
            fee_rate=0.001,
        )

        self.assertEqual(trade_amounts.dtype, np.int64)
        self.assertEqual(int(trade_amounts[0]), -2_467_816_000)

    def test_reward_receives_trade_amounts(self):
        env = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict=make_data_dict(40, ("AAA",)),
            window_size=30,
            random_start=False,
            reward_scaling=1.0,
        )
        dummy_reward = DummyReward()
        env.reward_fn = dummy_reward

        env.reset()
        _, reward, terminated, truncated, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)
        self.assertEqual(len(dummy_reward.calls), 1)
        self.assertIsNotNone(dummy_reward.calls[0][2])
        self.assertTrue(np.any(np.abs(dummy_reward.calls[0][2]) > 0))

    def test_continuous_action_executes_at_next_open(self):
        open_values = np.array([10.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float32)
        env = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict={"AAA": make_price_frame(5, close_start=10.0, open_values=open_values)},
            window_size=3,
            random_start=False,
            min_shares=1,
            initial_balance=100.0,
            fee_rate=0.0,
            reward_scaling=1.0,
        )

        env.reward_fn.calculate = lambda v_old, v_new, trade_amounts=None: v_new - v_old

        env.reset()
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["execution_prices"][0], 20.0)
        self.assertEqual(env.holdings[0], 5)
        self.assertEqual(reward, -35.0)

    def test_continuous_action_deadband_skips_small_rebalance(self):
        n_days = 5
        env = TradingEnv(
            tickers=["AAA", "BBB"],
            mode="continuous",
            data_dict={
                "AAA": make_price_frame(
                    n_days,
                    close_values=np.array([10.0, 10.0, 10.0, 10.0, 10.1], dtype=np.float32),
                    open_values=np.array([10.0, 10.0, 10.0, 10.0, 10.1], dtype=np.float32),
                ),
                "BBB": make_price_frame(
                    n_days,
                    close_values=np.array([10.0, 10.0, 10.0, 10.0, 9.9], dtype=np.float32),
                    open_values=np.array([10.0, 10.0, 10.0, 10.0, 9.9], dtype=np.float32),
                ),
            },
            window_size=3,
            random_start=False,
            min_shares=1,
            initial_balance=100.0,
            fee_rate=0.0,
            reward_scaling=1.0,
            trade_deadband=0.01,
        )

        env.reward_fn.calculate = lambda v_old, v_new, trade_amounts=None: 0.0
        env.reset()
        env.step(np.array([0.5, 0.5, 0.0], dtype=np.float32))
        _, _, terminated, truncated, info = env.step(np.array([0.5, 0.5, 0.0], dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        np.testing.assert_array_equal(info["trades"], np.array([0, 0], dtype=np.int32))

    def test_continuous_action_max_weight_change_caps_trade_size(self):
        env = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict={"AAA": make_price_frame(5, close_start=10.0, open_values=np.full(5, 10.0, dtype=np.float32))},
            window_size=3,
            random_start=False,
            min_shares=1,
            initial_balance=100.0,
            fee_rate=0.0,
            reward_scaling=1.0,
            max_weight_change_per_step=0.2,
        )

        env.reward_fn.calculate = lambda v_old, v_new, trade_amounts=None: 0.0
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertEqual(info["trades"][0], 2)
        self.assertEqual(env.holdings[0], 2)

    def test_random_start_respects_episode_boundary_and_seed(self):
        env = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict=make_data_dict(40, ("AAA",)),
            window_size=30,
            max_steps=3,
            random_start=True,
        )

        max_valid_start = env.max_t - env.max_steps
        starts = []
        for seed in range(20):
            env.reset(seed=seed)
            starts.append(env.t)

        self.assertTrue(all(env.state_space.window_size - 1 <= t <= max_valid_start for t in starts))

        env.reset(seed=123)
        first_start = env.t
        env.reset(seed=123)
        self.assertEqual(first_start, env.t)

    def test_reset_accepts_explicit_start_t(self):
        env = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict=make_data_dict(40, ("AAA",)),
            window_size=30,
            max_steps=3,
            random_start=True,
        )

        env.reset(options={"start_t": 31})
        self.assertEqual(env.t, 31)

    def test_env_can_select_reward_function_by_name(self):
        env_sharpe = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict=make_data_dict(40, ("AAA",)),
            window_size=30,
            random_start=False,
            reward_name="sharpe",
            reward_kwargs={"window": 7},
        )
        env_sharpe_plus = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict=make_data_dict(40, ("AAA",)),
            window_size=30,
            random_start=False,
            reward_name="sharpe_plus",
            reward_kwargs={"window": 10},
        )
        env_advanced = TradingEnv(
            tickers=["AAA"],
            mode="continuous",
            data_dict=make_data_dict(40, ("AAA",)),
            window_size=30,
            random_start=False,
            reward_name="advanced",
            reward_kwargs={"window": 9},
        )

        self.assertIsInstance(env_sharpe.reward_fn, SharpeRewardFunction)
        self.assertIsInstance(env_sharpe_plus.reward_fn, SharpePlusRewardFunction)
        self.assertIsInstance(env_advanced.reward_fn, AdvancedRewardFunction)
        self.assertEqual(env_sharpe.reward_fn.window, 7)
        self.assertEqual(env_sharpe_plus.reward_fn.window, 10)
        self.assertEqual(env_advanced.reward_fn.window, 9)

    def test_short_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "window_size \\+ 1 common trading days"):
            TradingEnv(
                tickers=["AAA"],
                mode="continuous",
                data_dict=make_data_dict(30, ("AAA",)),
                window_size=30,
                random_start=False,
            )

    def test_multidiscrete_action_space_is_used(self):
        env = TradingEnv(
            tickers=["AAA", "BBB"],
            mode="MultiDiscrete",
            data_dict=make_data_dict(40, ("AAA", "BBB")),
            window_size=30,
            random_start=False,
        )

        self.assertEqual(tuple(env.action_space.nvec.tolist()), (env.k, env.k))

        env.reset()
        _, _, terminated, truncated, info = env.step(np.array([1, 2], dtype=np.int64))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["trades"].shape, (2,))
        self.assertEqual(info["trades"][0], 0)
        self.assertGreater(info["trades"][1], 0)

    def test_mode_discrete_is_alias_of_multidiscrete(self):
        env = TradingEnv(
            tickers=["AAA", "BBB"],
            mode="discrete",
            data_dict=make_data_dict(40, ("AAA", "BBB")),
            window_size=30,
            random_start=False,
        )

        self.assertEqual(env.mode, "MultiDiscrete")
        env.reset()
        _, _, _, _, info = env.step(np.array([1, 2], dtype=np.int64))

        self.assertEqual(info["trades"].shape, (2,))
        self.assertEqual(info["trades"][0], 0)
        self.assertGreater(info["trades"][1], 0)

    def test_multidiscrete_sell_all_liquidates_odd_lot(self):
        env = TradingEnv(
            tickers=["AAA"],
            mode="MultiDiscrete",
            data_dict=make_data_dict(40, ("AAA",)),
            window_size=30,
            random_start=False,
            min_shares=100,
            initial_balance=0.0,
            fee_rate=0.0,
            initial=False,
            previous_state=[0.0, 50, 10.0],
        )

        env.reward_fn.calculate = lambda v_old, v_new, trade_amounts=None: 0.0
        env.reset()
        _, _, _, _, info = env.step(np.array([0], dtype=np.int64))

        self.assertEqual(int(info["trades"][0]), -50)
        self.assertEqual(int(env.holdings[0]), 0)


if __name__ == "__main__":
    unittest.main()
