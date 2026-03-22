import unittest

import numpy as np

from src.environment.reward_function import (
    AdvancedRewardFunction,
    SharpeRewardFunction,
    SharpePlusRewardFunction,
    build_reward_function,
)


class RewardFunctionLogicTests(unittest.TestCase):
    def test_legacy_reward_is_still_available(self):
        reward_fn = AdvancedRewardFunction()
        reward = reward_fn.calculate(100.0, 101.0, trade_amounts=np.array([10.0]))
        self.assertTrue(np.isfinite(reward))

    # --- SharpeRewardFunction tests ---

    def test_sharpe_reward_builds_from_factory(self):
        reward_fn = build_reward_function("sharpe", window=10)
        self.assertIsInstance(reward_fn, SharpeRewardFunction)

    def test_sharpe_reward_positive_for_outperformance(self):
        reward_fn = SharpeRewardFunction(window=5)
        reward = reward_fn.calculate(
            v_old=100.0,
            v_new=120.0,
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([11.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )
        self.assertGreater(reward, 0.0)

    def test_sharpe_reward_penalizes_turnover(self):
        reward_fn = SharpeRewardFunction(window=5)
        low_turnover = reward_fn.calculate(
            v_old=100.0,
            v_new=110.0,
            trade_amounts=np.array([0.0, 0.0], dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([11.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        reward_fn.reset()
        high_turnover = reward_fn.calculate(
            v_old=100.0,
            v_new=110.0,
            trade_amounts=np.array([200.0, 100.0], dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([11.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        self.assertLess(high_turnover, low_turnover)

    def test_sharpe_reward_rolling_window_fills_up(self):
        """After filling the window, Sharpe component should be active and finite."""
        reward_fn = SharpeRewardFunction(window=3)
        rewards = []
        for i in range(5):
            r = reward_fn.calculate(
                v_old=100.0 + i,
                v_new=101.0 + i,
                trade_amounts=np.zeros(2, dtype=np.float32),
                execution_prices=np.array([10.0, 10.0], dtype=np.float32),
                next_prices=np.array([10.1, 10.0], dtype=np.float32),
                post_trade_value=100.0 + i,
            )
            rewards.append(r)
            self.assertTrue(np.isfinite(r))
        # All rewards should be finite after window fills
        self.assertEqual(len(rewards), 5)

    def test_sharpe_reward_returns_negative_for_bankruptcy(self):
        reward_fn = SharpeRewardFunction()
        reward = reward_fn.calculate(v_old=0.0, v_new=100.0)
        self.assertEqual(reward, -5.0)

    # --- SharpePlusRewardFunction tests ---

    def test_sharpe_plus_builds_from_factory(self):
        reward_fn = build_reward_function("sharpe_plus", window=10)
        self.assertIsInstance(reward_fn, SharpePlusRewardFunction)

    def test_sharpe_plus_builds_from_factory_alias(self):
        reward_fn = build_reward_function("sharpeplus")
        self.assertIsInstance(reward_fn, SharpePlusRewardFunction)

    def test_sharpe_plus_positive_for_outperformance(self):
        reward_fn = SharpePlusRewardFunction(window=5)
        reward = reward_fn.calculate(
            v_old=100.0,
            v_new=120.0,
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([11.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )
        self.assertGreater(reward, 0.0)

    def test_sharpe_plus_holding_bonus_rewards_low_turnover(self):
        """Agent giữ vị thế (no trade) nên được thưởng hơn agent giao dịch nhiều."""
        reward_fn = SharpePlusRewardFunction(window=5, holding_scale=1.0)
        low_turnover = reward_fn.calculate(
            v_old=100.0,
            v_new=110.0,
            trade_amounts=np.array([0.0, 0.0], dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([11.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        reward_fn.reset()
        high_turnover = reward_fn.calculate(
            v_old=100.0,
            v_new=110.0,
            trade_amounts=np.array([200.0, 100.0], dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([11.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        self.assertGreater(low_turnover, high_turnover)

    def test_sharpe_plus_momentum_alignment_positive_when_same_direction(self):
        """Momentum bonus dương khi portfolio và benchmark cùng chiều."""
        # Tách riêng momentum component bằng cách set excess_scale=0 và
        # dùng momentum_scale lớn
        reward_fn = SharpePlusRewardFunction(
            window=5, momentum_scale=5.0, excess_scale=0.0,
            sharpe_scale=0.0, drawdown_scale=0.0, turnover_scale=0.0,
            holding_scale=0.0,
        )
        # Cùng chiều: portfolio tăng 5%, benchmark tăng ~5%
        reward_aligned = reward_fn.calculate(
            v_old=100.0, v_new=105.0,
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([10.5, 10.5], dtype=np.float32),
            post_trade_value=100.0,
        )

        reward_fn.reset()
        # Ngược chiều: portfolio tăng 5%, benchmark giảm ~5%
        reward_misaligned = reward_fn.calculate(
            v_old=100.0, v_new=105.0,
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([9.5, 9.5], dtype=np.float32),
            post_trade_value=100.0,
        )

        self.assertGreater(reward_aligned, reward_misaligned)

    def test_sharpe_plus_asymmetric_drawdown_escalates(self):
        """Drawdown vượt ngưỡng bị phạt nặng hơn so với drawdown nhỏ."""
        # Trường hợp 1: drawdown nhỏ (dưới ngưỡng)
        reward_fn = SharpePlusRewardFunction(
            window=5, drawdown_scale=5.0, dd_threshold=0.05, dd_escalation=3.0,
        )
        # Đẩy max_portfolio_value lên 100
        reward_fn.max_portfolio_value = 100.0
        small_dd = reward_fn.calculate(
            v_old=100.0,
            v_new=97.0,  # drawdown = 3%
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([10.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        # Trường hợp 2: drawdown lớn (vượt ngưỡng)
        reward_fn.reset()
        reward_fn.max_portfolio_value = 100.0
        large_dd = reward_fn.calculate(
            v_old=100.0,
            v_new=85.0,  # drawdown = 15%
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([10.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        self.assertGreater(small_dd, large_dd)

    def test_sharpe_plus_returns_negative_for_bankruptcy(self):
        reward_fn = SharpePlusRewardFunction()
        reward = reward_fn.calculate(v_old=0.0, v_new=100.0)
        self.assertEqual(reward, -5.0)

    def test_sharpe_plus_rolling_window_fills_up(self):
        """After filling the window, reward should be finite."""
        reward_fn = SharpePlusRewardFunction(window=3)
        rewards = []
        for i in range(5):
            r = reward_fn.calculate(
                v_old=100.0 + i,
                v_new=101.0 + i,
                trade_amounts=np.zeros(2, dtype=np.float32),
                execution_prices=np.array([10.0, 10.0], dtype=np.float32),
                next_prices=np.array([10.1, 10.0], dtype=np.float32),
                post_trade_value=100.0 + i,
            )
            rewards.append(r)
            self.assertTrue(np.isfinite(r))
        self.assertEqual(len(rewards), 5)

    def test_build_reward_function_unsupported_raises(self):
        with self.assertRaises(ValueError):
            build_reward_function("nonexistent_reward")

    def test_build_reward_function_default_is_sharpe(self):
        reward_fn = build_reward_function()
        self.assertIsInstance(reward_fn, SharpeRewardFunction)


if __name__ == "__main__":
    unittest.main()
