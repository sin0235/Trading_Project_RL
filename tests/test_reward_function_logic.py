import unittest

import numpy as np

from src.environment.reward_function import AdvancedRewardFunction, TmpRewardFunction, SharpeRewardFunction, build_reward_function


class RewardFunctionLogicTests(unittest.TestCase):
    def test_legacy_reward_is_still_available(self):
        reward_fn = AdvancedRewardFunction()
        reward = reward_fn.calculate(100.0, 101.0, trade_amounts=np.array([10.0]))
        self.assertTrue(np.isfinite(reward))

    def test_tmp_reward_prefers_excess_return_over_equal_weight_baseline(self):
        reward_fn = TmpRewardFunction()
        better = reward_fn.calculate(
            v_old=100.0,
            v_new=120.0,
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([12.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        reward_fn.reset()
        worse = reward_fn.calculate(
            v_old=100.0,
            v_new=105.0,
            trade_amounts=np.zeros(2, dtype=np.float32),
            execution_prices=np.array([10.0, 10.0], dtype=np.float32),
            next_prices=np.array([12.0, 10.0], dtype=np.float32),
            post_trade_value=100.0,
        )

        self.assertGreater(better, worse)

    def test_tmp_reward_penalizes_turnover_in_notional_terms(self):
        reward_fn = TmpRewardFunction()
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


if __name__ == "__main__":
    unittest.main()
