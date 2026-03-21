import unittest

import numpy as np

from src.environment.reward_function import AdvancedRewardFunction, TmpRewardFunction


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


if __name__ == "__main__":
    unittest.main()
