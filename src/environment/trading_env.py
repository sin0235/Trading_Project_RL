import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional

from src.environment.state_space import StateSpace
from src.environment.action_space import (
    decode_discrete_action,
    decode_continuous_action,
    apply_constraints,
)
from src.environment.reward_function import RewardFunction
from src.constants import WINDOW_SIZE, DATA_PATH, FEATURES


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment cho thi truong chung khoan Viet Nam.
    Ho tro 2 che do:
        - "discrete": DQN (moi buoc chon 1 trong K*N hanh dong)
        - "continuous": PPO (output vector [-1,1]^N)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tickers: List[str],
        mode: str = "discrete",
        initial_balance: float = 1_000_000_000,
        max_shares: int = 100,
        fee_rate: float = 0.0015,
        reward_type: str = "simple",
        window_size: int = WINDOW_SIZE,
        data_path: str = DATA_PATH,
        features: List[str] = FEATURES,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.mode = mode
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.fee_rate = fee_rate
        self.render_mode = render_mode

        self.state_space = StateSpace(
            tickers=tickers,
            window_size=window_size,
            data_path=data_path,
            features=features,
            mode="flatten",
        )
        self.reward_fn = RewardFunction(reward_type=reward_type)

        self.n_stocks = self.state_space.n_stocks
        self.k = 3

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=self.state_space.observation_shape,
            dtype=np.float32,
        )

        if self.mode == "discrete":
            self.action_space = spaces.Discrete(self.k * self.n_stocks)
        elif self.mode == "continuous":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.n_stocks,),
                dtype=np.float32,
            )

        self.t = None
        self.cash = None
        self.holdings = None
        self.portfolio_value = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.state_space.window_size - 1
        self.cash = float(self.initial_balance)
        self.holdings = np.zeros(self.n_stocks, dtype=np.float64)
        self.reward_fn.reset()

        prices = self.state_space.get_prices(self.t)
        self.portfolio_value = self.cash + np.sum(self.holdings * prices)

        obs = self.state_space.get_state(self.t, self.cash, self.holdings)
        info = self._build_info(prices)

        return obs, info

    def step(self, action):
        prices = self.state_space.get_prices(self.t)
        v_old = self.cash + np.sum(self.holdings * prices)

        if self.mode == "discrete":
            trade_amounts = decode_discrete_action(
                action, self.n_stocks, self.max_shares, self.k
            )
        else:
            trade_amounts = decode_continuous_action(action, self.max_shares)

        trade_amounts = apply_constraints(
            trade_amounts, self.cash, self.holdings, prices, self.fee_rate
        )

        total_fees = self._execute_trades(trade_amounts, prices)

        self.t += 1

        new_prices = self.state_space.get_prices(self.t)
        v_new = self.cash + np.sum(self.holdings * new_prices)
        self.portfolio_value = v_new

        reward = self.reward_fn.calculate(v_old, v_new)

        obs = self.state_space.get_state(self.t, self.cash, self.holdings)

        terminated = self.t >= self.state_space.n_days - 1 or v_new <= 0
        truncated = False

        info = self._build_info(new_prices, trade_amounts, total_fees)

        return obs, float(reward), terminated, truncated, info

    def _execute_trades(self, trade_amounts: np.ndarray,
                        prices: np.ndarray) -> float:
        """Thuc hien giao dich: ban truoc, mua sau. Tra ve tong phi."""
        total_fees = 0.0

        for i in range(self.n_stocks):
            if trade_amounts[i] < 0:
                shares = abs(trade_amounts[i])
                value = shares * prices[i]
                fee = value * self.fee_rate
                self.cash += value - fee
                self.holdings[i] -= shares
                total_fees += fee

        for i in range(self.n_stocks):
            if trade_amounts[i] > 0:
                shares = trade_amounts[i]
                value = shares * prices[i]
                fee = value * self.fee_rate
                cost = value + fee
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings[i] += shares
                    total_fees += fee

        return total_fees

    def _build_info(self, prices, trades=None, fees=0.0):
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "prices": prices.copy(),
            "trades": trades if trades is not None else np.zeros(self.n_stocks, dtype=np.int32),
            "fees": fees,
            "date": str(self.state_space.dates[self.t]),
            "step": self.t - (self.state_space.window_size - 1),
        }
