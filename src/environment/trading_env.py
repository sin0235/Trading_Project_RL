import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

from src.environment.state_space import StateSpace
from src.environment.action_space import (
    decode_discrete_action,
    decode_continuous_action,
    apply_constraints,
)
from src.environment.reward_function import build_reward_function
from src.constants import WINDOW_SIZE, DATA_PATH, FEATURES


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment cho thi truong chung khoan Viet Nam.
    Ho tro 2 che do:
        - "discrete": DRQN / DDQ (scalar action 0..K*N-1; xem src.agents.ddq_agent, src.training.DDQ)
        - "continuous": PPO (output vector [0,1]^(N+1), N stocks + 1 cash)

    Semantics nghien cuu:
        - Observation tai ngay t chi su dung du lieu den het ngay t
        - Action tai ngay t duoc khop o gia mo cua ngay t+1
        - Portfolio value / reward sau step duoc danh dau tai gia dong cua ngay t+1

    Cach nay loai bo same-bar look-ahead bias nhung van giu mo hinh daily backtest
    gon nhe, phu hop cho bai toan nghien cuu.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tickers: List[str],
        mode: str = "discrete",
        initial_balance: float = 1_000_000_000,
        min_shares: int = 100,
        fee_rate: float = 0.0015,
        window_size: int = WINDOW_SIZE,
        data_path: str = DATA_PATH,
        data_dict: Optional[dict] = None,
        features: List[str] = FEATURES,
        render_mode: Optional[str] = None,
        max_steps: int = 100,
        random_start: bool = True,
        reward_scaling: float = 1.0,
        reward_name: str = "sharpe",
        reward_kwargs: Optional[dict] = None,
        trade_deadband: float = 0.0,
        max_weight_change_per_step: float = 1.0,
        make_plots: bool = False,
        print_verbosity: int = 10,
        initial: bool = True,
        previous_state: Optional[list] = None,
        model_name: str = "",
        iteration: str = "",
    ):
        super().__init__()

        self.mode = mode
        self.initial_balance = initial_balance
        self.min_shares = min_shares
        self.fee_rate = fee_rate
        self.render_mode = render_mode
        self.reward_scaling = reward_scaling
        self.reward_name = reward_name
        self.reward_kwargs = dict(reward_kwargs or {})
        self.trade_deadband = float(trade_deadband)
        self.max_weight_change_per_step = float(max_weight_change_per_step)
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state or []
        self.model_name = model_name
        self.iteration = iteration

        if self.trade_deadband < 0:
            raise ValueError("trade_deadband phải >= 0.")
        if not (0 < self.max_weight_change_per_step <= 1.0):
            raise ValueError("max_weight_change_per_step phải trong khoảng (0, 1].")

        # Alias for FinRL compatibility
        self.initial_amount = self.initial_balance

        self.state_space = StateSpace(
            tickers=tickers,
            window_size=window_size,
            data_path=data_path,
            data_dict=data_dict,
            features=features,
            mode="flatten",
        )
        self.reward_fn = build_reward_function(self.reward_name, **self.reward_kwargs)

        self.n_stocks = self.state_space.n_stocks
        self.k = 3
        self.max_t = self.state_space.n_days - 1

        if self.state_space.max_steps < 1:
            raise ValueError(
                "TradingEnv requires at least window_size + 1 common trading days "
                f"(got n_days={self.state_space.n_days}, window_size={self.state_space.window_size})."
            )

        # Giới hạn số bước trên mỗi episode để không vượt quá độ dài dữ liệu
        # và tránh truy cập ngoài range khi hết data (vấn đề trading_env cũ gặp phải).
        self.max_steps = min(max_steps, self.state_space.max_steps)
        self.random_start = random_start
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=self.state_space.observation_shape,
            dtype=np.float32,
        )

        if self.mode == "discrete":
            self.action_space = spaces.Discrete(self.k * self.n_stocks)
        elif self.mode == "MultiDiscrete":
            self.action_space = spaces.MultiDiscrete([self.k] * self.n_stocks)
        elif self.mode == "continuous":
            self.action_space = spaces.Box(
                low=0, high=1.0,
                shape=(self.n_stocks + 1,),
                dtype=np.float32,
            )

        # Initialize state variables
        self.t = None
        self.cash = None
        self.holdings = None
        self.portfolio_value = None
        self._terminated = False
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # Memory tracking (from FinRL)
        self.asset_memory = []
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        min_t = self.state_space.window_size - 1
        # Đảm bảo episode có thể chạy tối đa self.max_steps bước mà không vượt quá self.max_t
        max_start_t = self.max_t - self.max_steps
        requested_start = None if options is None else options.get("start_t")

        if requested_start is not None:
            self.t = int(requested_start)
            if not (min_t <= self.t <= max_start_t):
                raise ValueError(
                    f"start_t ({self.t}) must be in [{min_t}, {max_start_t}]"
                )
        elif self.random_start:
            self.t = int(self.np_random.integers(min_t, max_start_t + 1))
        else:
            self.t = min_t

        if self.initial:
            self.cash = float(self.initial_balance)
            self.holdings = np.zeros(self.n_stocks, dtype=np.int64)
            self.asset_memory = [self.initial_balance]
        else:
            # Using Previous State
            if len(self.previous_state) > 0:
                self.cash = self.previous_state[0]
                self.holdings = np.array(self.previous_state[1:(self.n_stocks + 1)], dtype=np.int64)
                previous_total_asset = self.previous_state[0] + sum(
                    np.array(self.previous_state[1:(self.n_stocks + 1)]) *
                    np.array(self.previous_state[(self.n_stocks + 1):(self.n_stocks * 2 + 1)])
                )
                self.asset_memory = [previous_total_asset]
            else:
                self.cash = float(self.initial_balance)
                self.holdings = np.zeros(self.n_stocks, dtype=np.int64)
                self.asset_memory = [self.initial_balance]

        self.reward_fn.reset()
        self.current_step = 0
        self._terminated = False
        self.cost = 0
        self.trades = 0
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        prices = self.state_space.get_prices(self.t, field="close")
        self.portfolio_value = self.cash + np.sum(self.holdings * prices)

        obs = self.state_space.get_state(self.t, self.cash, self.holdings)
        info = self._build_info(prices)

        return obs, info

    def step(self, action):
        if self.t is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._terminated:
            raise RuntimeError("Episode already terminated. Call reset() before stepping again.")
        if self.t >= self.max_t:
            raise RuntimeError("Cannot step from the last available trading day. Call reset().")

        self.current_step += 1
        current_close_prices = self.state_space.get_prices(self.t, field="close")
        v_old = self.cash + np.sum(self.holdings * current_close_prices)
        execution_t = self.t + 1
        execution_prices = self.get_trade_prices(execution_t)

        "-------------------------------------ACTION-------------------------------------------"
        if self.mode == "discrete":
            action = self._normalize_discrete_action(action)
            trade_amounts = decode_discrete_action(
                action, self.n_stocks, self.min_shares, self.cash, self.holdings, execution_prices,
                fee_rate=self.fee_rate,
            )
        elif self.mode == "MultiDiscrete":
            trade_amounts = decode_discrete_action(
                action, self.n_stocks, self.min_shares, self.cash, self.holdings, execution_prices,
                fee_rate=self.fee_rate,
            )
        else:
            action = np.asarray(action, dtype=np.float32)
            if action.ndim > 1:
                action = np.squeeze(action)
            if action.shape != (self.n_stocks + 1,):
                raise ValueError(
                    f"Continuous action shape {action.shape} != ({self.n_stocks + 1},)"
                )

            # Hỗ trợ tương thích ngược: nếu policy trả [-1, 1] thì map sang [0, 1]
            if float(np.min(action)) < 0.0:
                action = (action + 1.0) / 2.0

            action = np.clip(action, 0.0, 1.0)
            s = float(np.sum(action))
            if not np.isfinite(s) or s <= 1e-8:
                action = np.zeros(self.n_stocks + 1, dtype=np.float32)
                action[-1] = 1.0
            else:
                action = action / s

            ratio = self.state_space.get_portfolio_state(self.cash, self.holdings, execution_prices)
            trade_amounts = decode_continuous_action(
                action,
                ratio,
                self.cash,
                self.holdings,
                execution_prices,
                trade_deadband=self.trade_deadband,
                max_weight_change_per_step=self.max_weight_change_per_step,
            )

        trade_amounts = apply_constraints(
            trade_amounts, self.cash, self.holdings, execution_prices, self.fee_rate, self.min_shares
        )

        total_fees = self._execute_trades(trade_amounts, execution_prices)
        self.cash = max(0.0, self.cash)
        self.cost += total_fees
        self.trades += int(np.sum(np.abs(trade_amounts) > 0))
        "-------------------------------------END ACTION-------------------------------------------"

        # Sau khi khop lenh o ngay t+1 (gia mo), danh dau gia tri danh muc tai gia dong ngay t+1.
        self.t = execution_t

        new_prices = self.state_space.get_prices(self.t, field="close")
        v_new = self.cash + np.sum(self.holdings * new_prices)
        self.portfolio_value = v_new

        post_trade_value = self.cash + np.sum(self.holdings * execution_prices)
        turnover_ratio = float(
            np.sum(np.abs(trade_amounts) * execution_prices) / max(v_old, 1e-12)
        )
        reward = self._calculate_reward(
            v_old=v_old,
            v_new=v_new,
            trade_amounts=trade_amounts,
            execution_prices=execution_prices,
            next_prices=new_prices,
            post_trade_value=post_trade_value,
        )
        # Scale reward giống FinRL
        reward = reward * self.reward_scaling

        obs = self.state_space.get_state(self.t, self.cash, self.holdings)

        # Ghi lại lịch sử tài sản, phần thưởng, hành động, thời gian (FinRL-style logging)
        self.asset_memory.append(self.portfolio_value)
        self.rewards_memory.append(reward)
        self.actions_memory.append(trade_amounts.copy())
        self.date_memory.append(self._get_date())

        terminated = (self.t >= self.max_t) or (self.current_step >= self.max_steps) or (v_new <= 0)
        truncated = False
        self._terminated = terminated or truncated

        info = self._build_info(
            new_prices,
            trade_amounts,
            total_fees,
            execution_prices=execution_prices,
            turnover_ratio=turnover_ratio,
        )

        # Logging at episode end
        if terminated:
            if self.make_plots:
                self._make_plot()
            end_total_asset = v_new
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = end_total_asset - self.initial_balance
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (252 ** 0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std()
            else:
                sharpe = 0

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.t}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            # Save results if model_name provided
            if self.model_name != "":
                df_actions = self.save_action_memory()
                df_actions.to_csv(f"results/actions_{self.model_name}_{self.iteration}.csv")
                df_total_value.to_csv(f"results/account_value_{self.model_name}_{self.iteration}.csv", index=False)
                pd.DataFrame(self.rewards_memory, columns=["account_rewards"]).to_csv(
                    f"results/account_rewards_{self.model_name}_{self.iteration}.csv", index=False
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(f"results/account_value_{self.model_name}_{self.iteration}.png")
                plt.close()

        return obs, float(reward), terminated, truncated, info

    def _normalize_discrete_action(self, action) -> np.ndarray:
        """
        Chuan hoa action discrete ve dang vector per-stock:
        - scalar int trong [0, k*n_stocks): chi tac dong len 1 ma, cac ma khac hold
        - vector/list/ndarray do dai n_stocks: legacy mode
        """
        if np.isscalar(action):
            action_idx = int(action)
            if action_idx < 0 or action_idx >= self.k * self.n_stocks:
                raise ValueError(
                    f"Discrete action index {action_idx} out of range [0, {self.k * self.n_stocks - 1}]"
                )
            normalized = np.ones(self.n_stocks, dtype=np.int64)
            stock_idx = action_idx // self.k
            decision = action_idx % self.k
            normalized[stock_idx] = decision
            return normalized

        normalized = np.asarray(action, dtype=np.int64)
        if normalized.shape != (self.n_stocks,):
            raise ValueError(
                f"Discrete action shape {normalized.shape} != ({self.n_stocks},)"
            )
        if np.any((normalized < 0) | (normalized >= self.k)):
            raise ValueError(f"Discrete action values must be in [0, {self.k - 1}]")
        return normalized

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
                self.cash -= cost
                self.holdings[i] += shares
                total_fees += fee

        return total_fees

    def get_trade_prices(self, t: Optional[int] = None) -> np.ndarray:
        if self.t is None and t is None:
            raise RuntimeError("Environment must be reset before accessing trade prices.")

        trade_t = (self.t + 1) if t is None else int(t)
        if trade_t > self.max_t:
            raise RuntimeError(
                f"Trade prices for t={trade_t} are not available (max_t={self.max_t})."
            )
        return self.state_space.get_prices(trade_t, field="open")

    def _calculate_reward(
        self,
        v_old: float,
        v_new: float,
        trade_amounts: np.ndarray,
        execution_prices: np.ndarray,
        next_prices: np.ndarray,
        post_trade_value: float,
    ) -> float:
        try:
            return float(
                self.reward_fn.calculate(
                    v_old,
                    v_new,
                    trade_amounts=trade_amounts,
                    execution_prices=execution_prices,
                    next_prices=next_prices,
                    post_trade_value=post_trade_value,
                )
            )
        except TypeError:
            return float(self.reward_fn.calculate(v_old, v_new, trade_amounts, execution_prices))

    def _build_info(self, prices, trades=None, fees=0.0, execution_prices=None, turnover_ratio: float = 0.0):
        if execution_prices is None:
            execution_prices = prices
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "prices": prices.copy(),
            "execution_prices": np.asarray(execution_prices).copy(),
            "trades": trades if trades is not None else np.zeros(self.n_stocks, dtype=np.int32),
            "fees": fees,
            "turnover_ratio": float(turnover_ratio),
            "date": str(self.state_space.dates[self.t]),
            "step": self.t - (self.state_space.window_size - 1),
        }

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def _get_date(self):
        return self.state_space.dates[self.t]

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({"date": date_list, "account_value": asset_list})
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.state_space.tickers
        df_actions.index = df_date.date
        return df_actions
