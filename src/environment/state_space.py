import pandas as pd
import numpy as np
from typing import List
import os
from src.constants import WINDOW_SIZE, DATA_PATH, FEATURES


class StateSpace:
    def __init__(self, tickers: List[str],
                 window_size: int = WINDOW_SIZE,
                 data_path: str = DATA_PATH,
                 features: List[str] = FEATURES,
                 mode: str = "flatten"):
        """
        Args:
            tickers: Danh sach ma co phieu (sorted)
            window_size: So ngay lich su (default 30)
            data_path: Duong dan den thu muc data processed
            features: Danh sach features su dung
            mode: 'flatten' hoac 'sequential'
        """
        self.tickers = tickers
        self.window_size = window_size
        self.data_path = data_path
        self.n_stocks = len(tickers)
        self.features = features
        self.n_features = len(self.features)
        self.mode = mode

        self.market_dim = self.window_size * self.n_stocks * self.n_features
        self.portfolio_dim = 1 + self.n_stocks
        self.state_dim = self.market_dim + self.portfolio_dim
        self._load_data()

    def _load_data(self):
        """Load data tu data_path, dieu chinh lai theo ngay thang de tranh thoi gian khong dong nhat"""
        dfs = {}
        close_series = {}
        for ticker in self.tickers:
            file_path = os.path.join(self.data_path, f"{ticker}.csv")
            df = pd.read_csv(file_path, parse_dates=['time'])
            df = df.set_index('time')
            dfs[ticker] = df[self.features]
            close_series[ticker] = df['close']

        common_dates = dfs[self.tickers[0]].index
        for ticker in self.tickers[1:]:
            common_dates = common_dates.intersection(dfs[ticker].index)
        common_dates = common_dates.sort_values()

        data_list = []
        for ticker in self.tickers:
            data_list.append(dfs[ticker].loc[common_dates].values)
        # (n_stocks, T, n_features) -> (T, n_stocks, n_features)
        self.data = np.stack(data_list, axis=0).transpose(1, 0, 2)

        self.dates = common_dates
        self.n_days = len(self.dates)

        self.close_prices = np.stack(
            [close_series[t].loc[common_dates].values for t in self.tickers],
            axis=1
        )

    def get_market_state(self, t: int) -> np.ndarray:
        """
        Lay thong tin thi truong tai thoi diem t, voi dieu kien t >= window_size - 1
        """
        start = t - self.window_size + 1
        window_data = self.data[start:t + 1]
        if self.mode == 'flatten':
            return window_data.flatten()
        elif self.mode == 'sequential':
            return window_data

    def get_portfolio_state(self, cash: float, holdings: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Lay thong tin danh muc, tra ve vector chua thong tin portfolio state
        voi cac tham so truyen vao:
        holdings: so luong co phieu dang nam giu
        cash: so tien mat hien co
        prices: gia co phieu tai thoi diem hien tai

        tra ve vector co 1 + n_stocks chieu, gom:
        - cash_ratio: ti le tien mat so voi portfolio value
        - holdings_ratio_0: ti le von dang nam giu so voi portfolio value
        - ...
        - holdings_ratio_n: ti le von dang nam giu so voi portfolio value
        """
        portfolio_value = cash + np.sum(holdings * prices)
        if portfolio_value <= 0:
            return np.zeros(self.portfolio_dim)

        cash_ratio = cash / portfolio_value
        holdings_ratio = (holdings * prices) / portfolio_value

        return np.concatenate([[cash_ratio], holdings_ratio])

    def get_state(self, t: int, cash: float, holdings: np.ndarray) -> np.ndarray:
        """
        Lay thong tin state tai thoi diem t, voi dieu kien t >= window_size - 1
        tra ve:
        - Neu mode la 'flatten', tra ve vector 1D co chieu la market_dim + portfolio_dim
        - Neu mode la 'sequential', tra ve tuple co 2 phan tu: market_state va portfolio_state
        """
        if t < self.window_size - 1:
            raise ValueError(f"t ({t}) must be >= window_size - 1 ({self.window_size - 1})")
        if t >= self.n_days:
            raise ValueError(f"t ({t}) must be < n_days ({self.n_days})")
        
        if cash < 0:
            raise ValueError(f"Cash cannot be negative: {cash}")
        
        if len(holdings) != self.n_stocks:
            raise ValueError(f"Holdings length ({len(holdings)}) != n_stocks ({self.n_stocks})")
        if np.any(holdings < 0):
            raise ValueError(f"Holdings cannot be negative: {holdings}")
        
        
        market_state = self.get_market_state(t)
        prices = self.close_prices[t]
        portfolio_state = self.get_portfolio_state(cash, holdings, prices)
        if self.mode == 'flatten':
            # Clip de tranh xuat hien cac gia tri qua lon hoac qua nho phat sinh loi trong qua trinh training
            return np.clip(np.concatenate([market_state, portfolio_state]), -5, 5).astype(np.float32)
        elif self.mode == 'sequential':
            return market_state.astype(np.float32), portfolio_state.astype(np.float32)

    def get_prices(self, t: int) -> np.ndarray:
        return self.close_prices[t]

    @property
    def observation_shape(self):
        if self.mode == 'flatten':
            return (self.state_dim,)
        elif self.mode == 'sequential':
            return (self.window_size, self.n_stocks, self.n_features)

    @property
    def max_steps(self):
        return self.n_days - self.window_size
