import pandas as pd
import numpy as np
from typing import List


class DataProcessor:
    def __init__(self, dataset: List[pd.DataFrame]):
        self.dataset = dataset

    def clean_data(self) -> List[pd.DataFrame]:
        for data in self.dataset:
            data.sort_values('time', inplace=True)
            data.reset_index(drop=True, inplace=True)
            data.drop_duplicates(subset='time', inplace=True)
            data['time'] = pd.to_datetime(data['time'])
            data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].astype(float)
            data['volume'] = data['volume'].fillna(0).astype(float)
            if data.isnull().any().any():
                data.ffill(inplace=True)
                data.bfill(inplace=True)
        return self.dataset

    def calculate_features(self) -> List[pd.DataFrame]:
        """Tinh 6 features theo state.md: close_norm, return_1d, return_5d, macd, rsi, volume_norm"""
        for data in self.dataset:
            data['close_norm'] = (
                (data['close'] - data['close'].rolling(60).mean())
                / data['close'].rolling(60).std()
            )

            data['return_1d'] = data['close'].pct_change(1)
            data['return_5d'] = data['close'].pct_change(5)

            # MACD histogram: (EMA12 - EMA26) - Signal(9)
            ema12 = data['close'].ewm(span=12, adjust=False).mean()
            ema26 = data['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            # Chuan hoa MACD theo rolling std cua close de scale dong nhat giua cac ma
            close_std = data['close'].rolling(60).std()
            data['macd'] = macd_hist / close_std

            data['rsi'] = self._calculate_rsi(data['close'], window=14) / 100.0

            data['volume_norm'] = (
                (data['volume'] - data['volume'].rolling(60).mean())
                / data['volume'].rolling(60).std()
            )

        return self.dataset

    @staticmethod
    def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def drop_na(self, min_window: int = 60) -> List[pd.DataFrame]:
        """Xoa cac dong dau chua du window de tinh features"""
        for i, data in enumerate(self.dataset):
            self.dataset[i] = data.iloc[min_window:].reset_index(drop=True)
        return self.dataset

    def process(self) -> List[pd.DataFrame]:
        self.clean_data()
        self.calculate_features()
        self.drop_na()
        return self.dataset
