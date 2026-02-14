import pandas as pd
import numpy as np
from typing import List
import os

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
        """Tinh 7 features theo state.md: close_norm, return_1d, return_5d, macd, rsi, volume_norm"""
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
            
            data['adx'] = self._calculate_adx(data, window=14) / 100.0

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
    

    def _calculate_adx(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement (+DM, -DM)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        smooth_plus_dm = plus_dm.ewm(alpha=1/window, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/window, adjust=False).mean()

        # +DI, -DI, DX
        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)  # tranh chia cho 0
        dx = 100 * (plus_di - minus_di).abs() / di_sum

        # ADX = smoothed DX
        adx = dx.ewm(alpha=1/window, adjust=False).mean()

        return adx

    def process(self) -> List[pd.DataFrame]:
        self.clean_data()
        self.calculate_features()
        self.drop_na()
        return self.dataset
    
    def save_data(self, folder_path: str = "data/processed") -> None:
        os.makedirs(folder_path, exist_ok=True)
        for data in self.dataset:
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else None
            name = f"{symbol}.csv" if symbol else f"data_{id(data)}.csv"
            file_path = os.path.join(folder_path, name)
            data.to_csv(file_path, index=False)
            print(f"Da luu: {file_path}")