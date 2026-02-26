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

    def _zscore_rolling(self, series: pd.Series, window: int = 60) -> pd.Series:
        return (series - series.rolling(window).mean()) / series.rolling(window).std()

    def calculate_features(self) -> List[pd.DataFrame]:
        """Tinh 7 features, tat ca deu duoc chuan hoa ve ~mean=0, std=1 (Z-score rolling 60)"""
        for data in self.dataset:
            data['close_norm'] = self._zscore_rolling(data['close'])

            raw_return_1d = data['close'].pct_change(1)
            data['return_1d'] = self._zscore_rolling(raw_return_1d)

            raw_return_5d = data['close'].pct_change(5)
            data['return_5d'] = self._zscore_rolling(raw_return_5d)

            ema12 = data['close'].ewm(span=12, adjust=False).mean()
            ema26 = data['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            data['macd'] = self._zscore_rolling(macd_hist)

            raw_rsi = self._calculate_rsi(data['close'], window=14)
            data['rsi'] = self._zscore_rolling(raw_rsi)

            raw_adx = self._calculate_adx(data, window=14)
            data['adx'] = self._zscore_rolling(raw_adx)

            data['volume_norm'] = self._zscore_rolling(data['volume'])

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

    def drop_na(self, start_date: str = '2015-01-01') -> List[pd.DataFrame]:
        """Xoa NaN va loc du lieu tu start_date tro di"""
        for i, data in enumerate(self.dataset):
            data = data.dropna()
            data = data[data['time'] >= start_date]
            self.dataset[i] = data.reset_index(drop=True)
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