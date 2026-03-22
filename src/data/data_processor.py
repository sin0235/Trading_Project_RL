import pandas as pd
import numpy as np
from typing import List
import os

class DataProcessor:
    def __init__(self, dataset: List[pd.DataFrame]):
        self.dataset = [data.copy() for data in dataset]

    def clean_data(self) -> List[pd.DataFrame]:
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for i, data in enumerate(self.dataset):
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns: {missing_cols}")

            data['time'] = pd.to_datetime(data['time'], errors='coerce')
            data.sort_values('time', inplace=True)
            data.reset_index(drop=True, inplace=True)
            data.drop_duplicates(subset='time', inplace=True)
            data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].apply(
                pd.to_numeric, errors='coerce'
            )
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0).astype(float)

            # Chỉ forward-fill để tránh dùng dữ liệu tương lai cho quá khứ.
            data[['open', 'high', 'low', 'close']] = data[['open', 'high', 'low', 'close']].ffill()
            data.dropna(subset=['time', 'open', 'high', 'low', 'close'], inplace=True)
            self.dataset[i] = data.reset_index(drop=True)
        return self.dataset

    def _zscore_rolling(self, series: pd.Series, window: int = 60) -> pd.Series:
        rolling = series.rolling(window=window, min_periods=window)
        mean = rolling.mean()
        std = rolling.std()
        zscore = (series - mean) / std
        zero_std_mask = std.eq(0) & std.notna()
        zscore = zscore.mask(zero_std_mask, 0.0)
        return zscore.replace([np.inf, -np.inf], np.nan)

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
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        flat_mask = avg_gain.eq(0) & avg_loss.eq(0)
        rsi = rsi.mask(flat_mask, 50.0)
        rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
        rsi = rsi.mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
        return rsi

    def drop_na(self, start_date: str = '2015-01-01') -> List[pd.DataFrame]:
        """Xoa NaN va loc du lieu tu start_date tro di"""
        for i, data in enumerate(self.dataset):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
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
        atr_nonzero = atr.replace(0, np.nan)
        plus_di = 100 * smooth_plus_dm / atr_nonzero
        minus_di = 100 * smooth_minus_dm / atr_nonzero
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)  # tranh chia cho 0
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        dx = dx.mask((atr == 0) | di_sum.isna(), 0.0)

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

    # ---------------------------------------------------------------------------
    # EXTENDED FEATURES — ham moi, KHONG sua doi ham cu de co the fallback
    # ---------------------------------------------------------------------------

    def calculate_extended_features(self) -> List[pd.DataFrame]:
        """
        Tinh them 2 features moi (giu nguyen 7 features cu):
            - return_20d: rolling 20-day return (trend dai han)
            - volatility_20d: rolling 20-day realized volatility (regime detection)

        Tat ca deu duoc chuan hoa bang Z-score rolling 60 nhu cac features cu.
        Ham nay KHONG anh huong den calculate_features() — chi them cot moi.
        """
        for data in self.dataset:
            # return_20d: pct_change(20) -> zscore
            raw_return_20d = data['close'].pct_change(20)
            data['return_20d'] = self._zscore_rolling(raw_return_20d)

            # volatility_20d: rolling std cua daily returns (20 ngay) -> zscore
            daily_returns = data['close'].pct_change(1)
            raw_volatility_20d = daily_returns.rolling(window=20, min_periods=20).std()
            data['volatility_20d'] = self._zscore_rolling(raw_volatility_20d)

        return self.dataset

    def process_extended(self) -> List[pd.DataFrame]:
        """
        Pipeline mo rong: clean -> features cu -> features moi -> drop na.
        Tuong duong process() nhung them 2 features moi.
        Giu process() nguyen ven de fallback.
        """
        self.clean_data()
        self.calculate_features()
        self.calculate_extended_features()
        self.drop_na()
        return self.dataset
