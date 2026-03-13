import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.constants import TICKERS, DATA_PATH
import os


@dataclass
class DataSplit:
    """Kết quả chia dữ liệu, bao gồm cả metadata để ghi vào báo cáo."""
    train: Dict[str, pd.DataFrame]
    val: Dict[str, pd.DataFrame]
    test: Dict[str, pd.DataFrame]
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    n_train_days: int
    n_val_days: int
    n_test_days: int

    def summary(self) -> dict:
        return {
            "train": {"start": self.train_start, "end": self.train_end, "days": self.n_train_days},
            "val":   {"start": self.val_start,   "end": self.val_end,   "days": self.n_val_days},
            "test":  {"start": self.test_start,  "end": self.test_end,  "days": self.n_test_days},
        }


def load_data(
    tickers: List[str] = None,
    data_path: str = DATA_PATH,
) -> Dict[str, pd.DataFrame]:
    """
    Load toàn bộ dữ liệu đã processed từ data_path.
    Trả về dict {ticker: DataFrame} với cột 'time' đã parse.
    """
    if tickers is None:
        tickers = TICKERS

    data_dict: Dict[str, pd.DataFrame] = {}
    missing = []
    for ticker in tickers:
        file_path = os.path.join(data_path, f"{ticker}.csv")
        if not os.path.exists(file_path):
            missing.append(ticker)
            continue
        df = pd.read_csv(file_path, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        data_dict[ticker] = df

    if missing:
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu cho các ticker: {missing}"
        )
    return data_dict


def split_by_ratio(
    data_dict: Dict[str, pd.DataFrame],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DataSplit:
    """
    Chia dữ liệu theo tỷ lệ thời gian (không shuffle).
    Dùng ngày chung (intersection) của tất cả ticker làm trục thời gian.

    Tỷ lệ mặc định: 70% train / 15% val / 15% test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "train_ratio + val_ratio + test_ratio phải bằng 1.0"

    # Lấy tập ngày chung
    common_dates = _get_common_dates(data_dict)
    n = len(common_dates)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    train_dates = common_dates[:n_train]
    val_dates   = common_dates[n_train : n_train + n_val]
    test_dates  = common_dates[n_train + n_val :]

    train, val, test = _slice_by_dates(data_dict, train_dates, val_dates, test_dates)

    return DataSplit(
        train=train, val=val, test=test,
        train_start=str(train_dates[0].date()),
        train_end=str(train_dates[-1].date()),
        val_start=str(val_dates[0].date()),
        val_end=str(val_dates[-1].date()),
        test_start=str(test_dates[0].date()),
        test_end=str(test_dates[-1].date()),
        n_train_days=len(train_dates),
        n_val_days=len(val_dates),
        n_test_days=len(test_dates),
    )


def split_by_date(
    data_dict: Dict[str, pd.DataFrame],
    train_end: str,
    val_end: str,
) -> DataSplit:
    """
    Chia dữ liệu theo mốc ngày cố định.
    Ví dụ: train_end='2022-12-31', val_end='2023-12-31'
    → train: đến 2022-12-31 | val: 2023 | test: 2024 trở đi
    """
    common_dates = _get_common_dates(data_dict)

    train_dates = common_dates[common_dates <= train_end]
    val_dates   = common_dates[(common_dates > train_end) & (common_dates <= val_end)]
    test_dates  = common_dates[common_dates > val_end]

    if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
        raise ValueError(
            f"Một trong các tập dữ liệu rỗng. "
            f"train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)} ngày."
        )

    train, val, test = _slice_by_dates(data_dict, train_dates, val_dates, test_dates)

    return DataSplit(
        train=train, val=val, test=test,
        train_start=str(train_dates[0].date()),
        train_end=str(train_dates[-1].date()),
        val_start=str(val_dates[0].date()),
        val_end=str(val_dates[-1].date()),
        test_start=str(test_dates[0].date()),
        test_end=str(test_dates[-1].date()),
        n_train_days=len(train_dates),
        n_val_days=len(val_dates),
        n_test_days=len(test_dates),
    )


def _get_common_dates(data_dict: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    tickers = list(data_dict.keys())
    common = pd.DatetimeIndex(data_dict[tickers[0]]["time"].values)
    for ticker in tickers[1:]:
        common = common.intersection(pd.DatetimeIndex(data_dict[ticker]["time"].values))
    return common.sort_values()


def _slice_by_dates(
    data_dict: Dict[str, pd.DataFrame],
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> Tuple[Dict, Dict, Dict]:
    train, val, test = {}, {}, {}
    for ticker, df in data_dict.items():
        df = df.set_index("time")
        train[ticker] = df.loc[df.index.isin(train_dates)].reset_index()
        val[ticker]   = df.loc[df.index.isin(val_dates)].reset_index()
        test[ticker]  = df.loc[df.index.isin(test_dates)].reset_index()
    return train, val, test
