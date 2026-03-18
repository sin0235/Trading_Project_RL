import importlib
import sys
import unittest

import numpy as np
import pandas as pd

from src.data.data_processor import DataProcessor
from src.utils.data_splitter import split_by_ratio


def make_ohlcv_frame(n_days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "open": np.linspace(10, 20, n_days),
            "high": np.linspace(10.5, 20.5, n_days),
            "low": np.linspace(9.5, 19.5, n_days),
            "close": np.linspace(10, 20, n_days),
            "volume": np.linspace(100, 200, n_days),
            "symbol": ["AAA"] * n_days,
        }
    )


class DataPipelineLogicTests(unittest.TestCase):
    def test_src_data_import_is_lazy(self):
        sys.modules.pop("src.data", None)
        sys.modules.pop("src.data.download_data", None)

        module = importlib.import_module("src.data")

        self.assertEqual(module.__name__, "src.data")
        self.assertNotIn("src.data.download_data", sys.modules)

    def test_clean_data_does_not_backfill_from_future(self):
        df = make_ohlcv_frame(80)
        first_valid_time = df.loc[1, "time"]
        df.loc[0, "open"] = np.nan
        df.loc[0, "close"] = np.nan

        cleaned = DataProcessor([df]).clean_data()[0]

        self.assertEqual(cleaned.loc[0, "time"], first_valid_time)
        self.assertEqual(len(cleaned), 79)

    def test_process_constant_series_keeps_finite_rows(self):
        n = 120
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=n, freq="B"),
                "open": np.ones(n) * 10,
                "high": np.ones(n) * 10,
                "low": np.ones(n) * 10,
                "close": np.ones(n) * 10,
                "volume": np.ones(n) * 1000,
                "symbol": ["AAA"] * n,
            }
        )

        processed = DataProcessor([df]).process()[0]
        numeric = processed.select_dtypes(include=[np.number]).to_numpy()

        self.assertGreater(len(processed), 0)
        self.assertFalse(np.isnan(numeric).any())
        self.assertFalse(np.isinf(numeric).any())

    def test_split_by_ratio_raises_clear_error_for_tiny_dataset(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=3, freq="B"),
                "x": [1, 2, 3],
            }
        )

        with self.assertRaisesRegex(ValueError, "Tập dữ liệu quá nhỏ"):
            split_by_ratio({"AAA": df, "BBB": df}, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


if __name__ == "__main__":
    unittest.main()
