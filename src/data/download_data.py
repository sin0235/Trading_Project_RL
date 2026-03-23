import pandas as pd
from typing import List, Dict
import time
import os
import sys
import re
from importlib.util import find_spec

try:
    from src.constants import get_api_key_vnstock
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.constants import get_api_key_vnstock


class DownloadData:
    def __init__(self, tickers: List[str], start_date: str, end_date: str,
                 interval: str = '1D', source: str = 'KBS', delay: float = 1.0,
                 max_retries: int = 2, retry_buffer_seconds: float = 2.0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.source = source
        self.delay = delay
        self.max_retries = max_retries
        self.retry_buffer_seconds = retry_buffer_seconds
        self.data: Dict[str, pd.DataFrame] = {}
        self.dataset: List[pd.DataFrame] = []
        self._registered = False

    def _require_vnstock(self) -> None:
        if find_spec("vnstock") is None:
            raise ModuleNotFoundError("No module named 'vnstock'")

    def _ensure_registration(self) -> None:
        if self._registered:
            return
        self._require_vnstock()
        api_key = get_api_key_vnstock()
        if api_key:
            from vnstock import register_user
            register_user(api_key=api_key)
        self._registered = True

    def _extract_retry_wait_seconds(self, error: BaseException) -> float | None:
        text = str(error or "")
        patterns = [
            r"Chờ\s+(\d+)\s+giây",
            r"Wait to retry.*?(\d+)\s*giây",
            r"retry after\s+(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return float(match.group(1))
        return None

    def _is_rate_limit_error(self, error: BaseException) -> bool:
        text = str(error or "").lower()
        return "rate limit exceeded" in text or "giới hạn api đã đạt tối đa" in text

    def _get_single_with_retry(self, ticker: str) -> pd.DataFrame:
        attempts = max(int(self.max_retries), 0) + 1
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._get_single(ticker)
            except BaseException as error:
                last_error = error
                if not self._is_rate_limit_error(error) or attempt >= attempts:
                    raise
                wait_seconds = self._extract_retry_wait_seconds(error)
                if wait_seconds is None:
                    wait_seconds = 60.0
                wait_seconds = max(wait_seconds + float(self.retry_buffer_seconds), 1.0)
                print(
                    f"Cham gioi han API khi tai {ticker}, cho {wait_seconds:.0f} giay roi thu lai "
                    f"({attempt}/{attempts - 1} lan retry)"
                )
                time.sleep(wait_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Khong tai duoc du lieu cho {ticker}")

    def download_all(self):
        self.data = {}
        self._ensure_registration()
        for ticker in self.tickers:
            try:
                df = self._get_single_with_retry(ticker)
                self.data[ticker] = df
                print(f"Da tai: {ticker} - {len(df)} dong")
                if self.delay > 0:
                    time.sleep(self.delay)
            except BaseException as e:
                print(f"Loi khi tai {ticker}: {e}")

    def _get_single(self, ticker: str) -> pd.DataFrame:
        self._ensure_registration()
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=ticker, source=self.source)
        data = stock.quote.history(start=self.start_date, end=self.end_date, interval=self.interval)
        data['symbol'] = ticker
        return data

    def save_data(self, folder_path: str = "data\\raw"):
        os.makedirs(folder_path, exist_ok=True)
        for ticker, df in self.data.items():
            file_path = os.path.join(folder_path, f'{ticker}.csv')
            df.to_csv(file_path, index=False)
            print(f"Da luu: {file_path}")
        self.dataset = list(self.data.values()) 


if __name__ == '__main__':
    tickers = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG',
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'PLX', 'POW',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]
    downloader = DownloadData(
        tickers=tickers,
        start_date='2015-01-01',
        end_date='2025-12-31',
        interval='1D',
        source='VCI'
    )
    downloader.download_all()
    downloader.save_data()
