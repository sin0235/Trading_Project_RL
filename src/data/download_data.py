import pandas as pd
import numpy as np
from vnstock import Vnstock, register_user
from typing import List, Dict
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from constants import API_KEY_VNSTOCK

register_user(api_key=API_KEY_VNSTOCK)


class DownloadData:
    def __init__(self, tickers: List[str], start_date: str, end_date: str,
                 interval: str = '1D', source: str = 'KBS', delay: float = 1.0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.source = source
        self.delay = delay
        self.data: Dict[str, pd.DataFrame] = {}
        self.dataset: List[pd.DataFrame] = []

    def download_all(self):
        self.data = {}
        for ticker in self.tickers:
            try:
                df = self._get_single(ticker)
                self.data[ticker] = df
                print(f"Da tai: {ticker} - {len(df)} dong")
                time.sleep(self.delay)
            except Exception as e:
                print(f"Loi khi tai {ticker}: {e}")

    def _get_single(self, ticker: str) -> pd.DataFrame:
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
