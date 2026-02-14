from src.data.download_data import DownloadData
from src.data.data_processor import DataProcessor
from typing import List
import pandas as pd

def download_data():
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
        source='VCI')
    downloader.download_all()
    downloader.save_data()
    
    dataset = DataProcessor(downloader.dataset)
    dataset.process()
    dataset.save_data()


if __name__ == "__main__":
    download_data()