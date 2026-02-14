from src.data.download_data import DownloadData
from src.data.data_processor import DataProcessor
from src.constants import TICKERS


def download_data():
    downloader = DownloadData(
        tickers=TICKERS,
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
