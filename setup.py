from src.data.download_data import DownloadData
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
    
def clean_data():
    pass

def main():
    pass

if __name__ == "__main__":
    main()