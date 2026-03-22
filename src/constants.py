import os
from dotenv import load_dotenv

load_dotenv()

API_KEY_VNSTOCK = os.getenv("API_KEY_VNSTOCK")

ALL_TICKERS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG',
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'PLX', 'POW',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
]

TICKERS = [
    'ACB', 'BID', 'BVH', 'CTG', 'FPT',
    'GAS', 'HPG', 'MBB', 'MSN', 'MWG',
    'SHB', 'SSI', 'STB', 'VCB', 'VIC', 
    'VNM',
]


WINDOW_SIZE = 30
DATA_PATH = "data/processed"

FEATURES = [
    'close_norm', 'return_1d', 'return_5d', 'macd', 'rsi', 'adx', 'volume_norm'
]

FEATURES_EXTENDED = [
    'close_norm', 'return_1d', 'return_5d', 'return_20d',
    'macd', 'rsi', 'adx', 'volume_norm', 'volatility_20d'
]
