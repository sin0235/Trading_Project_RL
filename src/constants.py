API_KEY_VNSTOCK = "vnstock_6fa1ff85c804442f9e142c5a6f9deb3c"
TICKERS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG',
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'PLX', 'POW',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]
WINDOW_SIZE = 30
DATA_PATH = "data/processed"

FEATURES = [
    'close_norm', 'return_1d', 'return_5d', 'macd', 'rsi', 'adx', 'volume_norm'
]