"""
Script tao dataset extended (9 features) tu data processed hien co.
Doc data/processed (7 features), tinh them return_20d va volatility_20d,
luu vao data/processed_v2.

Chay: python scripts/generate_extended_dataset.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.data_processor import DataProcessor
from src.constants import TICKERS


def main():
    raw_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed_v2")

    # Load raw data
    datasets = []
    for ticker in TICKERS:
        file_path = os.path.join(raw_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            print(f"WARN: Khong tim thay {file_path}, skip {ticker}")
            continue
        df = pd.read_csv(file_path)
        df['symbol'] = ticker
        datasets.append(df)
        print(f"  Loaded {ticker}: {len(df)} rows")

    if not datasets:
        print("ERROR: Khong co data nao duoc load!")
        return

    print(f"\nTotal: {len(datasets)} tickers loaded")

    # Chay pipeline mo rong (7 features cu + 2 features moi)
    processor = DataProcessor(datasets)
    processor.process_extended()

    # Luu vao thu muc moi
    processor.save_data(folder_path=output_dir)

    # Verify
    print(f"\n--- Verification ---")
    sample = pd.read_csv(os.path.join(output_dir, f"{TICKERS[0]}.csv"))
    print(f"Sample ticker: {TICKERS[0]}")
    print(f"Columns: {list(sample.columns)}")
    print(f"Rows: {len(sample)}")

    expected_features = [
        'close_norm', 'return_1d', 'return_5d', 'return_20d',
        'macd', 'rsi', 'adx', 'volume_norm', 'volatility_20d'
    ]
    missing = [f for f in expected_features if f not in sample.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
    else:
        print(f"OK: All 9 features present")
        print(f"\nSample data (last 3 rows):")
        print(sample[expected_features].tail(3).to_string())


if __name__ == "__main__":
    main()
