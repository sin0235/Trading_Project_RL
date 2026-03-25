import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs):
        return False

def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        load_dotenv(candidate, override=False)

    # Keep the default lookup as a last fallback for environments that inject
    # .env elsewhere or already expose variables in the current process.
    load_dotenv(override=False)


_load_project_env()

def get_api_key_vnstock() -> str | None:
    _load_project_env()
    return os.getenv("API_KEY_VNSTOCK")


def get_env_float(
    name: str,
    default: float,
    *,
    fallback_names: tuple[str, ...] = (),
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    _load_project_env()
    for key in (name, *fallback_names):
        raw = os.getenv(key)
        if raw in (None, ""):
            continue
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value
    return float(default)


API_KEY_VNSTOCK = get_api_key_vnstock()
CHART_COMPARE_ALPHA = get_env_float(
    "CHART_COMPARE_ALPHA",
    0.95,
    fallback_names=("alpha",),
    min_value=0.0,
)

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


WINDOW_SIZE = 60
DATA_PATH = "data/processed"

FEATURES = [
    'close_norm', 'return_1d', 'return_5d', 'macd', 'rsi', 'adx', 'volume_norm'
]

FEATURES_EXTENDED = [
    'close_norm', 'return_1d', 'return_5d', 'return_20d',
    'macd', 'rsi', 'adx', 'volume_norm', 'volatility_20d'
]
