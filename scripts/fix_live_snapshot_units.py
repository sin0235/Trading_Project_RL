from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_NOTE = ROOT / "notebooks" / "project_RL_nhom_09 .zpln"
RUNTIME_NOTE = Path(r"D:\Programs\zeppelin-0.12.0-bin-all\docker-data\notebook\project_RL_nhom_09 _2MNYSRNK4.zpln")
CACHE_FILES = [
    ROOT / ".zeppelin_cache" / "dashboard_train_payload.json",
    ROOT / ".zeppelin_cache" / "dashboard_payload.json",
]


OLD_BLOCK = """def build_live(data_dir, symbols, enable_live, suffix, limit):\r
    available = sorted([path.stem.upper() for path in data_dir.glob("*.csv")]) if data_dir.exists() else []\r
    tracked = (symbols or available)[:limit]\r
    rows = []\r
    alerts = []\r
    live_ready = False\r
    for symbol in tracked:\r
        last = last_csv_row(data_dir / (symbol + ".csv")) if data_dir.exists() else None\r
        row = {\r
            "symbol": symbol,\r
            "last_date": last.get("time") if last else None,\r
            "last_close": sfloat(last.get("close")) if last else None,\r
            "live_price": None,\r
            "change_pct": None,\r
            "gap_vs_last_close_pct": None,\r
            "status": "HISTORICAL_ONLY",\r
        }\r
        if enable_live:\r
            try:\r
                quote = live_quote(symbol, suffix)\r
                row.update(quote)\r
                row["status"] = "LIVE"\r
                live_ready = True\r
            except (urlerror.URLError, urlerror.HTTPError, ValueError, KeyError):\r
                pass\r
        if row["live_price"] is not None and row["last_close"] not in (None, 0):\r
            row["gap_vs_last_close_pct"] = round(((row["live_price"] / row["last_close"]) - 1.0) * 100.0, 4)\r
            if abs(row["gap_vs_last_close_pct"]) >= 2.0:\r
                alerts.append("{0} lệch {1:+.2f}% so với giá đóng cửa đã xử lý".format(symbol, row["gap_vs_last_close_pct"]))\r
        rows.append(row)\r
    status = "Đã lấy được giá trực tiếp" if live_ready else ("Đang chạy với dữ liệu cục bộ" if not enable_live else "Không lấy được giá trực tiếp")\r
    return {\r
        "rows": rows,\r
        "alerts": alerts,\r
        "status_label": status,\r
        "tracked_symbols": tracked,\r
        "refreshed_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),\r
    }\r
"""


NEW_BLOCK = """def align_price_units(last_close, live_price):\r
    last_close = sfloat(last_close)\r
    live_price = sfloat(live_price)\r
    if last_close in (None, 0) or live_price in (None, 0):\r
        return last_close\r
    abs_last = abs(last_close)\r
    abs_live = abs(live_price)\r
    if abs_last < 1000 <= abs_live:\r
        ratio = abs_live / max(abs_last, 1e-9)\r
        if ratio >= 20:\r
            return round(last_close * 1000.0, 4)\r
    return last_close\r
\r
\r
def build_live(data_dir, symbols, enable_live, suffix, limit):\r
    available = sorted([path.stem.upper() for path in data_dir.glob("*.csv")]) if data_dir.exists() else []\r
    tracked = (symbols or available)[:limit]\r
    rows = []\r
    alerts = []\r
    live_ready = False\r
    for symbol in tracked:\r
        last = last_csv_row(data_dir / (symbol + ".csv")) if data_dir.exists() else None\r
        historical_close = sfloat(last.get("close")) if last else None\r
        row = {\r
            "symbol": symbol,\r
            "last_date": last.get("time") if last else None,\r
            "last_close": historical_close,\r
            "reference_close": historical_close,\r
            "live_price": None,\r
            "change_pct": None,\r
            "gap_vs_last_close_pct": None,\r
            "status": "HISTORICAL_ONLY",\r
        }\r
        if enable_live:\r
            try:\r
                quote = live_quote(symbol, suffix)\r
                row.update(quote)\r
                row["status"] = "LIVE"\r
                live_ready = True\r
            except (urlerror.URLError, urlerror.HTTPError, ValueError, KeyError):\r
                pass\r
        row["reference_close"] = align_price_units(row.get("last_close"), row.get("live_price"))\r
        if row["live_price"] is not None and row["reference_close"] not in (None, 0):\r
            row["gap_vs_last_close_pct"] = round(((row["live_price"] / row["reference_close"]) - 1.0) * 100.0, 4)\r
            if abs(row["gap_vs_last_close_pct"]) >= 2.0:\r
                alerts.append("{0} lệch {1:+.2f}% so với close tham chiếu".format(symbol, row["gap_vs_last_close_pct"]))\r
        rows.append(row)\r
    status = "Đã lấy được giá trực tiếp" if live_ready else ("Đang chạy với dữ liệu cục bộ" if not enable_live else "Không lấy được giá trực tiếp")\r
    return {\r
        "rows": rows,\r
        "alerts": alerts,\r
        "status_label": status,\r
        "tracked_symbols": tracked,\r
        "refreshed_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),\r
    }\r
"""


def align_price_units(last_close, live_price):
    try:
        last = float(last_close)
        live = float(live_price)
    except (TypeError, ValueError):
        return last_close
    if last == 0 or live == 0:
        return last_close
    if abs(last) < 1000 <= abs(live):
        ratio = abs(live) / max(abs(last), 1e-9)
        if ratio >= 20:
            return round(last * 1000.0, 4)
    return last_close


def patch_note(path: Path) -> None:
    note = json.loads(path.read_text(encoding="utf-8-sig"))
    paragraph = note["paragraphs"][0]
    text = paragraph["text"]
    if OLD_BLOCK not in text:
        raise RuntimeError(f"Không tìm thấy build_live cũ trong {path}")
    paragraph["text"] = text.replace(OLD_BLOCK, NEW_BLOCK)
    path.write_text(json.dumps(note, ensure_ascii=False, separators=(",", ":")), encoding="utf-8-sig")


def patch_cache(path: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    live_market = payload.get("liveMarket") or {}
    rows = live_market.get("rows") or []
    alerts = []
    for row in rows:
        row["reference_close"] = align_price_units(row.get("last_close"), row.get("live_price"))
        if row.get("live_price") is not None and row.get("reference_close") not in (None, 0):
            row["gap_vs_last_close_pct"] = round(((float(row["live_price"]) / float(row["reference_close"])) - 1.0) * 100.0, 4)
            if abs(float(row["gap_vs_last_close_pct"])) >= 2.0:
                alerts.append(f'{row.get("symbol")} lệch {float(row["gap_vs_last_close_pct"]):+.2f}% so với close tham chiếu')
    if isinstance(live_market, dict):
        live_market["alerts"] = alerts
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    for note in (LOCAL_NOTE, RUNTIME_NOTE):
        patch_note(note)
        print(f"patched note {note}")
    for cache in CACHE_FILES:
        patch_cache(cache)
        print(f"patched cache {cache}")


if __name__ == "__main__":
    main()
