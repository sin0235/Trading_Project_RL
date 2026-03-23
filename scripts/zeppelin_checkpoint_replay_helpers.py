"""Helpers to build a Zeppelin checkpoint replay demo for PPO trading runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from src.models.lstm import DRQNNetwork, PPOLSTMActorCritic
from src.environment.trading_env import TradingEnv
from src.training.DDQ import (
    infer_run_config_from_checkpoint as infer_ddq_run_config_from_checkpoint,
    load_run_config as load_ddq_run_config,
    resolve_ddq_config,
)
from src.training.PPO import (
    DEFAULT_CONFIG,
    infer_run_config_from_checkpoint,
    load_run_config,
)
from src.data.data_processor import DataProcessor
from src.data.download_data import DownloadData
from src.utils.metrics import compute_all
from scripts.dashboard_paths import (
    DDQ_COMPARE_LABEL,
    FIXED_PPO_REPLAY_RUN_ID,
    DashboardProjectPaths,
)


def _results_root(project_root: str | Path, results_dir: str | Path | None = None) -> Path:
    paths = DashboardProjectPaths.from_project_root(project_root)
    root = paths.project_root
    candidate = Path(results_dir) if results_dir is not None else paths.runs_root
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _data_root(project_root: str | Path, data_dir: str | Path | None = None) -> Path:
    paths = DashboardProjectPaths.from_project_root(project_root)
    root = paths.project_root
    candidate = Path(data_dir) if data_dir is not None else paths.processed_data_root
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _candidate_local_data_roots(
    project_root: str | Path,
    data_dir: str | Path | None,
    config_data_path: str | Path | None,
    features: list[str],
) -> list[Path]:
    paths = DashboardProjectPaths.from_project_root(project_root)
    root = paths.project_root
    candidates: list[Path] = []

    def add(candidate: str | Path | None) -> None:
        if candidate in (None, ""):
            return
        path = Path(candidate)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path not in candidates:
            candidates.append(path)

    add(data_dir)
    add(config_data_path)
    add(paths.processed_data_root)
    if any(feature in {"return_20d", "volatility_20d"} for feature in features):
        add(paths.processed_v2_root)

    return candidates


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Không đọc được file JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"File JSON phải là object/dict: {path}")
    return raw


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("checkpoint_"):
        return -1
    try:
        return int(stem.split("checkpoint_", 1)[1])
    except (TypeError, ValueError, IndexError):
        return -1


def _summary_run_score(run_dir: Path) -> tuple[float, float, float]:
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = pd.read_json(summary_path, typ="series").to_dict()  # type: ignore[assignment]
        except ValueError:
            try:
                import json

                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

    final_metrics = (summary or {}).get("final_metrics") or {}
    total_return = final_metrics.get("total_return")
    sharpe_ratio = final_metrics.get("sharpe_ratio")
    try:
        total_return = float(total_return)
    except (TypeError, ValueError):
        total_return = float("-inf")
    try:
        sharpe_ratio = float(sharpe_ratio)
    except (TypeError, ValueError):
        sharpe_ratio = float("-inf")
    try:
        mtime = float(run_dir.stat().st_mtime)
    except OSError:
        mtime = float("-inf")
    return total_return, sharpe_ratio, mtime
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _even_sample_indices(n: int, max_len: int) -> list[int]:
    """Chọn index đều trên [0, n) để giảm kích thước payload JSON (Zeppelin angularBind)."""
    if n <= 0:
        return []
    if n <= max_len:
        return list(range(n))
    if max_len <= 1:
        return [n - 1]
    positions = np.linspace(0, n - 1, num=max_len)
    idxs = sorted({int(round(float(p))) for p in positions})
    idxs[0] = 0
    idxs[-1] = n - 1
    return sorted(set(idxs))


def _slim_replay_frame_for_zeppelin(frame: dict[str, Any]) -> None:
    rows = frame.get("trade_rows")
    if isinstance(rows, list) and len(rows) > 16:
        frame["trade_rows"] = rows[:16]


def _sample_evenly(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or not paths:
        return []
    if len(paths) <= count:
        return list(paths)

    positions = np.linspace(0, len(paths) - 1, num=count)
    selected: list[Path] = []
    seen = set()
    for raw_idx in positions:
        idx = int(round(float(raw_idx)))
        idx = max(0, min(idx, len(paths) - 1))
        path = paths[idx]
        if path not in seen:
            selected.append(path)
            seen.add(path)
    return selected


def select_replay_checkpoints(run_dir: str | Path, numeric_count: int = 4) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    checkpoint_dirs = [run_path / "checkpoints", run_path]

    checkpoint_dir = next((path for path in checkpoint_dirs if path.exists()), None)
    if checkpoint_dir is None:
        return []

    numeric_paths = sorted(
        checkpoint_dir.glob("checkpoint_*.pt"),
        key=lambda candidate: (_checkpoint_step(candidate), candidate.stat().st_mtime),
    )
    sampled = []
    if numeric_paths:
        sampled.append(numeric_paths[0])
    sampled.extend(_sample_evenly(numeric_paths, numeric_count))

    candidates: list[tuple[str, str, Path]] = []
    for path in sampled:
        step = _checkpoint_step(path)
        label = f"Checkpoint {step:,} bước" if step >= 0 else path.name
        candidates.append((f"checkpoint_{step}", label, path))

    for file_name, candidate_id, label in (
        ("best_model.pt", "best_model", "Checkpoint tốt nhất"),
        ("final_model.pt", "final_model", "Checkpoint cuối"),
    ):
        path = checkpoint_dir / file_name
        if path.exists():
            candidates.append((candidate_id, label, path))

    deduped: list[dict[str, Any]] = []
    seen_paths = set()
    for candidate_id, label, path in candidates:
        key = str(path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        deduped.append(
            {
                "checkpoint_id": candidate_id,
                "label": label,
                "path": path.resolve(),
                "kind": "numeric" if path.name.startswith("checkpoint_") else path.stem,
                "step": _checkpoint_step(path),
            }
        )

    numeric_items = [item for item in deduped if item["kind"] == "numeric"]
    extra_items = [item for item in deduped if item["kind"] != "numeric"]
    numeric_items.sort(key=lambda item: item["step"])
    return numeric_items + extra_items


def select_replay_checkpoints_best_and_second(run_dir: str | Path) -> list[dict[str, Any]]:
    """Chỉ best_model + checkpoint số thứ 2 (file checkpoint_*.pt sắp xếp theo bước)."""
    run_path = Path(run_dir)
    checkpoint_dirs = [run_path / "checkpoints", run_path]
    checkpoint_dir = next((path for path in checkpoint_dirs if path.exists()), None)
    if checkpoint_dir is None:
        return []

    numeric_paths = sorted(
        checkpoint_dir.glob("checkpoint_*.pt"),
        key=lambda candidate: (_checkpoint_step(candidate), candidate.stat().st_mtime),
    )
    best_path = checkpoint_dir / "best_model.pt"
    out: list[dict[str, Any]] = []

    if best_path.exists():
        out.append(
            {
                "checkpoint_id": "best_model",
                "label": "Checkpoint tốt nhất",
                "path": best_path.resolve(),
                "kind": "best_model",
                "step": -1,
            }
        )

    second_path: Path | None = None
    if len(numeric_paths) >= 2:
        second_path = numeric_paths[1]
    elif len(numeric_paths) == 1:
        only = numeric_paths[0]
        if not out or str(only.resolve()) != str(out[0]["path"]):
            second_path = only

    if second_path is not None:
        step = _checkpoint_step(second_path)
        cid = f"checkpoint_{step}" if step >= 0 else second_path.stem
        label = f"Checkpoint thứ 2 ({step:,} bước)" if step >= 0 else second_path.name
        out.append(
            {
                "checkpoint_id": cid,
                "label": label,
                "path": second_path.resolve(),
                "kind": "numeric",
                "step": step,
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        key = str(item["path"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    if not deduped and numeric_paths:
        path = numeric_paths[0]
        step = _checkpoint_step(path)
        deduped.append(
            {
                "checkpoint_id": f"checkpoint_{step}" if step >= 0 else path.stem,
                "label": f"Checkpoint ({step:,} bước)" if step >= 0 else path.name,
                "path": path.resolve(),
                "kind": "numeric",
                "step": step,
            }
        )

    return deduped


def _load_run_config_for_replay(run_dir: Path) -> dict[str, Any]:
    checkpoints = select_replay_checkpoints(run_dir, numeric_count=1)
    if not checkpoints:
        raise FileNotFoundError(f"Run không có checkpoint để replay: {run_dir}")

    ckpt_path = checkpoints[0]["path"]
    try:
        return load_run_config(run_dir, overrides={"device": "cpu"})
    except FileNotFoundError:
        return infer_run_config_from_checkpoint(
            ckpt_path,
            base_config={**DEFAULT_CONFIG, "device": "cpu"},
            overrides={"device": "cpu"},
        )


def _load_ddq_compare_config(
    project_root: str | Path,
    checkpoint_path: Path,
    base_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    paths = DashboardProjectPaths.from_project_root(project_root)

    for cfg_path in paths.ddq_config_json_candidates:
        if not cfg_path.exists():
            continue
        try:
            if cfg_path.name == "config.json":
                return load_ddq_run_config(cfg_path.parent, overrides={"device": "cpu"}), "ddq_config_json"
            raw_cfg = _load_json_mapping(cfg_path)
            cfg = resolve_ddq_config(config={**base_config, **raw_cfg, "device": "cpu"})
            return cfg, cfg_path.name
        except Exception:
            continue

    cfg = infer_ddq_run_config_from_checkpoint(
        checkpoint_path,
        base_config={**base_config, "device": "cpu"},
        overrides={"device": "cpu"},
    )
    return cfg, "checkpoint_inferred"


def _normalize_weights(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size == 0:
        return action

    if float(np.min(action)) < 0.0:
        action = (action + 1.0) / 2.0
    action = np.clip(action, 0.0, 1.0)
    total = float(np.sum(action))
    if not np.isfinite(total) or total <= 1e-8:
        normalized = np.zeros_like(action)
        normalized[-1] = 1.0
        return normalized
    return action / total


def _as_date_str(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def _common_dates(data_dict: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    tickers = list(data_dict.keys())
    common = pd.DatetimeIndex(pd.to_datetime(data_dict[tickers[0]]["time"]))
    for ticker in tickers[1:]:
        common = common.intersection(pd.DatetimeIndex(pd.to_datetime(data_dict[ticker]["time"])))
    return common.sort_values()


def find_replay_start_t(
    dates: list[Any] | pd.DatetimeIndex,
    display_start: Any,
    window_size: int,
    max_start_t: int | None = None,
) -> int:
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    threshold = pd.Timestamp(display_start)
    min_start_t = max(window_size - 1, 0)
    matches = np.where(date_index >= threshold)[0]
    if len(matches) == 0:
        start_t = min_start_t
    else:
        start_t = int(max(min_start_t, matches[0] - 1))
    if max_start_t is not None:
        start_t = min(start_t, int(max_start_t))
    return int(start_t)


def _trim_to_common_window(
    data_dict: dict[str, pd.DataFrame],
    display_start: Any,
    carry_days: int,
) -> tuple[dict[str, pd.DataFrame], str, str, int]:
    common = _common_dates(data_dict)
    threshold = pd.Timestamp(display_start)
    display_dates = common[common >= threshold]
    if len(display_dates) < 10:
        display_dates = common[-min(len(common), 40):]
    if len(display_dates) < 10:
        raise ValueError("Không đủ số ngày giao dịch chung để demo replay theo ngày.")

    first_display_idx = int(common.get_indexer([display_dates[0]])[0])
    keep_from_idx = max(first_display_idx - max(int(carry_days), 0), 0)
    carry_dates = common[keep_from_idx:]
    allowed = set(carry_dates)
    trimmed: dict[str, pd.DataFrame] = {}
    for ticker, df in data_dict.items():
        current = df.copy()
        current["time"] = pd.to_datetime(current["time"])
        current = current[current["time"].isin(allowed)].sort_values("time").reset_index(drop=True)
        trimmed[ticker] = current

    return trimmed, _as_date_str(display_dates[0]), _as_date_str(display_dates[-1]), int(len(display_dates))


def _load_local_processed_data(
    data_roots: list[Path],
    tickers: list[str],
    features: list[str],
) -> tuple[dict[str, pd.DataFrame], Path]:
    problems = []
    for data_root in data_roots:
        data_dict: dict[str, pd.DataFrame] = {}
        missing = []
        for ticker in tickers:
            path = data_root / f"{ticker}.csv"
            if not path.exists():
                missing.append(ticker)
                continue
            df = pd.read_csv(path, parse_dates=["time"])
            df["symbol"] = ticker
            data_dict[ticker] = df.sort_values("time").reset_index(drop=True)

        if missing:
            problems.append(f"{data_root}: thiếu file cho {missing}")
            continue

        missing_features = [
            feature
            for feature in features
            if any(feature not in frame.columns for frame in data_dict.values())
        ]
        if missing_features:
            problems.append(f"{data_root}: thiếu feature {sorted(set(missing_features))}")
            continue

        return data_dict, data_root

    raise FileNotFoundError(" | ".join(problems) if problems else "Không tìm thấy dữ liệu cục bộ phù hợp.")


def _process_downloaded_frames(
    raw_frames: list[pd.DataFrame],
    features: list[str],
) -> dict[str, pd.DataFrame]:
    processor = DataProcessor(raw_frames)
    if any(feature in {"return_20d", "volatility_20d"} for feature in features):
        processed = processor.process_extended()
    else:
        processed = processor.process()

    processed_map: dict[str, pd.DataFrame] = {}
    for frame in processed:
        symbol = str(frame["symbol"].iloc[0]).upper()
        processed_map[symbol] = frame.sort_values("time").reset_index(drop=True)
    return processed_map


def _candidate_vnstock_sources(source: str) -> list[str]:
    primary = str(source or "VCI").strip().upper() or "VCI"
    candidates = [primary]
    for fallback in ("VCI", "TCBS", "KBS"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _download_recent_processed_data(
    tickers: list[str],
    features: list[str],
    start_date: str,
    end_date: str,
    source: str,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, str]]:
    requested_tickers = [str(ticker).upper() for ticker in tickers]
    remaining = list(requested_tickers)
    collected: dict[str, pd.DataFrame] = {}
    ticker_sources: dict[str, str] = {}
    used_sources: list[str] = []
    problems: list[str] = []

    for candidate_source in _candidate_vnstock_sources(source):
        if not remaining:
            break

        downloader = DownloadData(
            tickers=remaining,
            start_date=start_date,
            end_date=end_date,
            interval="1D",
            source=candidate_source,
            delay=3.5,
            max_retries=3,
            retry_buffer_seconds=3.0,
        )
        downloader.download_all()

        if downloader.data:
            used_sources.append(candidate_source)
            for ticker, frame in downloader.data.items():
                symbol = str(ticker).upper()
                collected[symbol] = frame
                ticker_sources[symbol] = candidate_source

        remaining = [ticker for ticker in remaining if ticker not in collected]
        if remaining:
            problems.append(f"{candidate_source}: thiếu {remaining}")

    if remaining:
        joined = " | ".join(problems) if problems else f"thiếu {remaining}"
        raise RuntimeError(f"Tải vnstock chưa đủ mã sau khi thử nhiều source: {joined}")

    raw_frames = [collected[ticker] for ticker in requested_tickers]
    return _process_downloaded_frames(raw_frames, features), used_sources, ticker_sources


def _vnstock_available() -> bool:
    return find_spec("vnstock") is not None


@dataclass
class DatasetBundle:
    data_dict: dict[str, pd.DataFrame]
    data_source: str
    source_label: str
    display_start: str
    display_end: str
    n_days: int
    warnings: list[str]


def _build_recent_dataset(
    project_root: str | Path,
    tickers: list[str],
    features: list[str],
    window_size: int,
    data_roots: list[Path],
    recent_months: int,
    warmup_months: int,
    vnstock_source: str,
    end_date: str | None = None,
) -> DatasetBundle:
    today = pd.Timestamp(end_date or "2026-02-28")
    effective_recent_months = max(int(recent_months), 12 if int(window_size) >= 60 else 4)
    display_start = today - pd.Timedelta(days=effective_recent_months * 31)
    feature_buffer_days = max(int(window_size) * 2, 120)
    requested_buffer_days = max(effective_recent_months + warmup_months, 2) * 31
    fetch_start = today - pd.Timedelta(days=max((effective_recent_months * 31) + feature_buffer_days, requested_buffer_days))
    warnings: list[str] = []

    requested_source = str(vnstock_source or "VCI").strip()
    if requested_source.lower() == "local":
        raise RuntimeError("Replay demo hiện chỉ hỗ trợ dữ liệu mới từ vnstock. Không còn fallback sang dữ liệu cục bộ.")
    if not _vnstock_available():
        raise RuntimeError("Môi trường Zeppelin chưa có module vnstock, nên không thể dựng replay từ dữ liệu mới.")

    try:
        recent_data, used_sources, ticker_sources = _download_recent_processed_data(
            tickers=tickers,
            features=features,
            start_date=_as_date_str(fetch_start),
            end_date=_as_date_str(today),
            source=requested_source,
        )
        source_name = "vnstock"
        source_label = f"vnstock ({' + '.join(used_sources)})" if used_sources else f"vnstock ({requested_source})"
        if len(set(ticker_sources.values())) > 1:
            warnings.append(
                "Một số mã không có đủ dữ liệu ở source chính, đã tự thử source online khác: "
                + ", ".join(f"{ticker}:{ticker_sources[ticker]}" for ticker in sorted(ticker_sources))
            )
    except BaseException as exc:
        raise RuntimeError(
            "Không lấy được dữ liệu replay từ vnstock. "
            f"Hãy kiểm tra source/API rồi chạy lại Build Replay Cache. Chi tiết: {exc}"
        ) from exc

    missing_features = [
        feature
        for feature in features
        if any(feature not in frame.columns for frame in recent_data.values())
    ]
    if missing_features:
        raise ValueError(f"Bộ dữ liệu replay thiếu feature cần thiết: {sorted(set(missing_features))}")

    trimmed, start_str, end_str, n_days = _trim_to_common_window(
        recent_data,
        display_start=display_start,
        carry_days=max(int(window_size), 35),
    )
    return DatasetBundle(
        data_dict=trimmed,
        data_source=source_name,
        source_label=source_label,
        display_start=start_str,
        display_end=end_str,
        n_days=n_days,
        warnings=warnings,
    )


def _build_model(config: dict[str, Any], checkpoint_path: Path) -> PPOLSTMActorCritic:
    model = PPOLSTMActorCritic(
        n_stocks=len(config["tickers"]),
        n_features=len(config["features"]),
        seq_len=int(config["window_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        log_std_init=float(config.get("log_std_init", DEFAULT_CONFIG["log_std_init"])),
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint không chứa model_state_dict hợp lệ: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_ddq_model(config: dict[str, Any], checkpoint_path: Path) -> DRQNNetwork:
    model = DRQNNetwork(
        n_stocks=len(config["tickers"]),
        n_features=len(config["features"]),
        seq_len=int(config["window_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        k=int(config.get("k", 3)),
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint DDQ không chứa model_state_dict hợp lệ: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _make_eval_env(
    config: dict[str, Any],
    data_dict: dict[str, pd.DataFrame],
    *,
    min_shares: int | None = None,
    trade_deadband: float | None = None,
    max_weight_change_per_step: float | None = None,
) -> TradingEnv:
    return TradingEnv(
        tickers=list(config["tickers"]),
        mode="continuous",
        initial_balance=float(config["initial_balance"]),
        fee_rate=float(config["fee_rate"]),
        window_size=int(config["window_size"]),
        min_shares=int(min_shares if min_shares is not None else config.get("min_shares", 100)),
        data_dict=data_dict,
        features=list(config["features"]),
        max_steps=999999,
        random_start=False,
        reward_scaling=float(config["reward_scaling"]),
        reward_name=str(config["reward_name"]),
        reward_kwargs={"window": int(config.get("reward_window", 30))},
        trade_deadband=float(trade_deadband if trade_deadband is not None else config["trade_deadband"]),
        max_weight_change_per_step=float(
            max_weight_change_per_step
            if max_weight_change_per_step is not None
            else config["max_weight_change_per_step"]
        ),
        print_verbosity=999999,
    )


def _make_eval_env_discrete(
    config: dict[str, Any],
    data_dict: dict[str, pd.DataFrame],
) -> TradingEnv:
    return TradingEnv(
        tickers=list(config["tickers"]),
        mode="discrete",
        initial_balance=float(config["initial_balance"]),
        fee_rate=float(config["fee_rate"]),
        window_size=int(config["window_size"]),
        min_shares=int(config.get("min_shares", 100)),
        data_dict=data_dict,
        features=list(config["features"]),
        max_steps=999999,
        random_start=False,
        reward_scaling=float(config["reward_scaling"]),
        reward_name=str(config["reward_name"]),
        reward_kwargs={"window": int(config.get("reward_window", 30))},
        trade_deadband=float(config.get("trade_deadband", 0.0)),
        max_weight_change_per_step=float(config.get("max_weight_change_per_step", 1.0)),
        print_verbosity=999999,
    )


def _top_allocations(action: np.ndarray, tickers: list[str], limit: int = 6) -> list[dict[str, Any]]:
    labels = tickers + ["Tiền mặt"]
    rows = [
        {"label": label, "weight_pct": round(float(weight) * 100.0, 2)}
        for label, weight in zip(labels, action)
    ]
    rows.sort(key=lambda item: item["weight_pct"], reverse=True)
    return rows[:limit]


def _top_positions(
    holdings: np.ndarray,
    close_prices: np.ndarray,
    tickers: list[str],
    portfolio_value: float,
    cash: float,
    limit: int = 6,
) -> list[dict[str, Any]]:
    rows = [
        {
            "label": ticker,
            "shares": int(holding),
            "close": round(float(price), 2),
            "position_value": round(float(holding * price), 2),
            "weight_pct": round((float(holding * price) / portfolio_value) * 100.0, 2) if portfolio_value else 0.0,
        }
        for ticker, holding, price in zip(tickers, holdings, close_prices)
        if float(holding * price) > 0
    ]
    if cash > 0:
        rows.append(
            {
                "label": "Tiền mặt",
                "shares": None,
                "close": None,
                "position_value": round(float(cash), 2),
                "weight_pct": round((float(cash) / portfolio_value) * 100.0, 2) if portfolio_value else 0.0,
            }
        )
    rows.sort(key=lambda item: item["position_value"], reverse=True)
    return rows[:limit]


def _trade_rows(
    trades: np.ndarray,
    execution_prices: np.ndarray,
    close_prices: np.ndarray,
    holdings: np.ndarray,
    target_action: np.ndarray,
    tickers: list[str],
    limit: int = 6,
) -> list[dict[str, Any]]:
    rows = []
    for idx, ticker in enumerate(tickers):
        trade_shares = int(abs(trades[idx]))
        if trade_shares <= 0:
            continue
        trade_value = abs(float(trades[idx]) * float(execution_prices[idx]))
        direction = "Giữ"
        if trades[idx] > 0:
            direction = "Mua"
        elif trades[idx] < 0:
            direction = "Bán"
        rows.append(
            {
                "symbol": ticker,
                "direction": direction,
                "shares": trade_shares,
                "execution_price": round(float(execution_prices[idx]), 2),
                "close_price": round(float(close_prices[idx]), 2),
                "holding_shares": int(holdings[idx]),
                "target_weight_pct": round(float(target_action[idx]) * 100.0, 2),
                "trade_value": round(trade_value, 2),
            }
        )

    rows.sort(key=lambda item: (item["trade_value"], item["target_weight_pct"]), reverse=True)
    return rows[:limit]


def _headline(action: np.ndarray, tickers: list[str]) -> str:
    top = _top_allocations(action, tickers, limit=3)
    labels = [f"{item['label']} {item['weight_pct']:.1f}%" for item in top]
    return "Ưu tiên: " + ", ".join(labels)


def _svg_points(
    values: list[float],
    width: int = 620,
    height: int = 220,
    pad: int = 14,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[str, list[float], float | None, float | None]:
    if not values:
        return "", [], None, None

    chart_min = min(values) if min_value is None else min_value
    chart_max = max(values) if max_value is None else max_value
    if chart_min == chart_max:
        chart_min -= 1.0
        chart_max += 1.0

    denom = max(len(values) - 1, 1)
    yr = chart_max - chart_min
    points: list[str] = []
    xs: list[float] = []
    for index, value in enumerate(values):
        x = pad + ((width - (pad * 2)) * index / denom)
        y = height - pad - (((value - chart_min) / yr) * (height - (pad * 2)))
        points.append(f"{x:.1f},{y:.1f}")
        xs.append(round(float(x), 2))

    return " ".join(points), xs, round(float(chart_min), 4), round(float(chart_max), 4)


def _run_episode_replay(
    env: TradingEnv,
    policy_fn,
    start_t: int,
    tickers: list[str],
    initial_balance: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    obs, _ = env.reset(options={"start_t": start_t})
    frames: list[dict[str, Any]] = []
    values: list[float] = []
    done = False
    step_index = 0

    while not done:
        decision_date = _as_date_str(env.state_space.dates[env.t])
        previous_value = float(env.portfolio_value)
        action = _normalize_weights(policy_fn(env, obs, step_index))

        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        portfolio_value = float(env.portfolio_value)
        values.append(portfolio_value)

        holdings = np.asarray(info["holdings"], dtype=np.int64)
        close_prices = np.asarray(info["prices"], dtype=np.float64)
        execution_prices = np.asarray(info["execution_prices"], dtype=np.float64)
        trades = np.asarray(info["trades"], dtype=np.int64)
        trade_count = int(np.sum(np.abs(trades) > 0))
        day_return_pct = ((portfolio_value / previous_value) - 1.0) * 100.0 if previous_value else 0.0

        frames.append(
            {
                "index": step_index,
                "decision_date": decision_date,
                "date": _as_date_str(info["date"]),
                "portfolio_value": round(portfolio_value, 2),
                "day_return_pct": round(day_return_pct, 4),
                "total_return_pct": round(((portfolio_value / initial_balance) - 1.0) * 100.0, 4),
                "cash": round(float(info["cash"]), 2),
                "fees": round(float(info["fees"]), 2),
                "trade_count": trade_count,
                "cash_weight_pct": round(float(action[-1]) * 100.0, 2),
                "headline": _headline(action, tickers),
                "top_allocations": _top_allocations(action, tickers),
                "top_positions": _top_positions(
                    holdings=holdings,
                    close_prices=close_prices,
                    tickers=tickers,
                    portfolio_value=portfolio_value,
                    cash=float(info["cash"]),
                ),
                "trade_rows": _trade_rows(
                    trades=trades,
                    execution_prices=execution_prices,
                    close_prices=close_prices,
                    holdings=holdings,
                    target_action=action,
                    tickers=tickers,
                ),
            }
        )
        step_index += 1

    return frames, values


def _run_ddq_episode_values_only(
    env: TradingEnv,
    model: DRQNNetwork,
    start_t: int,
) -> list[float]:
    obs, _ = env.reset(options={"start_t": start_t})
    values: list[float] = []
    done = False

    while not done:
        market_state, portfolio_state = env.state_space.flat_obs_to_sequential(obs)
        market_tensor = torch.tensor(market_state, dtype=torch.float32).unsqueeze(0)
        portfolio_tensor = torch.tensor(portfolio_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = model.select_action(
                market_tensor,
                portfolio_tensor,
                hidden=None,
                epsilon=0.0,
            )
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        done = bool(terminated or truncated)
        values.append(float(env.portfolio_value))

    return values


def _series_summary(values: list[float], initial_balance: float) -> dict[str, Any]:
    if not values:
        return {
            "final_value": round(float(initial_balance), 2),
            "final_return_pct": 0.0,
            "best_value": round(float(initial_balance), 2),
            "worst_value": round(float(initial_balance), 2),
        }
    final_value = float(values[-1])
    return {
        "final_value": round(final_value, 2),
        "final_return_pct": round(((final_value / float(initial_balance)) - 1.0) * 100.0, 4),
        "best_value": round(max(values), 2),
        "worst_value": round(min(values), 2),
    }


def _daily_return_series(values: list[float], initial_balance: float) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float64)
    prev = float(initial_balance)
    returns: list[float] = []
    for value in values:
        current = float(value)
        returns.append(((current / prev) - 1.0) if prev else 0.0)
        prev = current
    return np.asarray(returns, dtype=np.float64)


def _drawdown_pct_series(values: list[float], initial_balance: float) -> list[float]:
    if not values:
        return []
    arr = np.asarray([float(initial_balance)] + [float(value) for value in values], dtype=np.float64)
    peaks = np.maximum.accumulate(arr)
    drawdowns = ((arr - peaks) / np.where(peaks > 0, peaks, 1e-10)) * 100.0
    return [round(float(value), 4) for value in drawdowns[1:]]


def _rolling_quality_series(
    values: list[float],
    initial_balance: float,
    window: int = 20,
    risk_free_rate: float = 0.045,
    clip_abs: float = 6.0,
) -> tuple[list[float], list[float]]:
    daily_returns = _daily_return_series(values, initial_balance)
    if daily_returns.size == 0:
        return [], []

    rf_daily = float(risk_free_rate) / 252.0
    sharpe_series: list[float] = []
    sortino_series: list[float] = []
    effective_window = max(int(window), 5)

    for idx in range(len(daily_returns)):
        start = max(0, idx - effective_window + 1)
        chunk = daily_returns[start : idx + 1]
        if len(chunk) < 2:
            sharpe_series.append(0.0)
            sortino_series.append(0.0)
            continue

        excess = chunk - rf_daily
        std = float(np.std(excess, ddof=1))
        sharpe = 0.0 if std < 1e-10 else float(np.mean(excess) / std * np.sqrt(252.0))

        downside = excess[excess < 0]
        if len(downside) == 0:
            sortino = max(sharpe, 0.0)
        else:
            downside_std = float(np.sqrt(np.mean(downside**2)))
            sortino = 0.0 if downside_std < 1e-10 else float(np.mean(excess) / downside_std * np.sqrt(252.0))

        sharpe_series.append(round(float(np.clip(sharpe, -clip_abs, clip_abs)), 4))
        sortino_series.append(round(float(np.clip(sortino, -clip_abs, clip_abs)), 4))

    return sharpe_series, sortino_series


def _risk_summary(values: list[float], initial_balance: float) -> dict[str, Any]:
    if not values:
        return {}

    metrics = compute_all(
        portfolio_values=[float(initial_balance)] + [float(value) for value in values],
        initial_balance=float(initial_balance),
    )

    def safe_value(name: str, scale: float = 1.0, digits: int = 4) -> float | None:
        raw = metrics.get(name)
        if raw is None:
            return None
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return round(numeric * scale, digits)

    return {
        "annualized_return_pct": safe_value("annualized_return", scale=100.0),
        "annualized_vol_pct": safe_value("annualized_vol", scale=100.0),
        "sharpe_ratio": safe_value("sharpe_ratio"),
        "sortino_ratio": safe_value("sortino_ratio"),
        "calmar_ratio": safe_value("calmar_ratio"),
        "max_drawdown_pct": safe_value("max_drawdown", scale=100.0),
        "max_drawdown_duration": int(metrics.get("max_drawdown_duration") or 0),
        "win_rate_pct": safe_value("win_rate", scale=100.0),
        "profit_factor": safe_value("profit_factor"),
    }


def _build_ddq_compare_overlay(
    project_root: str | Path,
    base_config: dict[str, Any],
    dataset: DatasetBundle,
) -> dict[str, Any] | None:
    paths = DashboardProjectPaths.from_project_root(project_root)
    checkpoint_path = next((candidate.resolve() for candidate in paths.ddq_checkpoint_candidates if candidate.exists()), None)
    if checkpoint_path is None:
        return None

    ddq_config, config_source = _load_ddq_compare_config(
        project_root=project_root,
        checkpoint_path=checkpoint_path,
        base_config=base_config,
    )

    if [str(t).upper() for t in ddq_config["tickers"]] != [str(t).upper() for t in base_config["tickers"]]:
        raise ValueError("Checkpoint DDQ cố định không cùng rổ ticker với replay PPO hiện tại.")

    missing_features = [
        feature
        for feature in ddq_config["features"]
        if any(feature not in frame.columns for frame in dataset.data_dict.values())
    ]
    if missing_features:
        raise ValueError(
            "Checkpoint DDQ yêu cầu feature không có trong dataset replay hiện tại: "
            f"{sorted(set(missing_features))}"
        )

    ddq_model = _build_ddq_model(ddq_config, checkpoint_path)
    ddq_env = _make_eval_env_discrete(ddq_config, dataset.data_dict)
    start_t = find_replay_start_t(
        dates=ddq_env.state_space.dates,
        display_start=dataset.display_start,
        window_size=int(ddq_config["window_size"]),
        max_start_t=int(ddq_env.max_t - ddq_env.max_steps),
    )
    values = _run_ddq_episode_values_only(
        env=ddq_env,
        model=ddq_model,
        start_t=start_t,
    )

    keep_from = max(int(dataset.n_days) - len(values), 0) if len(values) > dataset.n_days else 0
    values = values[keep_from:]

    return {
        "checkpoint_id": "ddq_best",
        "label": DDQ_COMPARE_LABEL,
        "checkpoint_path": str(checkpoint_path),
        "config_source": config_source,
        "values": values,
    }


def _build_benchmark_replay(
    config: dict[str, Any],
    data_dict: dict[str, pd.DataFrame],
    display_start: str,
) -> dict[str, Any]:
    env = _make_eval_env(
        config,
        data_dict,
        min_shares=1,
        trade_deadband=0.0,
        max_weight_change_per_step=1.0,
    )
    start_t = find_replay_start_t(
        dates=env.state_space.dates,
        display_start=display_start,
        window_size=int(config["window_size"]),
        max_start_t=int(env.max_t - env.max_steps),
    )
    n_stocks = len(config["tickers"])
    if n_stocks <= 0:
        raise ValueError("Replay benchmark không có ticker để mô phỏng.")

    # Danh mục đều: phân bổ đều toàn bộ vốn vào các mã cổ phiếu, không neo tiền mặt.
    # Mua và giữ: mua một lần theo tỷ trọng ban đầu rồi giữ nguyên.
    equal_weight = np.concatenate(
        [np.full(n_stocks, 1.0 / n_stocks, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )

    eq_frames, eq_values = _run_episode_replay(
        env=_make_eval_env(
            config,
            data_dict,
            min_shares=1,
            trade_deadband=0.0,
            max_weight_change_per_step=1.0,
        ),
        policy_fn=lambda _env, _obs, _step_idx: equal_weight,
        start_t=start_t,
        tickers=list(config["tickers"]),
        initial_balance=float(config["initial_balance"]),
    )

    def buy_hold_policy(current_env: TradingEnv, _obs, step_idx: int):
        if step_idx == 0:
            return equal_weight
        next_trade_prices = current_env.get_trade_prices()
        return current_env.state_space.get_portfolio_state(
            current_env.cash,
            current_env.holdings,
            next_trade_prices,
        ).astype(np.float32)

    bh_frames, bh_values = _run_episode_replay(
        env=_make_eval_env(
            config,
            data_dict,
            min_shares=1,
            trade_deadband=0.0,
            max_weight_change_per_step=1.0,
        ),
        policy_fn=buy_hold_policy,
        start_t=start_t,
        tickers=list(config["tickers"]),
        initial_balance=float(config["initial_balance"]),
    )

    keep_from = next((idx for idx, frame in enumerate(eq_frames) if str(frame["date"]) >= str(display_start)), 0)
    eq_frames = eq_frames[keep_from:]
    eq_values = eq_values[keep_from:]
    bh_frames = bh_frames[keep_from:]
    bh_values = bh_values[keep_from:]

    return {
        "display_start": display_start,
        "start_t": start_t,
        "equal_weight_frames": eq_frames,
        "equal_weight_values": eq_values,
        "buy_hold_frames": bh_frames,
        "buy_hold_values": bh_values,
    }


def _checkpoint_payload(
    checkpoint_info: dict[str, Any],
    config: dict[str, Any],
    dataset: DatasetBundle,
    benchmark: dict[str, Any],
    ddq_compare: dict[str, Any] | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    model = _build_model(config, checkpoint_info["path"])
    env = _make_eval_env(config, dataset.data_dict)
    start_t = int(benchmark["start_t"])

    def model_policy(_env: TradingEnv, obs, _step_idx: int):
        market_state, portfolio_state = env.state_space.flat_obs_to_sequential(obs)
        with torch.no_grad():
            market_tensor = torch.tensor(market_state, dtype=torch.float32).unsqueeze(0)
            portfolio_tensor = torch.tensor(portfolio_state, dtype=torch.float32).unsqueeze(0)
            concentration, _, _ = model.forward(market_tensor, portfolio_tensor, hidden=None)
            action = concentration / concentration.sum(dim=-1, keepdim=True)
        return action.cpu().numpy().squeeze(0)

    frames, values = _run_episode_replay(
        env=env,
        policy_fn=model_policy,
        start_t=start_t,
        tickers=list(config["tickers"]),
        initial_balance=float(config["initial_balance"]),
    )

    eq_values = benchmark["equal_weight_values"]
    bh_values = benchmark["buy_hold_values"]
    ddq_values = list(ddq_compare.get("values") or []) if ddq_compare else []
    lengths = [len(values), len(eq_values), len(bh_values)]
    if ddq_values:
        lengths.append(len(ddq_values))
    common_length = min(lengths)
    if common_length <= 0:
        raise ValueError("Replay checkpoint không có đủ dữ liệu để dựng theo ngày.")
    if (
        common_length != len(values)
        or common_length != len(eq_values)
        or common_length != len(bh_values)
        or (ddq_values and common_length != len(ddq_values))
    ):
        frames = frames[:common_length]
        values = values[:common_length]
        eq_values = eq_values[:common_length]
        bh_values = bh_values[:common_length]
        if ddq_values:
            ddq_values = ddq_values[:common_length]

    initial_balance = float(config["initial_balance"])
    model_drawdown_pct = _drawdown_pct_series(values, initial_balance)
    eq_drawdown_pct = _drawdown_pct_series(eq_values, initial_balance)
    bh_drawdown_pct = _drawdown_pct_series(bh_values, initial_balance)
    model_rolling_sharpe, model_rolling_sortino = _rolling_quality_series(values, initial_balance)
    ddq_drawdown_pct: list[float] = []
    ddq_rolling_sharpe: list[float] = []
    ddq_rolling_sortino: list[float] = []
    if ddq_values:
        ddq_drawdown_pct = _drawdown_pct_series(ddq_values, initial_balance)
        ddq_rolling_sharpe, ddq_rolling_sortino = _rolling_quality_series(ddq_values, initial_balance)

    for idx, frame in enumerate(frames):
        equal_value = float(eq_values[idx])
        buy_hold_value = float(bh_values[idx])
        model_return_pct = ((float(frame["portfolio_value"]) / initial_balance) - 1.0) * 100.0 if initial_balance else None
        equal_return_pct = ((equal_value / initial_balance) - 1.0) * 100.0 if initial_balance else None
        buy_hold_return_pct = ((buy_hold_value / initial_balance) - 1.0) * 100.0 if initial_balance else None
        frame["equal_weight_value"] = round(equal_value, 2)
        frame["buy_hold_value"] = round(buy_hold_value, 2)
        frame["equal_weight_return_pct"] = round(equal_return_pct, 4) if equal_return_pct is not None else None
        frame["buy_hold_return_pct"] = round(buy_hold_return_pct, 4) if buy_hold_return_pct is not None else None
        frame["vs_equal_weight_pct"] = round(model_return_pct - equal_return_pct, 4) if model_return_pct is not None and equal_return_pct is not None else None
        frame["vs_buy_hold_pct"] = round(model_return_pct - buy_hold_return_pct, 4) if model_return_pct is not None and buy_hold_return_pct is not None else None
        frame["drawdown_pct"] = model_drawdown_pct[idx] if idx < len(model_drawdown_pct) else None
        frame["rolling_sharpe"] = model_rolling_sharpe[idx] if idx < len(model_rolling_sharpe) else None
        frame["rolling_sortino"] = model_rolling_sortino[idx] if idx < len(model_rolling_sortino) else None

    if max_frames is not None and len(frames) > max_frames:
        idxs = _even_sample_indices(len(frames), max_frames)
        frames = [frames[i] for i in idxs]
        values = [values[i] for i in idxs]
        eq_values = [eq_values[i] for i in idxs]
        bh_values = [bh_values[i] for i in idxs]
        model_drawdown_pct = [model_drawdown_pct[i] for i in idxs]
        eq_drawdown_pct = [eq_drawdown_pct[i] for i in idxs]
        bh_drawdown_pct = [bh_drawdown_pct[i] for i in idxs]
        model_rolling_sharpe = [model_rolling_sharpe[i] for i in idxs]
        model_rolling_sortino = [model_rolling_sortino[i] for i in idxs]
        if ddq_values:
            ddq_values = [ddq_values[i] for i in idxs]
            ddq_drawdown_pct = [ddq_drawdown_pct[i] for i in idxs]
            ddq_rolling_sharpe = [ddq_rolling_sharpe[i] for i in idxs]
            ddq_rolling_sortino = [ddq_rolling_sortino[i] for i in idxs]

    for frame in frames:
        _slim_replay_frame_for_zeppelin(frame)

    series_pool = values + eq_values + bh_values + (ddq_values if ddq_values else [])
    series_min = min(series_pool)
    series_max = max(series_pool)
    model_points, marker_xs, chart_min, chart_max = _svg_points(values, min_value=series_min, max_value=series_max)
    eq_points, _, _, _ = _svg_points(eq_values, min_value=series_min, max_value=series_max)
    bh_points, _, _, _ = _svg_points(bh_values, min_value=series_min, max_value=series_max)
    drawdown_pool = model_drawdown_pct + bh_drawdown_pct + (ddq_drawdown_pct if ddq_drawdown_pct else [])
    drawdown_pool.extend(eq_drawdown_pct)
    drawdown_min = min(drawdown_pool) if drawdown_pool else -1.0
    drawdown_max = max(drawdown_pool) if drawdown_pool else 0.0
    drawdown_points, _, _, _ = _svg_points(model_drawdown_pct, min_value=drawdown_min, max_value=drawdown_max)
    eq_drawdown_points, _, _, _ = _svg_points(eq_drawdown_pct, min_value=drawdown_min, max_value=drawdown_max)
    bh_drawdown_points, _, _, _ = _svg_points(bh_drawdown_pct, min_value=drawdown_min, max_value=drawdown_max)
    quality_pool = model_rolling_sharpe + model_rolling_sortino
    quality_pool.extend(ddq_rolling_sharpe)
    quality_min = min(quality_pool) if quality_pool else -1.0
    quality_max = max(quality_pool) if quality_pool else 1.0
    rolling_sharpe_points, _, _, _ = _svg_points(model_rolling_sharpe, min_value=quality_min, max_value=quality_max)
    rolling_sortino_points, _, _, _ = _svg_points(model_rolling_sortino, min_value=quality_min, max_value=quality_max)
    ddq_points = None
    ddq_summary = None
    ddq_drawdown_points = None
    ddq_rolling_sharpe_points = None
    ddq_rolling_sortino_points = None
    if ddq_values:
        ddq_points, _, _, _ = _svg_points(ddq_values, min_value=series_min, max_value=series_max)
        ddq_summary = {
            **_series_summary(ddq_values, float(config["initial_balance"])),
            "risk_summary": _risk_summary(ddq_values, float(config["initial_balance"])),
        }
        ddq_drawdown_points, _, _, _ = _svg_points(ddq_drawdown_pct, min_value=drawdown_min, max_value=drawdown_max)
        ddq_rolling_sharpe_points, _, _, _ = _svg_points(
            ddq_rolling_sharpe,
            min_value=quality_min,
            max_value=quality_max,
        )
        ddq_rolling_sortino_points, _, _, _ = _svg_points(
            ddq_rolling_sortino,
            min_value=quality_min,
            max_value=quality_max,
        )
    for idx, frame in enumerate(frames):
        frame["marker_x"] = marker_xs[idx] if idx < len(marker_xs) else None

    final_value = float(values[-1]) if values else float(config["initial_balance"])
    return {
        "checkpoint_id": checkpoint_info["checkpoint_id"],
        "label": checkpoint_info["label"],
        "kind": checkpoint_info["kind"],
        "step": checkpoint_info["step"] if checkpoint_info["step"] >= 0 else None,
        "file_name": checkpoint_info["path"].name,
        "frameCount": len(frames),
        "frames": frames,
        "summary": {
            "final_value": round(final_value, 2),
            "final_return_pct": round(((final_value / initial_balance) - 1.0) * 100.0, 4),
            "best_value": round(max(values), 2),
            "worst_value": round(min(values), 2),
            "risk_summary": _risk_summary(values, float(config["initial_balance"])),
        },
        "ddq_compare": (
            {
                "checkpoint_id": ddq_compare["checkpoint_id"],
                "label": ddq_compare["label"],
                "checkpoint_path": ddq_compare["checkpoint_path"],
                "config_source": ddq_compare["config_source"],
                "summary": ddq_summary,
                "chart": {
                    "model_points": ddq_points,
                    "drawdown_points": ddq_drawdown_points,
                    "drawdown_series": ddq_drawdown_pct,
                    "rolling_sharpe_points": ddq_rolling_sharpe_points,
                    "rolling_sharpe_series": ddq_rolling_sharpe,
                    "rolling_sortino_points": ddq_rolling_sortino_points,
                    "rolling_sortino_series": ddq_rolling_sortino,
                },
            }
            if ddq_compare and ddq_points and ddq_summary
            else None
        ),
        "chart": {
            "model_points": model_points,
            "equal_weight_points": eq_points,
            "buy_hold_points": bh_points,
            "drawdown_points": drawdown_points,
            "drawdown_series": model_drawdown_pct,
            "equal_weight_drawdown_points": eq_drawdown_points,
            "equal_weight_drawdown_series": eq_drawdown_pct,
            "buy_hold_drawdown_points": bh_drawdown_points,
            "buy_hold_drawdown_series": bh_drawdown_pct,
            "rolling_sharpe_points": rolling_sharpe_points,
            "rolling_sharpe_series": model_rolling_sharpe,
            "rolling_sortino_points": rolling_sortino_points,
            "rolling_sortino_series": model_rolling_sortino,
            "marker_xs": marker_xs,
            "min_value": chart_min,
            "max_value": chart_max,
        },
    }


def build_checkpoint_replay_payload(
    project_root: str | Path,
    results_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    recent_months: int = 12,
    warmup_months: int = 4,
    run_limit: int = 1,
    checkpoint_samples: int = 4,
    vnstock_source: str = "VCI",
    end_date: str | None = "2026-02-28",
    max_frames_per_checkpoint: int | None = 420,
) -> dict[str, Any]:
    runs_root = _results_root(project_root, results_dir)
    if not runs_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục run: {runs_root}")

    run_dirs = sorted(
        [path for path in runs_root.iterdir() if path.is_dir()],
        key=_summary_run_score,
        reverse=True,
    )
    if FIXED_PPO_REPLAY_RUN_ID:
        run_dirs = [path for path in run_dirs if path.name == FIXED_PPO_REPLAY_RUN_ID]

    replay_runs: list[dict[str, Any]] = []
    dataset_cache: dict[tuple[str, ...], tuple[DatasetBundle, dict[str, Any]]] = {}
    warnings: list[str] = []
    if max_frames_per_checkpoint is not None and max_frames_per_checkpoint > 0:
        warnings.append(
            f"Mỗi checkpoint chỉ giữ tối đa {max_frames_per_checkpoint} frame trong payload Zeppelin "
            "(tránh angularBind/JSON.parse quá lớn trên trình duyệt)."
        )

    for run_dir in run_dirs:
        if len(replay_runs) >= max(run_limit, 1):
            break

        try:
            config = _load_run_config_for_replay(run_dir)
        except Exception as exc:
            warnings.append(f"Bỏ qua {run_dir.name}: {exc}")
            continue

        checkpoints = select_replay_checkpoints(
            run_dir,
            numeric_count=max(int(checkpoint_samples), 3),
        )
        if not checkpoints:
            warnings.append(f"Bỏ qua {run_dir.name}: không có checkpoint hợp lệ.")
            continue

        tickers = [str(ticker).upper() for ticker in config["tickers"]]
        feature_key = tuple(str(feature) for feature in config["features"])
        cache_key = tuple(tickers) + ("__features__",) + feature_key

        if cache_key not in dataset_cache:
            data_roots = _candidate_local_data_roots(
                project_root=project_root,
                data_dir=data_dir,
                config_data_path=config.get("data_path"),
                features=list(config["features"]),
            )
            dataset = _build_recent_dataset(
                project_root=project_root,
                tickers=tickers,
                features=list(config["features"]),
                window_size=int(config["window_size"]),
                data_roots=data_roots,
                recent_months=recent_months,
                warmup_months=warmup_months,
                vnstock_source=vnstock_source,
                end_date=end_date,
            )
            benchmark = _build_benchmark_replay(
                config=config,
                data_dict=dataset.data_dict,
                display_start=dataset.display_start,
            )
            dataset_cache[cache_key] = (dataset, benchmark)

        dataset, benchmark = dataset_cache[cache_key]
        ddq_compare = None
        try:
            ddq_compare = _build_ddq_compare_overlay(
                project_root=project_root,
                base_config=config,
                dataset=dataset,
            )
        except Exception as exc:
            warnings.append(f"Không dựng được line {DDQ_COMPARE_LABEL}: {exc}")

        checkpoint_payloads = [
            _checkpoint_payload(
                checkpoint_info=checkpoint,
                config=config,
                dataset=dataset,
                benchmark=benchmark,
                ddq_compare=ddq_compare,
                max_frames=max_frames_per_checkpoint,
            )
            for checkpoint in checkpoints
        ]

        best_checkpoint = next(
            (item for item in checkpoint_payloads if item["checkpoint_id"] == "best_model"),
            checkpoint_payloads[0],
        )
        worst_checkpoint = min(
            checkpoint_payloads,
            key=lambda item: (
                float(item.get("summary", {}).get("final_return_pct", 0.0)),
                float(item.get("summary", {}).get("worst_value", float("inf"))),
            ),
        )
        first_numeric_checkpoint = next(
            (item for item in checkpoint_payloads if item.get("kind") == "numeric"),
            checkpoint_payloads[0],
        )
        for item in checkpoint_payloads:
            item["compare_to_best"] = {
                "best_checkpoint_id": best_checkpoint["checkpoint_id"],
                "delta_final_value": round(
                    float(item["summary"]["final_value"]) - float(best_checkpoint["summary"]["final_value"]),
                    2,
                ),
                "delta_final_return_pct": round(
                    float(item["summary"]["final_return_pct"]) - float(best_checkpoint["summary"]["final_return_pct"]),
                    4,
                ),
                "delta_best_value": round(
                    float(item["summary"]["best_value"]) - float(best_checkpoint["summary"]["best_value"]),
                    2,
                ),
                "delta_worst_value": round(
                    float(item["summary"]["worst_value"]) - float(best_checkpoint["summary"]["worst_value"]),
                    2,
                ),
            }

        default_checkpoint = best_checkpoint

        replay_runs.append(
            {
                "run_id": run_dir.name,
                "label": f"{run_dir.name} | {dataset.n_days} ngày gần nhất",
                "tickers": tickers,
                "data_source": dataset.data_source,
                "data_source_label": dataset.source_label,
                "display_start": dataset.display_start,
                "display_end": dataset.display_end,
                "n_days": dataset.n_days,
                "checkpoints": checkpoint_payloads,
                "checkpointMap": {item["checkpoint_id"]: item for item in checkpoint_payloads},
                "defaultCheckpointId": default_checkpoint["checkpoint_id"],
                "bestCheckpointId": best_checkpoint["checkpoint_id"],
                "worstCheckpointId": worst_checkpoint["checkpoint_id"],
                "firstCheckpointId": first_numeric_checkpoint["checkpoint_id"],
                "defaultCompareCheckpointId": (
                    worst_checkpoint["checkpoint_id"]
                    if worst_checkpoint["checkpoint_id"] != default_checkpoint["checkpoint_id"]
                    else first_numeric_checkpoint["checkpoint_id"]
                ),
                "ddqCompareAvailable": bool(ddq_compare),
            }
        )
        warnings.extend(dataset.warnings)

    if not replay_runs:
        return {
            "status": "empty",
            "message": "Không dựng được payload replay từ các run hiện có.",
            "warnings": warnings,
            "runs": [],
            "runMap": {},
            "defaultRunId": None,
        }

    default_run = replay_runs[0]
    return {
        "status": "ready",
        "message": "Đã dựng payload replay checkpoint theo ngày.",
        "warnings": warnings,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recent_months": int(recent_months),
        "warmup_months": int(warmup_months),
        "run_limit": int(run_limit),
        "checkpoint_samples": int(checkpoint_samples),
        "vnstock_source": str(vnstock_source or "VCI"),
        "end_date": str(end_date or "2026-02-28"),
        "max_frames_per_checkpoint": max_frames_per_checkpoint,
        "defaultRunId": default_run["run_id"],
        "runs": replay_runs,
        "runMap": {run["run_id"]: run for run in replay_runs},
    }
