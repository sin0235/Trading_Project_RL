"""Helpers to build a Zeppelin checkpoint replay demo for PPO trading runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from src.models.lstm import PPOLSTMActorCritic
from src.environment.trading_env import TradingEnv
from src.training.PPO import (
    DEFAULT_CONFIG,
    infer_run_config_from_checkpoint,
    load_run_config,
)
from src.data.data_processor import DataProcessor
from src.data.download_data import DownloadData


def _results_root(project_root: str | Path, results_dir: str | Path | None = None) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(results_dir) if results_dir is not None else root / "results" / "runs"
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _data_root(project_root: str | Path, data_dir: str | Path | None = None) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(data_dir) if data_dir is not None else root / "data" / "processed"
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _candidate_local_data_roots(
    project_root: str | Path,
    data_dir: str | Path | None,
    config_data_path: str | Path | None,
    features: list[str],
) -> list[Path]:
    root = Path(project_root).resolve()
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
    add(root / "data" / "processed")
    if any(feature in {"return_20d", "volatility_20d"} for feature in features):
        add(root / "data" / "processed_v2")

    return candidates


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("checkpoint_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


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
    sampled = _sample_evenly(numeric_paths, numeric_count)

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
) -> int:
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    threshold = pd.Timestamp(display_start)
    min_start_t = max(window_size - 1, 0)
    matches = np.where(date_index >= threshold)[0]
    if len(matches) == 0:
        return min_start_t
    return int(max(min_start_t, matches[0] - 1))


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


def _download_recent_processed_data(
    tickers: list[str],
    features: list[str],
    start_date: str,
    end_date: str,
    source: str,
) -> dict[str, pd.DataFrame]:
    downloader = DownloadData(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval="1D",
        source=source,
        delay=0.0,
    )
    downloader.download_all()
    if len(downloader.data) != len(tickers):
        missing = sorted(set(tickers) - set(downloader.data))
        raise RuntimeError(f"Tải vnstock chưa đủ mã: {missing}")
    raw_frames = [downloader.data[ticker] for ticker in tickers]
    return _process_downloaded_frames(raw_frames, features)


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
    today = pd.Timestamp(end_date or datetime.now().date())
    display_start = today - pd.Timedelta(days=max(recent_months, 1) * 31)
    feature_buffer_days = max(int(window_size) * 2, 120)
    requested_buffer_days = max(recent_months + warmup_months, 2) * 31
    fetch_start = today - pd.Timedelta(days=max((max(recent_months, 1) * 31) + feature_buffer_days, requested_buffer_days))
    warnings: list[str] = []

    try:
        recent_data = _download_recent_processed_data(
            tickers=tickers,
            features=features,
            start_date=_as_date_str(fetch_start),
            end_date=_as_date_str(today),
            source=vnstock_source,
        )
        source_name = "vnstock"
        source_label = f"vnstock ({vnstock_source})"
    except Exception as exc:
        warnings.append(f"Không lấy được dữ liệu vnstock, dùng dữ liệu cục bộ: {exc}")
        recent_data, resolved_local_root = _load_local_processed_data(data_roots, tickers, features)
        source_name = "local"
        source_label = f"Dữ liệu cục bộ ({resolved_local_root})"

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


def _make_eval_env(config: dict[str, Any], data_dict: dict[str, pd.DataFrame]) -> TradingEnv:
    return TradingEnv(
        tickers=list(config["tickers"]),
        mode="continuous",
        initial_balance=float(config["initial_balance"]),
        fee_rate=float(config["fee_rate"]),
        window_size=int(config["window_size"]),
        data_dict=data_dict,
        features=list(config["features"]),
        max_steps=999999,
        random_start=False,
        reward_scaling=float(config["reward_scaling"]),
        reward_name=str(config["reward_name"]),
        reward_kwargs={"window": int(config.get("reward_window", 30))},
        trade_deadband=float(config["trade_deadband"]),
        max_weight_change_per_step=float(config["max_weight_change_per_step"]),
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
                "shares": int(abs(trades[idx])),
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


def _build_benchmark_replay(
    config: dict[str, Any],
    data_dict: dict[str, pd.DataFrame],
    display_start: str,
) -> dict[str, Any]:
    env = _make_eval_env(config, data_dict)
    start_t = find_replay_start_t(
        dates=env.state_space.dates,
        display_start=display_start,
        window_size=int(config["window_size"]),
    )
    n_assets = len(config["tickers"]) + 1
    equal_weight = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

    eq_frames, eq_values = _run_episode_replay(
        env=_make_eval_env(config, data_dict),
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
        env=_make_eval_env(config, data_dict),
        policy_fn=buy_hold_policy,
        start_t=start_t,
        tickers=list(config["tickers"]),
        initial_balance=float(config["initial_balance"]),
    )

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
    if len(eq_values) != len(values) or len(bh_values) != len(values):
        raise ValueError("Replay benchmark và checkpoint không cùng số bước.")

    for idx, frame in enumerate(frames):
        equal_value = float(eq_values[idx])
        buy_hold_value = float(bh_values[idx])
        frame["equal_weight_value"] = round(equal_value, 2)
        frame["buy_hold_value"] = round(buy_hold_value, 2)
        frame["vs_equal_weight_pct"] = round(((frame["portfolio_value"] / equal_value) - 1.0) * 100.0, 4) if equal_value else None
        frame["vs_buy_hold_pct"] = round(((frame["portfolio_value"] / buy_hold_value) - 1.0) * 100.0, 4) if buy_hold_value else None

    series_min = min(values + eq_values + bh_values)
    series_max = max(values + eq_values + bh_values)
    model_points, marker_xs, chart_min, chart_max = _svg_points(values, min_value=series_min, max_value=series_max)
    eq_points, _, _, _ = _svg_points(eq_values, min_value=series_min, max_value=series_max)
    bh_points, _, _, _ = _svg_points(bh_values, min_value=series_min, max_value=series_max)

    final_value = float(values[-1]) if values else float(config["initial_balance"])
    initial_balance = float(config["initial_balance"])
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
        },
        "chart": {
            "model_points": model_points,
            "equal_weight_points": eq_points,
            "buy_hold_points": bh_points,
            "marker_xs": marker_xs,
            "min_value": chart_min,
            "max_value": chart_max,
        },
    }


def build_checkpoint_replay_payload(
    project_root: str | Path,
    results_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    recent_months: int = 4,
    warmup_months: int = 4,
    run_limit: int = 3,
    checkpoint_samples: int = 4,
    vnstock_source: str = "VCI",
    end_date: str | None = None,
) -> dict[str, Any]:
    runs_root = _results_root(project_root, results_dir)
    if not runs_root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục run: {runs_root}")

    run_dirs = sorted(
        [path for path in runs_root.iterdir() if path.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    replay_runs: list[dict[str, Any]] = []
    dataset_cache: dict[tuple[str, ...], tuple[DatasetBundle, dict[str, Any]]] = {}
    warnings: list[str] = []

    for run_dir in run_dirs:
        if len(replay_runs) >= max(run_limit, 1):
            break

        try:
            config = _load_run_config_for_replay(run_dir)
        except Exception as exc:
            warnings.append(f"Bỏ qua {run_dir.name}: {exc}")
            continue

        checkpoints = select_replay_checkpoints(run_dir, numeric_count=checkpoint_samples)
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
        checkpoint_payloads = [
            _checkpoint_payload(
                checkpoint_info=checkpoint,
                config=config,
                dataset=dataset,
                benchmark=benchmark,
            )
            for checkpoint in checkpoints
        ]

        default_checkpoint = next(
            (item for item in checkpoint_payloads if item["checkpoint_id"] == "final_model"),
            next(
                (item for item in checkpoint_payloads if item["checkpoint_id"] == "best_model"),
                checkpoint_payloads[-1],
            ),
        )

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
        "run_limit": int(run_limit),
        "defaultRunId": default_run["run_id"],
        "runs": replay_runs,
        "runMap": {run["run_id"]: run for run in replay_runs},
    }
