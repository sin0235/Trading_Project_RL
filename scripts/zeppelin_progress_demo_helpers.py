"""Helpers for a Zeppelin dashboard that compares early vs late PPO training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_BINDING_NAME = "learningProgressDashboard"


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _avg(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []

    running_sum = 0.0
    ma: list[float] = []
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        current_window = min(index + 1, window)
        ma.append(running_sum / current_window)
    return ma


def _build_svg_points(
    values: list[float],
    width: int = 620,
    height: int = 220,
    padding: int = 14,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[str, float | None, float | None, float | None]:
    if not values:
        return "", None, None, None

    chart_min = min(values) if min_value is None else min_value
    chart_max = max(values) if max_value is None else max_value
    if chart_min == chart_max:
        chart_min -= 1.0
        chart_max += 1.0

    value_range = chart_max - chart_min
    usable_width = width - (padding * 2)
    usable_height = height - (padding * 2)
    denom = max(len(values) - 1, 1)

    points: list[str] = []
    for index, value in enumerate(values):
        x = padding + (usable_width * index / denom)
        normalized = (value - chart_min) / value_range
        y = height - padding - (normalized * usable_height)
        points.append(f"{x:.1f},{y:.1f}")

    zero_y = None
    if chart_min <= 0.0 <= chart_max:
        zero_ratio = (0.0 - chart_min) / value_range
        zero_y = height - padding - (zero_ratio * usable_height)

    return " ".join(points), round(chart_min, 4), round(chart_max, 4), zero_y


def _episode_window_size(total_episodes: int, preview_rows: int) -> int:
    if total_episodes <= 0:
        return 0
    if total_episodes <= 4:
        return 1
    if total_episodes <= 12:
        return max(2, total_episodes // 4)
    return min(preview_rows, 10)


def _stage_payload(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        return {
            "label": label,
            "episode_start": None,
            "episode_end": None,
            "avg_return_pct": None,
            "avg_portfolio_value": None,
            "avg_trades": None,
            "rows": [],
        }

    return {
        "label": label,
        "episode_start": rows[0]["episode"],
        "episode_end": rows[-1]["episode"],
        "avg_return_pct": round(_avg([row["total_return_pct"] for row in rows]) or 0.0, 4),
        "avg_portfolio_value": round(_avg([row["portfolio_value"] for row in rows]) or 0.0, 2),
        "avg_trades": round(_avg([float(row["n_trades"]) for row in rows]) or 0.0, 2),
        "rows": rows,
    }


def _summary_metrics(summary_json: dict[str, Any]) -> dict[str, Any]:
    final_metrics = summary_json.get("final_metrics", {}) if summary_json else {}
    return {
        "total_return": _safe_float(final_metrics.get("total_return")),
        "annualized_return": _safe_float(final_metrics.get("annualized_return")),
        "sharpe_ratio": _safe_float(final_metrics.get("sharpe_ratio")),
        "max_drawdown": _safe_float(final_metrics.get("max_drawdown")),
        "final_value": _safe_float(final_metrics.get("final_value")),
    }


def _eval_payload(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    val_rows = [row for row in eval_rows if row.get("split") == "val"]
    test_rows = [row for row in eval_rows if row.get("split") == "test"]

    def _pick(row: dict[str, Any] | None) -> dict[str, Any]:
        if not row:
            return {}
        return {
            "episode": row.get("episode"),
            "total_return": row.get("total_return"),
            "sharpe_ratio": row.get("sharpe_ratio"),
            "max_drawdown": row.get("max_drawdown"),
            "final_value": row.get("final_value"),
        }

    return {
        "first_val": _pick(val_rows[0]) if val_rows else {},
        "last_val": _pick(val_rows[-1]) if val_rows else {},
        "final_test": _pick(test_rows[-1]) if test_rows else {},
    }


def _parse_episode_rows(metrics_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in metrics_rows:
        parsed.append(
            {
                "episode": _safe_int(row.get("episode")) or 0,
                "timestamp": row.get("timestamp"),
                "total_reward": _safe_float(row.get("total_reward")),
                "portfolio_value": _safe_float(row.get("portfolio_value")),
                "total_return_pct": _safe_float(row.get("total_return_pct")),
                "n_trades": _safe_int(row.get("n_trades")) or 0,
                "total_cost": _safe_float(row.get("total_cost")),
                "steps": _safe_int(row.get("steps")) or 0,
            }
        )
    return sorted(parsed, key=lambda item: item["episode"])


def _parse_eval_rows(eval_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in eval_rows:
        parsed.append(
            {
                "episode": _safe_int(row.get("episode")) or 0,
                "split": row.get("split"),
                "total_return": _safe_float(row.get("total_return")),
                "sharpe_ratio": _safe_float(row.get("sharpe_ratio")),
                "max_drawdown": _safe_float(row.get("max_drawdown")),
                "final_value": _safe_float(row.get("final_value")),
            }
        )
    return sorted(parsed, key=lambda item: (item["episode"], item["split"] or ""))


def build_learning_progress_payload(
    project_root: str | Path,
    results_dir: str | Path | None = None,
    preview_rows: int = 10,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runs_root = Path(results_dir) if results_dir is not None else root / "results" / "runs"
    if not runs_root.is_absolute():
        runs_root = (root / runs_root).resolve()

    if not runs_root.exists():
        raise FileNotFoundError(f"Results directory not found: {runs_root}")

    runs: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), key=lambda item: item.name):
        metrics_rows = _parse_episode_rows(_load_csv(run_dir / "metrics.csv"))
        if not metrics_rows:
            continue

        train_step_rows = _load_csv(run_dir / "train_steps.csv")
        eval_rows = _parse_eval_rows(_load_csv(run_dir / "eval_metrics.csv"))
        config_json = _load_json(run_dir / "config.json")
        summary_json = _load_json(run_dir / "summary.json")

        total_episodes = len(metrics_rows)
        window_size = _episode_window_size(total_episodes=total_episodes, preview_rows=preview_rows)
        early_rows = metrics_rows[:window_size]
        late_rows = metrics_rows[-window_size:]

        return_series = [row["total_return_pct"] or 0.0 for row in metrics_rows]
        moving_average = _moving_average(return_series, window=max(2, min(10, total_episodes)))
        chart_min = min(return_series + moving_average) if return_series else None
        chart_max = max(return_series + moving_average) if return_series else None
        raw_points, chart_min, chart_max, zero_y = _build_svg_points(
            return_series,
            min_value=chart_min,
            max_value=chart_max,
        )
        ma_points, _, _, _ = _build_svg_points(
            moving_average,
            min_value=chart_min,
            max_value=chart_max,
        )

        early_stage = _stage_payload(early_rows, label=f"Early ({early_rows[0]['episode']} -> {early_rows[-1]['episode']})")
        late_stage = _stage_payload(late_rows, label=f"Late ({late_rows[0]['episode']} -> {late_rows[-1]['episode']})")

        delta_return = None
        if early_stage["avg_return_pct"] is not None and late_stage["avg_return_pct"] is not None:
            delta_return = round(late_stage["avg_return_pct"] - early_stage["avg_return_pct"], 4)

        best_episode = max(metrics_rows, key=lambda item: item["total_return_pct"] if item["total_return_pct"] is not None else float("-inf"))
        worst_episode = min(metrics_rows, key=lambda item: item["total_return_pct"] if item["total_return_pct"] is not None else float("inf"))

        extra = summary_json.get("extra", {}) if summary_json else {}
        split_info = extra.get("data_split", {})

        config_brief = {
            "total_timesteps": config_json.get("total_timesteps"),
            "n_steps": config_json.get("n_steps"),
            "batch_size": config_json.get("batch_size"),
            "hidden_size": config_json.get("hidden_size"),
            "max_steps_train": config_json.get("max_steps_train"),
            "max_steps_eval": config_json.get("max_steps_eval"),
        }

        run_payload = {
            "run_id": run_dir.name,
            "label": f"{run_dir.name} ({total_episodes} episodes)",
            "total_episodes": total_episodes,
            "total_updates": len(train_step_rows),
            "window_size": window_size,
            "early": early_stage,
            "late": late_stage,
            "delta_return_pct": delta_return,
            "best_episode": {
                "episode": best_episode["episode"],
                "total_return_pct": round(best_episode["total_return_pct"] or 0.0, 4),
                "portfolio_value": round(best_episode["portfolio_value"] or 0.0, 2),
            },
            "worst_episode": {
                "episode": worst_episode["episode"],
                "total_return_pct": round(worst_episode["total_return_pct"] or 0.0, 4),
                "portfolio_value": round(worst_episode["portfolio_value"] or 0.0, 2),
            },
            "eval": _eval_payload(eval_rows),
            "summary": _summary_metrics(summary_json),
            "split": split_info,
            "config": config_brief,
            "chart": {
                "raw_points": raw_points,
                "ma_points": ma_points,
                "min_value": chart_min,
                "max_value": chart_max,
                "zero_y": round(zero_y, 2) if zero_y is not None else None,
                "latest_value": round(return_series[-1], 4) if return_series else None,
            },
        }
        runs.append(run_payload)

    if not runs:
        raise ValueError(f"No run artifacts with metrics.csv found under: {runs_root}")

    default_run = max(runs, key=lambda item: item["total_episodes"])
    run_map = {run["run_id"]: run for run in runs}

    return {
        "project_root": str(root),
        "results_dir": str(runs_root),
        "defaultRunId": default_run["run_id"],
        "runs": runs,
        "runMap": run_map,
    }


def bind_learning_progress_payload(z, payload: dict[str, Any], binding_name: str = DEFAULT_BINDING_NAME) -> None:
    z.angularBind(binding_name, payload)
