"""Helpers for a Zeppelin PySpark + Angular demo notebook."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.constants import FEATURES


BASE_COLUMNS = ["time", "symbol", "open", "high", "low", "close", "volume"]
DEFAULT_BINDING_NAME = "demoDashboard"
SPLIT_SEQUENCE = ("train", "val", "test")


def _resolve_data_dir(project_root: str | Path, data_dir: str | Path | None = None) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(data_dir) if data_dir is not None else root / "data" / "processed"
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _timestamp_to_str(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _format_int(value: Any) -> int:
    return int(value) if value is not None else 0


def _build_svg_points(
    rows: list[dict[str, Any]],
    width: int = 520,
    height: int = 180,
    padding: int = 12,
) -> tuple[str, float | None, float | None]:
    if not rows:
        return "", None, None

    values = [float(row["close"]) for row in rows]
    min_close = min(values)
    max_close = max(values)
    value_range = max(max_close - min_close, 1e-9)
    usable_width = width - (padding * 2)
    usable_height = height - (padding * 2)
    denom = max(len(rows) - 1, 1)

    points: list[str] = []
    for index, row in enumerate(rows):
        x = padding + (usable_width * index / denom)
        normalized = (float(row["close"]) - min_close) / value_range
        y = height - padding - (normalized * usable_height)
        points.append(f"{x:.1f},{y:.1f}")

    return " ".join(points), round(min_close, 4), round(max_close, 4)


def load_feature_dataset(
    spark,
    project_root: str | Path,
    data_dir: str | Path | None = None,
) -> tuple[DataFrame, Path, list[str]]:
    resolved_data_dir = _resolve_data_dir(project_root=project_root, data_dir=data_dir)
    if not resolved_data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {resolved_data_dir}")

    file_pattern = str(resolved_data_dir / "*.csv")
    dataset = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(file_pattern)
    )

    if not dataset.columns:
        raise ValueError(f"No CSV files found under: {resolved_data_dir}")

    missing_columns = sorted(set(BASE_COLUMNS) - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for demo notebook: {missing_columns}")

    feature_columns = [column for column in FEATURES if column in dataset.columns]
    selected_columns = BASE_COLUMNS + feature_columns

    normalized = dataset.withColumn("time", F.to_timestamp("time"))
    for column in selected_columns:
        if column not in ("time", "symbol"):
            normalized = normalized.withColumn(column, F.col(column).cast("double"))

    normalized = (
        normalized.select(*selected_columns)
        .filter(F.col("symbol").isNotNull())
        .dropna(subset=["time", "symbol", "close"])
        .cache()
    )
    normalized.count()
    return normalized, resolved_data_dir, feature_columns


def attach_time_split(
    dataset: DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[DataFrame, list[dict[str, Any]], int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    tickers = [row["symbol"] for row in dataset.select("symbol").distinct().orderBy("symbol").collect()]
    if not tickers:
        raise ValueError("No ticker symbols available in dataset.")

    ticker_count = len(tickers)
    common_dates = (
        dataset.select("time", "symbol")
        .dropDuplicates()
        .groupBy("time")
        .agg(F.countDistinct("symbol").alias("ticker_count"))
        .filter(F.col("ticker_count") == ticker_count)
        .select("time")
        .orderBy("time")
    )

    n_common_days = common_dates.count()
    if n_common_days < 3:
        raise ValueError("Not enough common trading days to build train/val/test split.")

    n_train = int(n_common_days * train_ratio)
    n_val = int(n_common_days * val_ratio)
    n_test = n_common_days - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            "Split configuration creates an empty partition: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    labeled_dates = (
        common_dates.withColumn("row_number", F.row_number().over(Window.orderBy("time")))
        .withColumn(
            "split",
            F.when(F.col("row_number") <= n_train, F.lit("train"))
            .when(F.col("row_number") <= n_train + n_val, F.lit("val"))
            .otherwise(F.lit("test")),
        )
    )

    split_dataset = (
        dataset.join(labeled_dates.select("time", "split"), on="time", how="inner")
        .cache()
    )
    split_dataset.count()

    split_order = F.when(F.col("split") == "train", F.lit(1)).when(F.col("split") == "val", F.lit(2)).otherwise(F.lit(3))
    split_summary_rows = (
        split_dataset.groupBy("split")
        .agg(
            F.min("time").alias("start_date"),
            F.max("time").alias("end_date"),
            F.countDistinct("time").alias("n_days"),
            F.count("*").alias("n_rows"),
            F.countDistinct("symbol").alias("n_tickers"),
        )
        .orderBy(split_order)
        .collect()
    )

    split_summary = [
        {
            "split": row["split"],
            "start_date": _timestamp_to_str(row["start_date"]),
            "end_date": _timestamp_to_str(row["end_date"]),
            "n_days": _format_int(row["n_days"]),
            "n_rows": _format_int(row["n_rows"]),
            "n_tickers": _format_int(row["n_tickers"]),
        }
        for row in split_summary_rows
    ]

    return split_dataset, split_summary, n_common_days


def build_dashboard_payload(
    split_dataset: DataFrame,
    project_root: str | Path,
    data_dir: str | Path,
    feature_columns: list[str],
    split_summary: list[dict[str, Any]],
    n_common_days: int,
    sample_rows: int = 12,
    chart_rows: int = 120,
) -> dict[str, Any]:
    tickers = [row["symbol"] for row in split_dataset.select("symbol").distinct().orderBy("symbol").collect()]
    if not tickers:
        raise ValueError("No ticker symbols remain after applying common-date split.")

    dataset_bounds = split_dataset.agg(
        F.min("time").alias("start_date"),
        F.max("time").alias("end_date"),
        F.count("*").alias("n_rows"),
    ).first()

    summary_aggs = [
        F.min("time").alias("start_date"),
        F.max("time").alias("end_date"),
        F.count("*").alias("rows"),
        F.avg("close").alias("avg_close"),
        F.stddev("close").alias("close_std"),
        F.min("close").alias("min_close"),
        F.max("close").alias("max_close"),
        F.avg("volume").alias("avg_volume"),
    ]

    optional_feature_summary = {
        "return_1d": F.avg("return_1d").alias("avg_return_1d"),
        "return_5d": F.avg("return_5d").alias("avg_return_5d"),
        "rsi": F.avg("rsi").alias("avg_rsi"),
        "macd": F.avg("macd").alias("avg_macd"),
        "adx": F.avg("adx").alias("avg_adx"),
        "volume_norm": F.avg("volume_norm").alias("avg_volume_norm"),
    }
    for column, expression in optional_feature_summary.items():
        if column in split_dataset.columns:
            summary_aggs.append(expression)

    ticker_summary = (
        split_dataset.groupBy("symbol")
        .agg(*summary_aggs)
        .orderBy("symbol")
    )

    latest_window = Window.partitionBy("symbol").orderBy(F.col("time").desc())
    latest_columns = [F.col("symbol"), F.col("time").alias("latest_date"), F.col("close").alias("latest_close")]
    for column in ("return_1d", "return_5d", "rsi", "macd", "adx"):
        if column in split_dataset.columns:
            latest_columns.append(F.col(column).alias(f"latest_{column}"))

    latest_snapshot = (
        split_dataset.withColumn("row_number", F.row_number().over(latest_window))
        .filter(F.col("row_number") == 1)
        .select(*latest_columns)
    )

    summary_rows = ticker_summary.join(latest_snapshot, on="symbol", how="left").collect()
    summary_payload: list[dict[str, Any]] = []
    summary_map: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        item = {
            "symbol": row["symbol"],
            "start_date": _timestamp_to_str(row["start_date"]),
            "end_date": _timestamp_to_str(row["end_date"]),
            "rows": _format_int(row["rows"]),
            "avg_close": _round_or_none(row["avg_close"]),
            "close_std": _round_or_none(row["close_std"]),
            "min_close": _round_or_none(row["min_close"]),
            "max_close": _round_or_none(row["max_close"]),
            "avg_volume": _round_or_none(row["avg_volume"], 2),
            "latest_date": _timestamp_to_str(row["latest_date"]),
            "latest_close": _round_or_none(row["latest_close"]),
        }

        for column in ("avg_return_1d", "avg_return_5d", "avg_rsi", "avg_macd", "avg_adx", "avg_volume_norm"):
            if column in row.asDict():
                item[column] = _round_or_none(row[column])

        for column in ("latest_return_1d", "latest_return_5d", "latest_rsi", "latest_macd", "latest_adx"):
            if column in row.asDict():
                item[column] = _round_or_none(row[column])

        summary_payload.append(item)
        summary_map[item["symbol"]] = item

    sample_window = Window.partitionBy("symbol").orderBy(F.col("time").desc())
    sample_columns = [F.col("symbol"), F.col("time"), F.col("split"), F.col("close"), F.col("volume")]
    for column in ("return_1d", "return_5d", "rsi", "macd", "adx"):
        if column in split_dataset.columns:
            sample_columns.append(F.col(column))

    sample_records = (
        split_dataset.withColumn("row_number", F.row_number().over(sample_window))
        .filter(F.col("row_number") <= sample_rows)
        .select(*sample_columns)
        .orderBy("symbol", F.col("time").desc())
        .collect()
    )

    samples_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_records:
        samples_by_ticker[row["symbol"]].append(
            {
                "time": _timestamp_to_str(row["time"]),
                "split": row["split"],
                "close": _round_or_none(row["close"]),
                "volume": _round_or_none(row["volume"], 2),
                "return_1d": _round_or_none(row["return_1d"]) if "return_1d" in row.asDict() else None,
                "return_5d": _round_or_none(row["return_5d"]) if "return_5d" in row.asDict() else None,
                "rsi": _round_or_none(row["rsi"]) if "rsi" in row.asDict() else None,
                "macd": _round_or_none(row["macd"]) if "macd" in row.asDict() else None,
                "adx": _round_or_none(row["adx"]) if "adx" in row.asDict() else None,
            }
        )

    chart_window = Window.partitionBy("symbol").orderBy(F.col("time").desc())
    chart_records = (
        split_dataset.withColumn("row_number", F.row_number().over(chart_window))
        .filter(F.col("row_number") <= chart_rows)
        .select("symbol", "time", "close")
        .orderBy("symbol", "time")
        .collect()
    )

    chart_rows_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chart_records:
        chart_rows_by_ticker[row["symbol"]].append(
            {
                "time": _timestamp_to_str(row["time"]),
                "close": _round_or_none(row["close"]),
            }
        )

    chart_payload: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        rows = chart_rows_by_ticker[ticker]
        points, min_close, max_close = _build_svg_points(rows)
        latest_item = rows[-1] if rows else {"time": None, "close": None}
        chart_payload[ticker] = {
            "points": points,
            "min_close": min_close,
            "max_close": max_close,
            "latest_date": latest_item["time"],
            "latest_close": latest_item["close"],
            "rows": rows,
        }

    project_payload = {
        "project_root": str(Path(project_root).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "n_rows": _format_int(dataset_bounds["n_rows"]),
        "n_tickers": len(tickers),
        "n_common_days": n_common_days,
        "start_date": _timestamp_to_str(dataset_bounds["start_date"]),
        "end_date": _timestamp_to_str(dataset_bounds["end_date"]),
        "feature_columns": feature_columns,
        "feature_text": ", ".join(feature_columns) if feature_columns else "No engineered features available",
    }

    return {
        "project": project_payload,
        "tickers": tickers,
        "defaultTicker": tickers[0],
        "splits": split_summary,
        "summaryRows": summary_payload,
        "summaryMap": summary_map,
        "samples": dict(samples_by_ticker),
        "charts": chart_payload,
    }


def prepare_demo_dashboard(
    spark,
    project_root: str | Path,
    data_dir: str | Path | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    sample_rows: int = 12,
    chart_rows: int = 120,
) -> tuple[DataFrame, dict[str, Any]]:
    dataset, resolved_data_dir, feature_columns = load_feature_dataset(
        spark=spark,
        project_root=project_root,
        data_dir=data_dir,
    )
    split_dataset, split_summary, n_common_days = attach_time_split(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    payload = build_dashboard_payload(
        split_dataset=split_dataset,
        project_root=project_root,
        data_dir=resolved_data_dir,
        feature_columns=feature_columns,
        split_summary=split_summary,
        n_common_days=n_common_days,
        sample_rows=sample_rows,
        chart_rows=chart_rows,
    )
    return split_dataset, payload


def bind_dashboard_payload(z, payload: dict[str, Any], binding_name: str = DEFAULT_BINDING_NAME) -> None:
    z.angularBind(binding_name, payload)
