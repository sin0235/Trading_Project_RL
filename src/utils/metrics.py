"""
Metrics đánh giá hiệu quả trading agent.

Tất cả hàm nhận đầu vào là list/array portfolio_values (giá trị tài khoản theo từng bước)
và trả về dict có thể ghi thẳng vào báo cáo.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


TRADING_DAYS_PER_YEAR = 252


def compute_all(
    portfolio_values: List[float],
    initial_balance: float,
    risk_free_rate: float = 0.045,
    benchmark_values: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Tính toàn bộ metrics từ chuỗi giá trị tài khoản.

    Args:
        portfolio_values: Danh sách giá trị tài khoản theo từng bước (bao gồm giá trị ban đầu).
        initial_balance: Vốn ban đầu.
        risk_free_rate: Lãi suất phi rủi ro hàng năm (mặc định 4.5% - tương đương trái phiếu VN).
        benchmark_values: Chuỗi giá trị benchmark (ví dụ VN-Index) để tính alpha/beta.

    Returns:
        dict chứa tất cả metrics, sẵn sàng để log hoặc đưa vào báo cáo.
    """
    values = np.array(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return {}

    daily_returns = np.diff(values) / values[:-1]

    result: Dict[str, float] = {}

    result["total_return"]        = _total_return(values, initial_balance)
    result["annualized_return"]   = _annualized_return(daily_returns)
    result["annualized_vol"]      = _annualized_volatility(daily_returns)
    result["sharpe_ratio"]        = _sharpe_ratio(daily_returns, risk_free_rate)
    result["sortino_ratio"]       = _sortino_ratio(daily_returns, risk_free_rate)
    result["calmar_ratio"]        = _calmar_ratio(daily_returns, values)
    result["max_drawdown"]        = _max_drawdown(values)
    result["max_drawdown_duration"] = _max_drawdown_duration(values)
    result["win_rate"]            = _win_rate(daily_returns)
    result["profit_factor"]       = _profit_factor(daily_returns)
    result["avg_daily_return"]    = float(np.mean(daily_returns))
    result["std_daily_return"]    = float(np.std(daily_returns))
    result["skewness"]            = _skewness(daily_returns)
    result["kurtosis"]            = _kurtosis(daily_returns)
    result["var_95"]              = _value_at_risk(daily_returns, confidence=0.95)
    result["cvar_95"]             = _conditional_var(daily_returns, confidence=0.95)
    result["final_value"]         = float(values[-1])
    result["initial_value"]       = float(initial_balance)

    if benchmark_values is not None:
        bench = np.array(benchmark_values, dtype=np.float64)
        bench_returns = np.diff(bench) / bench[:-1]
        n = min(len(daily_returns), len(bench_returns))
        result["alpha"], result["beta"] = _alpha_beta(
            daily_returns[:n], bench_returns[:n], risk_free_rate
        )
        result["information_ratio"] = _information_ratio(
            daily_returns[:n], bench_returns[:n]
        )

    return result


# -----------------------------------------------------------------------
# Các hàm tính từng metric
# -----------------------------------------------------------------------

def _total_return(values: np.ndarray, initial_balance: float) -> float:
    return float((values[-1] - initial_balance) / initial_balance)


def _annualized_return(daily_returns: np.ndarray) -> float:
    n = len(daily_returns)
    if n == 0:
        return 0.0
    cumulative = float(np.prod(1 + daily_returns))
    return float(cumulative ** (TRADING_DAYS_PER_YEAR / n) - 1)


def _annualized_volatility(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(np.std(daily_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sharpe_ratio(daily_returns: np.ndarray, risk_free_rate: float) -> float:
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino_ratio(daily_returns: np.ndarray, risk_free_rate: float) -> float:
    """Chỉ phạt downside volatility (return < risk_free)."""
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-10:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(values: np.ndarray) -> float:
    """Max drawdown tính theo tỷ lệ từ đỉnh."""
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / np.where(peak > 0, peak, 1e-10)
    return float(np.max(drawdown))


def _max_drawdown_duration(values: np.ndarray) -> int:
    """Số bước dài nhất kể từ lần đỉnh trước đến khi phục hồi."""
    peak = np.maximum.accumulate(values)
    in_drawdown = values < peak
    max_dur = 0
    cur_dur = 0
    for flag in in_drawdown:
        if flag:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return int(max_dur)


def _calmar_ratio(daily_returns: np.ndarray, values: np.ndarray) -> float:
    ann_ret = _annualized_return(daily_returns)
    mdd = _max_drawdown(values)
    if mdd < 1e-10:
        return 0.0
    return float(ann_ret / mdd)


def _win_rate(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    return float(np.mean(daily_returns > 0))


def _profit_factor(daily_returns: np.ndarray) -> float:
    """Tổng lợi nhuận / tổng thua lỗ (tính theo giá trị tuyệt đối)."""
    gains  = daily_returns[daily_returns > 0].sum()
    losses = (-daily_returns[daily_returns < 0]).sum()
    if losses < 1e-10:
        return float("inf")
    return float(gains / losses)


def _skewness(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 3:
        return 0.0
    return float(pd.Series(daily_returns).skew())


def _kurtosis(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 4:
        return 0.0
    return float(pd.Series(daily_returns).kurt())


def _value_at_risk(daily_returns: np.ndarray, confidence: float = 0.95) -> float:
    """VaR: mức thua lỗ tối đa tại ngưỡng confidence (giá trị dương = mức lỗ)."""
    if len(daily_returns) == 0:
        return 0.0
    return float(-np.percentile(daily_returns, (1 - confidence) * 100))


def _conditional_var(daily_returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall): trung bình thua lỗ vượt VaR."""
    if len(daily_returns) == 0:
        return 0.0
    var = _value_at_risk(daily_returns, confidence)
    tail = daily_returns[daily_returns <= -var]
    if len(tail) == 0:
        return var
    return float(-np.mean(tail))


def _alpha_beta(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float,
) -> tuple:
    """
    Tính alpha và beta theo CAPM.
    alpha = annualized excess return không giải thích được bởi beta.
    """
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    rp = portfolio_returns - rf_daily
    rb = benchmark_returns - rf_daily

    cov_matrix = np.cov(rp, rb)
    var_bench = cov_matrix[1, 1]
    if var_bench < 1e-10:
        return 0.0, 0.0

    beta = float(cov_matrix[0, 1] / var_bench)
    alpha_daily = float(np.mean(rp) - beta * np.mean(rb))
    alpha_annual = float(alpha_daily * TRADING_DAYS_PER_YEAR)
    return alpha_annual, beta


def _information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """IR = mean(active_return) / std(active_return)."""
    active = portfolio_returns - benchmark_returns
    std = np.std(active, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(active) / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def format_report(metrics: Dict[str, float]) -> str:
    """
    Định dạng metrics thành chuỗi văn bản dễ đọc, phù hợp để đưa vào báo cáo.
    """
    lines = [
        "=" * 55,
        f"{'PERFORMANCE REPORT':^55}",
        "=" * 55,
        f"  Vốn ban đầu          : {metrics.get('initial_value', 0):>18,.0f} VND",
        f"  Giá trị cuối kỳ      : {metrics.get('final_value', 0):>18,.0f} VND",
        "-" * 55,
        f"  Tổng lợi nhuận       : {metrics.get('total_return', 0):>17.2%}",
        f"  Lợi nhuận hàng năm   : {metrics.get('annualized_return', 0):>17.2%}",
        f"  Biến động hàng năm   : {metrics.get('annualized_vol', 0):>17.2%}",
        "-" * 55,
        f"  Sharpe Ratio         : {metrics.get('sharpe_ratio', 0):>17.4f}",
        f"  Sortino Ratio        : {metrics.get('sortino_ratio', 0):>17.4f}",
        f"  Calmar Ratio         : {metrics.get('calmar_ratio', 0):>17.4f}",
        "-" * 55,
        f"  Max Drawdown         : {metrics.get('max_drawdown', 0):>17.2%}",
        f"  MDD Duration (steps) : {metrics.get('max_drawdown_duration', 0):>17}",
        "-" * 55,
        f"  Win Rate             : {metrics.get('win_rate', 0):>17.2%}",
        f"  Profit Factor        : {metrics.get('profit_factor', 0):>17.4f}",
        f"  Skewness             : {metrics.get('skewness', 0):>17.4f}",
        f"  Kurtosis             : {metrics.get('kurtosis', 0):>17.4f}",
        "-" * 55,
        f"  VaR 95%              : {metrics.get('var_95', 0):>17.4f}",
        f"  CVaR 95%             : {metrics.get('cvar_95', 0):>17.4f}",
    ]

    if "alpha" in metrics:
        lines += [
            "-" * 55,
            f"  Alpha (annual)       : {metrics.get('alpha', 0):>17.4f}",
            f"  Beta                 : {metrics.get('beta', 0):>17.4f}",
            f"  Information Ratio    : {metrics.get('information_ratio', 0):>17.4f}",
        ]

    lines.append("=" * 55)
    return "\n".join(lines)
