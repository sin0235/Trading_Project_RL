import numpy as np
from collections import deque


"""
Giữ lại reward cũ (AdvancedRewardFunction) để tham chiếu / tương thích.
Reward chính: SharpeRewardFunction, SharpePlusRewardFunction.
"""


class AdvancedRewardFunction:
    def __init__(
        self,
        window: int = 30,
        alpha: float = 0.1,  # Trọng số cho Volatility (Rủi ro)
        beta: float = 0.5,   # Trọng số cho Max Drawdown (Sụt giảm)
        gamma: float = 0.01,  # Trọng số cho Transaction Penalty (Chi phí)
        initial_balance = None 
    ):
        self.window = window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.initial_balance = initial_balance
        
        self.returns_history = deque(maxlen=window)
        self.max_portfolio_value = self.initial_balance if self.initial_balance is not None else -np.inf

    def reset(self):
        self.returns_history.clear()
        self.max_portfolio_value = self.initial_balance if self.initial_balance is not None else -np.inf

    def calculate(self, v_old: float, v_new: float, trade_amounts: np.ndarray = None, execution_prices: np.ndarray = None) -> float:
        """
        Hàm phần thưởng lai (Hybrid Reward):
        R = Log_Return - (alpha * Volatility) - (beta * Drawdown) - (gamma * Turnover)
        """

        if v_old <= 0 or v_new <= 0:
            return -1.0 # Phạt nặng nếu cháy tài khoản

        # 1. Thành phần Lợi nhuận (Log Return)
        # Sử dụng log return giúp ổn định toán học hơn so với tỷ lệ % đơn thuần
        log_return = float(np.log(v_new / v_old))
        self.returns_history.append(log_return)

        # 2. Thành phần Rủi ro (Volatility Penalty)
        vol_penalty = 0.0
        # if len(self.returns_history) >= 2:
            # vol_penalty = float(np.std(self.returns_history))
        vol_penalty = float(np.std(self.returns_history)) if len(self.returns_history) > 2 else 0.0

        # 3. Thành phần Sụt giảm (Drawdown Penalty)
        self.max_portfolio_value = max(self.max_portfolio_value, v_new)
        drawdown = (self.max_portfolio_value - v_new) / self.max_portfolio_value
        drawdown_penalty = float(drawdown)

        # 4. Thành phần Chi phí (Transaction/Turnover Penalty)
        # Phạt nếu Agent giao dịch quá nhiều (overtrading)
        turnover_penalty = 0.0
        if trade_amounts is not None:
            # Tính tổng số cổ phiếu được giao dịch
            # turnover_penalty = float(np.sum(np.abs(trade_amounts))) / 1000
            turnover_ratio = np.sum(np.abs(trade_amounts) * execution_prices * 0.001) / v_new if v_new != 0 else 0
            turnover_penalty = float(turnover_ratio)

        """ trade_amounts ví dụ như là 2000 | chỉ số log return thường thì nằm trong khoảng từ -0.05 đến 0.05 (mức biến động |5%|), turn_over_penalty"""

        # TỔNG HỢP REWARD
        reward = log_return - (self.alpha * vol_penalty) - (self.beta * drawdown_penalty) - (self.gamma * turnover_penalty)

        # Clip reward để tránh nhiễu quá lớn cho Agent
        return float(np.clip(reward, -1.0, 1.0))

class SharpeRewardFunction:
    """
    Reward dua tren rolling Sharpe ratio cua excess return.

    Thay vi thuong truc tiep excess return,
    reward nay uu tien excess return ON DINH bang cach dung
    risk-adjusted signal: mean(excess) / std(excess).

    Thanh phan:
        1. Rolling Sharpe cua excess return (vs equal-weight + cash benchmark)
        2. Drawdown penalty: phat ty le voi muc sut giam tu dinh
        3. Turnover penalty: phat theo notional traded / NAV

    Ky hieu:
        portfolio_return = log(V_close / V_after_trade_open)
        benchmark_return = equal-weight basket + cash trong cung open->close
        excess_return = portfolio_return - benchmark_return
        rolling_sharpe = mean(excess_history) / std(excess_history)
    """

    def __init__(
        self,
        window: int = 30,
        sharpe_scale: float = 1.0,
        excess_scale: float = 50.0,
        drawdown_scale: float = 3.0,
        turnover_scale: float = 0.5,
    ):
        self.window = window
        self.sharpe_scale = sharpe_scale
        self.excess_scale = excess_scale
        self.drawdown_scale = drawdown_scale
        self.turnover_scale = turnover_scale

        self.excess_return_history = deque(maxlen=window)
        self.max_portfolio_value = -np.inf

    def reset(self):
        self.excess_return_history.clear()
        self.max_portfolio_value = -np.inf

    @staticmethod
    def _safe_log_return(v_from: float, v_to: float) -> float:
        if v_from <= 0 or v_to <= 0:
            return 0.0
        return float(np.log(v_to / v_from))

    @staticmethod
    def _equal_weight_cash_benchmark_log_return(
        execution_prices: np.ndarray | None,
        next_prices: np.ndarray | None,
    ) -> float:
        if execution_prices is None or next_prices is None:
            return 0.0

        exec_arr = np.asarray(execution_prices, dtype=np.float64)
        next_arr = np.asarray(next_prices, dtype=np.float64)
        valid = np.isfinite(exec_arr) & np.isfinite(next_arr) & (exec_arr > 0) & (next_arr > 0)
        if not np.any(valid):
            return 0.0

        stock_growth = next_arr[valid] / exec_arr[valid]
        benchmark_growth = (float(np.sum(stock_growth)) + 1.0) / (len(stock_growth) + 1.0)
        return float(np.log(max(benchmark_growth, 1e-12)))

    def calculate(
        self,
        v_old: float,
        v_new: float,
        trade_amounts: np.ndarray = None,
        *,
        execution_prices: np.ndarray | None = None,
        next_prices: np.ndarray | None = None,
        post_trade_value: float | None = None,
    ) -> float:
        if post_trade_value is None:
            post_trade_value = v_old

        if v_old <= 0 or post_trade_value <= 0 or v_new <= 0:
            return -5.0

        # 1. Excess return vs benchmark
        portfolio_log_return = self._safe_log_return(post_trade_value, v_new)
        benchmark_log_return = self._equal_weight_cash_benchmark_log_return(
            execution_prices=execution_prices,
            next_prices=next_prices,
        )
        excess_log_return = portfolio_log_return - benchmark_log_return
        self.excess_return_history.append(excess_log_return)

        # 2. Rolling Sharpe component
        sharpe_reward = 0.0
        if len(self.excess_return_history) >= 2:
            history = np.array(self.excess_return_history, dtype=np.float64)
            mean_excess = float(np.mean(history))
            std_excess = float(np.std(history))
            if std_excess > 1e-8:
                sharpe_reward = mean_excess / std_excess
            else:
                # Std ~0: if mean positive, reward; if negative, penalize
                sharpe_reward = np.sign(mean_excess) * 2.0

        # 3. Direct excess return (for immediate signal before window fills)
        direct_excess = self.excess_scale * excess_log_return

        # 4. Drawdown penalty
        self.max_portfolio_value = max(self.max_portfolio_value, v_new)
        drawdown_penalty = float(
            max(0.0, (self.max_portfolio_value - v_new) / max(self.max_portfolio_value, 1e-12))
        )

        # 5. Turnover penalty
        turnover_penalty = 0.0
        if trade_amounts is not None and execution_prices is not None:
            traded_value = float(
                np.sum(np.abs(np.asarray(trade_amounts, dtype=np.float64)) * np.asarray(execution_prices, dtype=np.float64))
            )
            turnover_penalty = traded_value / max(post_trade_value, 1e-12)

        reward = (
            self.sharpe_scale * sharpe_reward
            + direct_excess
            - self.drawdown_scale * drawdown_penalty
            - self.turnover_scale * turnover_penalty
        )

        return float(np.clip(reward, -5.0, 5.0))


class SharpePlusRewardFunction:
    """
    Reward nang cao dua tren Sharpe + 3 ky thuat bo sung:

    1. Holding bonus: thuong agent khi giu vi the on dinh (khong overtrade)
    2. Momentum alignment: thuong khi portfolio di cung chieu trend thi truong
    3. Asymmetric drawdown: phat nang hon khi drawdown vuot nguong

    Cong thuc:
        reward = sharpe_scale * rolling_sharpe
               + excess_scale * excess_return
               + holding_scale * holding_bonus
               + momentum_scale * momentum_bonus
               - drawdown_scale * asymmetric_dd_penalty
               - turnover_scale * turnover_ratio
    """

    def __init__(
        self,
        window: int = 30,
        sharpe_scale: float = 1.0,
        excess_scale: float = 80.0,
        holding_scale: float = 0.1,
        momentum_scale: float = 0.3,
        drawdown_scale: float = 3.0,
        turnover_scale: float = 0.3,
        dd_threshold: float = 0.05,
        dd_escalation: float = 3.0,
    ):
        self.window = window
        self.sharpe_scale = sharpe_scale
        self.excess_scale = excess_scale
        self.holding_scale = holding_scale
        self.momentum_scale = momentum_scale
        self.drawdown_scale = drawdown_scale
        self.turnover_scale = turnover_scale
        self.dd_threshold = dd_threshold
        self.dd_escalation = dd_escalation

        self.excess_return_history = deque(maxlen=window)
        self.max_portfolio_value = -np.inf

    def reset(self):
        self.excess_return_history.clear()
        self.max_portfolio_value = -np.inf

    @staticmethod
    def _safe_log_return(v_from: float, v_to: float) -> float:
        if v_from <= 0 or v_to <= 0:
            return 0.0
        return float(np.log(v_to / v_from))

    @staticmethod
    def _equal_weight_cash_benchmark_log_return(
        execution_prices: np.ndarray | None,
        next_prices: np.ndarray | None,
    ) -> float:
        if execution_prices is None or next_prices is None:
            return 0.0
        exec_arr = np.asarray(execution_prices, dtype=np.float64)
        next_arr = np.asarray(next_prices, dtype=np.float64)
        valid = np.isfinite(exec_arr) & np.isfinite(next_arr) & (exec_arr > 0) & (next_arr > 0)
        if not np.any(valid):
            return 0.0
        stock_growth = next_arr[valid] / exec_arr[valid]
        benchmark_growth = (float(np.sum(stock_growth)) + 1.0) / (len(stock_growth) + 1.0)
        return float(np.log(max(benchmark_growth, 1e-12)))

    def calculate(
        self,
        v_old: float,
        v_new: float,
        trade_amounts: np.ndarray = None,
        *,
        execution_prices: np.ndarray | None = None,
        next_prices: np.ndarray | None = None,
        post_trade_value: float | None = None,
    ) -> float:
        if post_trade_value is None:
            post_trade_value = v_old

        if v_old <= 0 or post_trade_value <= 0 or v_new <= 0:
            return -5.0

        # --- 1. Excess return vs benchmark ---
        portfolio_log_return = self._safe_log_return(post_trade_value, v_new)
        benchmark_log_return = self._equal_weight_cash_benchmark_log_return(
            execution_prices=execution_prices,
            next_prices=next_prices,
        )
        excess_log_return = portfolio_log_return - benchmark_log_return
        self.excess_return_history.append(excess_log_return)

        # --- 2. Rolling Sharpe component ---
        sharpe_reward = 0.0
        if len(self.excess_return_history) >= 2:
            history = np.array(self.excess_return_history, dtype=np.float64)
            mean_excess = float(np.mean(history))
            std_excess = float(np.std(history))
            if std_excess > 1e-8:
                sharpe_reward = mean_excess / std_excess
            else:
                sharpe_reward = np.sign(mean_excess) * 2.0

        # --- 3. Direct excess return ---
        direct_excess = self.excess_scale * excess_log_return

        # --- 4. Turnover ratio ---
        turnover_ratio = 0.0
        if trade_amounts is not None and execution_prices is not None:
            traded_value = float(
                np.sum(
                    np.abs(np.asarray(trade_amounts, dtype=np.float64))
                    * np.asarray(execution_prices, dtype=np.float64)
                )
            )
            turnover_ratio = traded_value / max(post_trade_value, 1e-12)

        # --- 5. Holding bonus (KY THUAT NANG CAO 1) ---
        # Thuong agent khi giu vi the on dinh, turnover_ratio thap -> bonus cao
        holding_bonus = max(0.0, 1.0 - turnover_ratio)

        # --- 6. Momentum alignment (KY THUAT NANG CAO 2) ---
        # Thuong khi portfolio return cung chieu benchmark return
        momentum_bonus = 0.0
        if abs(benchmark_log_return) > 1e-10:
            # Positive khi cung chieu, negative khi nguoc chieu
            alignment = portfolio_log_return * benchmark_log_return
            momentum_bonus = max(0.0, alignment) / (abs(benchmark_log_return) + 1e-8)

        # --- 7. Asymmetric drawdown penalty (KY THUAT NANG CAO 3) ---
        self.max_portfolio_value = max(self.max_portfolio_value, v_new)
        drawdown = float(
            max(0.0, (self.max_portfolio_value - v_new) / max(self.max_portfolio_value, 1e-12))
        )
        dd_penalty = drawdown
        if drawdown > self.dd_threshold:
            # Phat tang nhanh khi vuot nguong
            dd_penalty *= (1.0 + self.dd_escalation * (drawdown - self.dd_threshold))

        # --- TONG HOP ---
        reward = (
            self.sharpe_scale * sharpe_reward
            + direct_excess
            + self.holding_scale * holding_bonus
            + self.momentum_scale * momentum_bonus
            - self.drawdown_scale * dd_penalty
            - self.turnover_scale * turnover_ratio
        )

        return float(np.clip(reward, -5.0, 5.0))


def build_reward_function(name: str = "sharpe", **kwargs):
    normalized = str(name or "sharpe").strip().lower()
    if normalized in {"sharpe", "sharpe_reward", "sharpereward"}:
        return SharpeRewardFunction(**kwargs)
    if normalized in {"sharpe_plus", "sharpeplus", "sharpe_plus_reward"}:
        return SharpePlusRewardFunction(**kwargs)
    if normalized in {"advanced", "legacy", "advanced_reward"}:
        return AdvancedRewardFunction(**kwargs)
    raise ValueError(f"Unsupported reward function: {name}")
