import numpy as np
from collections import deque

'''
Hàm phần thưởng dùng để test và debug, sau 
sẽ được thay thế bằng hàm phần thưởng thực tế sẽ được Sáng triển khai sau
'''
# class RewardFunction:
#     def __init__(self, reward_type: str = "simple", window: int = 30):
#         self.reward_type = reward_type
#         self.window = window
#         self.returns_history = deque(maxlen=window)

#     def reset(self):
#         self.returns_history.clear()

#     def calculate(self, v_old: float, v_new: float) -> float:
#         if self.reward_type == "simple":
#             return self._simple(v_old, v_new)
#         elif self.reward_type == "sharpe":
#             return self._sharpe(v_old, v_new)
#         else:
#             raise ValueError(f"Unknown reward type: {self.reward_type}")

#     def _simple(self, v_old: float, v_new: float) -> float:
#         """r_t = (V_new - V_old) / V_old, clip [-0.1, 0.1]"""
#         if v_old <= 0:
#             return 0.0
#         ret = (v_new - v_old) / v_old
#         return float(np.clip(ret, -0.1, 0.1))

#     def _sharpe(self, v_old: float, v_new: float) -> float:
#         """Sharpe ratio rolling window: mean(returns) / std(returns)"""
#         if v_old <= 0:
#             return 0.0

#         ret = (v_new - v_old) / v_old
#         self.returns_history.append(ret)

#         if len(self.returns_history) < 2:
#             return float(np.clip(ret, -0.1, 0.1))

#         returns = np.array(self.returns_history)
#         std = returns.std()
#         if std < 1e-8:
#             return 0.0

#         sharpe = returns.mean() / std
#         return float(np.clip(sharpe, -2.0, 2.0))


""" Thiết kế reward function để đánh giá hiệu quả của agent trong trading"""
class AdvancedRewardFunction:
    def __init__(
        self, 
        window: int = 30, 
        alpha: float = 0.1,  # Trọng số cho Volatility (Rủi ro)
        beta: float = 0.5,   # Trọng số cho Max Drawdown (Sụt giảm)
        gamma: float = 0.01  # Trọng số cho Transaction Penalty (Chi phí)
    ):
        self.window = window
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.returns_history = deque(maxlen=window)
        self.max_portfolio_value = -np.inf

    def reset(self):
        self.returns_history.clear()
        self.max_portfolio_value = -np.inf

    def calculate(self, v_old: float, v_new: float, trade_amounts: np.ndarray = None) -> float:
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
        if len(self.returns_history) >= 2:
            vol_penalty = float(np.std(self.returns_history))

        # 3. Thành phần Sụt giảm (Drawdown Penalty)
        self.max_portfolio_value = max(self.max_portfolio_value, v_new)
        drawdown = (self.max_portfolio_value - v_new) / self.max_portfolio_value
        drawdown_penalty = float(drawdown)

        # 4. Thành phần Chi phí (Transaction/Turnover Penalty)
        # Phạt nếu Agent giao dịch quá nhiều (overtrading)
        turnover_penalty = 0.0
        if trade_amounts is not None:
            # Tính tổng số cổ phiếu được giao dịch
            turnover_penalty = float(np.sum(np.abs(trade_amounts))) / 1000

        """ trade_amounts ví dụ như là 2000 | chỉ số log return thường thì nằm trong khoảng từ -0.05 đến 0.05 (mức biến động |5%|), turn_over_penalty"""

        # TỔNG HỢP REWARD
        reward = log_return - (self.alpha * vol_penalty) - (self.beta * drawdown_penalty) - (self.gamma * turnover_penalty)

        # Clip reward để tránh nhiễu quá lớn cho Agent
        return float(np.clip(reward, -1.0, 1.0))