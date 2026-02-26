import numpy as np
from collections import deque

'''
Hàm phần thưởng dùng để test và debug, sau 
sẽ được thay thế bằng hàm phần thưởng thực tế sẽ được Sáng triển khai sau
'''
class RewardFunction:
    def __init__(self, reward_type: str = "simple", window: int = 30):
        self.reward_type = reward_type
        self.window = window
        self.returns_history = deque(maxlen=window)

    def reset(self):
        self.returns_history.clear()

    def calculate(self, v_old: float, v_new: float) -> float:
        if self.reward_type == "simple":
            return self._simple(v_old, v_new)
        elif self.reward_type == "sharpe":
            return self._sharpe(v_old, v_new)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _simple(self, v_old: float, v_new: float) -> float:
        """r_t = (V_new - V_old) / V_old, clip [-0.1, 0.1]"""
        if v_old <= 0:
            return 0.0
        ret = (v_new - v_old) / v_old
        return float(np.clip(ret, -0.1, 0.1))

    def _sharpe(self, v_old: float, v_new: float) -> float:
        """Sharpe ratio rolling window: mean(returns) / std(returns)"""
        if v_old <= 0:
            return 0.0

        ret = (v_new - v_old) / v_old
        self.returns_history.append(ret)

        if len(self.returns_history) < 2:
            return float(np.clip(ret, -0.1, 0.1))

        returns = np.array(self.returns_history)
        std = returns.std()
        if std < 1e-8:
            return 0.0

        sharpe = returns.mean() / std
        return float(np.clip(sharpe, -2.0, 2.0))
