import numpy as np


def decode_discrete_action(action: int,
                           n_stocks: int,
                           max_shares: int,
                           k: int = 3) -> np.ndarray:
    """
    Chuyen discrete action thanh vector trade_amounts.

    Action space = k * n_stocks
    k = 3 -> [sell, hold, buy]

    Return:
        trade_amounts: ndarray (n_stocks,)
            So co phieu muon giao dich (am = sell, duong = buy)
    """

    trade_amounts = np.zeros(n_stocks, dtype=np.int32)

    stock_idx = action // k
    action_type = action % k

    if stock_idx >= n_stocks:
        return trade_amounts

    if action_type == 0:        # SELL
        trade_amounts[stock_idx] = -max_shares
    elif action_type == 2:      # BUY
        trade_amounts[stock_idx] = max_shares
    # HOLD -> 0

    return trade_amounts


def decode_continuous_action(action: np.ndarray,
                             max_shares: int) -> np.ndarray:
    """
    Chuyen continuous action [-1,1]^N thanh so luong co phieu.

    action_i > 0 -> buy
    action_i < 0 -> sell

    Return:
        trade_amounts: ndarray (n_stocks,)
    """

    action = np.clip(action, -1.0, 1.0)

    trade_amounts = action * max_shares

    return trade_amounts.astype(np.int32)


def apply_constraints(trade_amounts: np.ndarray,
                      cash: float,
                      holdings: np.ndarray,
                      prices: np.ndarray,
                      fee_rate: float) -> np.ndarray:
    """
    Dieu chinh trade_amounts de:
        - Khong ban qua so dang giu
        - Khong mua qua so tien mat
        - Khong tao cash am
    """

    trade_amounts = trade_amounts.copy()
    n_stocks = len(trade_amounts)

    # ===== 1️ SELL constraint =====
    for i in range(n_stocks):
        if trade_amounts[i] < 0:
            max_sell = holdings[i]
            trade_amounts[i] = -min(abs(trade_amounts[i]), max_sell)

    # ===== 2️ BUY constraint =====
    available_cash = cash

    for i in range(n_stocks):
        if trade_amounts[i] > 0:

            shares = trade_amounts[i]
            value = shares * prices[i]
            fee = value * fee_rate
            total_cost = value + fee

            if total_cost > available_cash:
                # Tinh so co phieu toi da co the mua
                max_affordable = int(
                    available_cash / (prices[i] * (1 + fee_rate))
                )
                trade_amounts[i] = max(0, max_affordable)

                # cap nhat lai gia tri
                shares = trade_amounts[i]
                value = shares * prices[i]
                fee = value * fee_rate
                total_cost = value + fee

            available_cash -= total_cost

    return trade_amounts