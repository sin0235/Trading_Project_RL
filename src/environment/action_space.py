import numpy as np


def decode_discrete_action(action, n_stocks,
                           cash, holdings, prices) -> np.ndarray:
    """
    Chuyen discrete action thanh vector trade_amounts.
    action space = n_stocks [0, 1, 2, 0, 1, 2, ...]

        action_i = 0 -> sell all
        action_i = 1 -> hold
        action_i = 2 -> buy 0.2 * NAV

    Return:
        trade_amounts: ndarray (n_stocks,)
            So co phieu muon giao dich (am = sell, duong = buy)
    """

    trade_amounts = np.zeros(n_stocks, dtype=np.int32)

    NAV = cash + np.sum(holdings * prices)
    cash_for_trades = min(NAV * 0.2, cash)
    n_stocks_to_buy = np.sum(action == 2)
    cash_per_stock = cash_for_trades // n_stocks_to_buy

    for i in range(n_stocks):
        if action[i] == 0:
            # Sell all
            trade_amounts[i] = -holdings[i]
        elif action[i] == 2:
            # Buy 0.2 * NAV
            trade_amounts[i] = int(cash_per_stock / prices[i])

    return trade_amounts


def decode_continuous_action(action: np.ndarray,
                             ratio: np.ndarray,cash, holdings, prices) -> np.ndarray:
    """
    Chuyen continuous action [0,1] (N+1) thanh so luong co phieu.

    action_i = rate for stock i, trong do:
    0 <= action_i <= 1.0

    Return:
        trade_amounts: ndarray (n_stocks,)
    """
    NAV = cash + np.sum(holdings * prices)
    action_trade = action[:-1] - ratio[:-1]

    trade_amounts = action_trade * NAV / prices
    return trade_amounts.astype(np.int32)
    
    


def apply_constraints(trade_amounts: np.ndarray,
                      cash: float,
                      holdings: np.ndarray,
                      prices: np.ndarray,
                      fee_rate: float,
                      min_shares: int) -> np.ndarray:
    """
    Dieu chinh trade_amounts de:
        - Khong ban qua so dang giu
        - Khong mua qua so tien mat
        - Khong tao cash am
    """

    trade_amounts = trade_amounts // min_shares
    n_stocks = len(trade_amounts)
    available_cash = cash

    cash_for_trade = np.sum([x * prices[i] * (1 + fee_rate) if x > 0 else -(x * prices[i] * (1 - fee_rate)) for i, x in enumerate(trade_amounts)])
    if cash_for_trade > available_cash:
        scale = available_cash / cash_for_trade
        trade_amounts = (np.array([x * scale if x > 0 else x for x in trade_amounts]) // 100).astype(np.int32)


    return trade_amounts