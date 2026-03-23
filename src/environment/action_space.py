import numpy as np

def decode_discrete_action(action, n_stocks, min_shares, cash, holdings, prices,
                           fee_rate: float = 0.001) -> np.ndarray:
    """
    Action 0: Sell all, 1: Hold, 2: Buy
    """
    # Dùng int64 để tránh tràn số khi holdings rất lớn (train dài, giá thấp).
    trade_amounts = np.zeros(n_stocks, dtype=np.int64)
    
    # 1. Xử lý lệnh BÁN trước để giải phóng tiền
    for i in range(n_stocks):
        if action[i] == 0:
            trade_amounts[i] = -holdings[i]

    # 2. Tính toán tiền khả dụng để MUA
    # Giả định: Tiền bán thu về được dùng ngay trong step (phù hợp slide của bạn)
    sell_proceeds = np.sum([
        abs(trade_amounts[i]) * prices[i] * (1 - fee_rate)
        for i in range(n_stocks) if trade_amounts[i] < 0
    ])
    total_available_cash = cash + sell_proceeds

    
    # Giới hạn mỗi lần mua tối đa 20% NAV hoặc tổng tiền mặt hiện có
    NAV = cash + np.sum(holdings * prices)
    cash_limit = min(total_available_cash, NAV * 0.2)
    n_stocks_to_buy = np.sum(action == 2)

    if n_stocks_to_buy > 0:
        # Chia đều tiền cho các mã muốn mua, giới hạn bởi 20% NAV tổng 
        budget_per_stock = int(cash_limit / n_stocks_to_buy)
        for i in range(n_stocks):
            if action[i] == 2:
                # Tính số lượng cổ phiếu (chưa xét lô 100, sẽ xét ở apply_constraints)
                trade_amounts[i] = int(budget_per_stock / (prices[i] * (1 + fee_rate)))

    return trade_amounts

def decode_continuous_action(
    action: np.ndarray,
    ratio: np.ndarray,
    cash,
    holdings,
    prices,
    trade_deadband: float = 0.0,
    max_weight_change_per_step: float = 1.0,
) -> np.ndarray:
    """
    action: vector tỷ trọng mục tiêu [0, 1] cho N stocks + 1 cash
    ratio: vector tỷ trọng hiện tại
    """
    NAV = cash + np.sum(holdings * prices)
    if NAV <= 0:
        return np.zeros_like(holdings, dtype=np.int64)

    # Tỷ trọng mục tiêu của các mã (bỏ phần tử cuối là cash)
    target_ratios = np.asarray(action[:-1], dtype=np.float64)
    current_ratios = np.asarray(ratio[:-1], dtype=np.float64)

    diff_ratio = target_ratios - current_ratios
    if trade_deadband > 0:
        diff_ratio[np.abs(diff_ratio) < trade_deadband] = 0.0

    if max_weight_change_per_step < 1.0:
        diff_ratio = np.clip(
            diff_ratio,
            -max_weight_change_per_step,
            max_weight_change_per_step,
        )

    # Lượng tiền cần dịch chuyển cho mỗi mã.
    # Dương là mua thêm, âm là bán bớt.
    trade_amounts = (diff_ratio * NAV) / prices
    return trade_amounts.astype(np.int64)

def apply_constraints(trade_amounts: np.ndarray, cash: float, 
                      holdings: np.ndarray, prices: np.ndarray, 
                      fee_rate: float, min_shares: int) -> np.ndarray:
    """
    Ràng buộc lô 100 và không âm tiền/cổ phiếu
    """
    adjusted_trade = np.asarray(trade_amounts, dtype=np.int64).copy()
    
    # 1. Ràng buộc BÁN: Không bán quá số lượng đang có và phải chia hết cho lô 100
    for i in range(len(adjusted_trade)):
        if adjusted_trade[i] < 0:
            # Không bán quá holdings
            can_sell = min(abs(adjusted_trade[i]), holdings[i])
            # Làm tròn xuống theo lô (ví dụ 150 -> 100)
            adjusted_trade[i] = -(int(can_sell // min_shares) * min_shares)

    # 2. Tính tiền mặt sau khi bán
    sell_value = np.sum([abs(adjusted_trade[i]) * prices[i] * (1 - fee_rate) for i in range(len(adjusted_trade)) if adjusted_trade[i] < 0])
    current_cash = cash + sell_value
    
    # 3. Ràng buộc MUA: Không mua quá số tiền đang có
    buy_indices = np.where(adjusted_trade > 0)[0]
    if len(buy_indices) > 0:
        # Tính tổng vốn cần để thực hiện tất cả lệnh mua
        total_buy_cost = np.sum([adjusted_trade[i] * prices[i] * (1 + fee_rate) for i in buy_indices])
        
        # Nếu thiếu tiền, scale đều tất cả lệnh mua
        if total_buy_cost > current_cash:
            scale = current_cash / total_buy_cost
            for i in buy_indices:
                adjusted_trade[i] = int((adjusted_trade[i] * scale) // min_shares) * min_shares
        else:
            # Vẫn cần làm tròn về lô 100 cho các lệnh mua đủ tiền
            for i in buy_indices:
                adjusted_trade[i] = int(adjusted_trade[i] // min_shares) * min_shares

    return adjusted_trade.astype(np.int64)
