# StateSpace API Reference

## Tong quan

`StateSpace` quan ly toan bo du lieu thi truong va cung cap state vector cho RL agent tai moi buoc thoi gian.

```
State = [Market Information] + [Portfolio Information]
```

**File:** `src/environment/state_space.py`

---

## Cau truc State

### Market Information

Tai moi buoc `t`, lay `window_size` ngay lich su cho tat ca ma co phieu, moi ngay gom cac features:

| Feature | Kieu normalize | Khoang gia tri | Mo ta |
|---|---|---|---|
| `close_norm` | Z-score (rolling 60) | ~ [-3, 3] | Gia dong cua chuan hoa |
| `return_1d` | Raw | ~ [-0.07, 0.07] | Loi nhuan 1 ngay |
| `return_5d` | Raw | ~ [-0.15, 0.15] | Loi nhuan 5 ngay |
| `macd` | Chia cho rolling std | ~ [-1, 1] | MACD histogram chuan hoa |
| `rsi` | Chia 100 | [0, 1] | Relative Strength Index |
| `adx` | Chia 100 | [0, 1] | Average Directional Index |
| `volume_norm` | Z-score (rolling 60) | ~ [-3, 3] | Volume chuan hoa |

**Shape market data noi bo:** `(T, n_stocks, n_features)` -- voi `T` la tong so ngay chung giua cac ma.

### Portfolio Information

| Feature | Cong thuc | Khoang |
|---|---|---|
| `cash_ratio` | `cash / portfolio_value` | [0, 1] |
| `holdings_ratio[i]` | `(holdings[i] * prices[i]) / portfolio_value` | [0, 1] |

Tong cac ratio luon = 1.0.

### Dimension

```
market_dim  = window_size * n_stocks * n_features
portfolio_dim = 1 + n_stocks
state_dim   = market_dim + portfolio_dim
```

Voi config hien tai (30 tickers, window=30, 7 features):

| Thanh phan | Gia tri |
|---|---|
| market_dim | 30 x 30 x 7 = **6300** |
| portfolio_dim | 1 + 30 = **31** |
| **state_dim** | **6331** |

---

## Khoi tao

```python
from src.constants import TICKERS, WINDOW_SIZE, DATA_PATH, FEATURES
from src.environment.state_space import StateSpace

state_space = StateSpace(
    tickers=TICKERS,          # List[str], mac dinh 30 ma VN30
    window_size=WINDOW_SIZE,  # int, mac dinh 30
    data_path=DATA_PATH,      # str, mac dinh "data/processed"
    features=FEATURES,        # List[str], mac dinh 7 features
    mode="flatten"            # "flatten" (DQN) hoac "sequential" (PPO+LSTM)
)
```

Khi khoi tao, `_load_data()` tu dong:
1. Doc tat ca CSV tu `data/processed/{ticker}.csv`
2. Tim ngay giao dich chung (intersection) giua 30 ma
3. Tao `self.data` shape `(T, 30, 7)` va `self.close_prices` shape `(T, 30)`

---

## API Methods

### `get_state(t, cash, holdings) -> np.ndarray`

**Day la method chinh, goi trong moi step cua environment.**

| Param | Type | Mo ta |
|---|---|---|
| `t` | `int` | Index thoi gian, yeu cau `t >= window_size - 1` |
| `cash` | `float` | Tien mat hien co |
| `holdings` | `np.ndarray` | So luong co phieu dang giu, shape `(n_stocks,)` |

**Tra ve:**
- Mode `flatten`: `np.ndarray` shape `(state_dim,)`, dtype `float32`, clip trong [-5, 5]
- Mode `sequential`: tuple `(market_state, portfolio_state)` voi market shape `(window_size, n_stocks, n_features)` va portfolio shape `(portfolio_dim,)`

**Vi du:**

```python
t = 29  # buoc dau tien kha dung (can 30 ngay lich su)
cash = 1_000_000_000.0
holdings = np.zeros(state_space.n_stocks)

state = state_space.get_state(t, cash, holdings)
# state.shape = (6331,) voi mode flatten
```

### `get_market_state(t) -> np.ndarray`

Lay thong tin thi truong (chi phan market, khong co portfolio).

- Mode `flatten`: `np.ndarray` shape `(market_dim,)`
- Mode `sequential`: `np.ndarray` shape `(window_size, n_stocks, n_features)`

### `get_portfolio_state(cash, holdings, prices) -> np.ndarray`

Tinh portfolio ratios. Tra ve `np.ndarray` shape `(portfolio_dim,)` = `(1 + n_stocks,)`.

Neu `portfolio_value <= 0` tra ve vector zero.

### `get_prices(t) -> np.ndarray`

Lay gia close tai buoc `t`. Tra ve shape `(n_stocks,)`.

### Properties

| Property | Tra ve | Mo ta |
|---|---|---|
| `observation_shape` | `tuple` | `(state_dim,)` hoac `(window_size, n_stocks, n_features)` |
| `max_steps` | `int` | So buoc toi da trong 1 episode = `n_days - window_size` |

### Attributes

| Attribute | Type | Mo ta |
|---|---|---|
| `data` | `np.ndarray` | Shape `(T, n_stocks, n_features)`, du lieu features |
| `close_prices` | `np.ndarray` | Shape `(T, n_stocks)`, gia close goc |
| `dates` | `DatetimeIndex` | Danh sach ngay chung |
| `n_days` | `int` | Tong so ngay du lieu |
| `n_stocks` | `int` | So luong ma co phieu |
| `n_features` | `int` | So luong features |
| `state_dim` | `int` | Tong chieu cua state vector |
| `market_dim` | `int` | Chieu cua phan market |
| `portfolio_dim` | `int` | Chieu cua phan portfolio |

---

## Luong su dung trong TradingEnv

```
Khoi tao:
    state_space = StateSpace(tickers=TICKERS)

Reset episode:
    t = state_space.window_size - 1   (= 29)
    cash = INITIAL_CASH
    holdings = np.zeros(n_stocks)
    state = state_space.get_state(t, cash, holdings)

Moi step:
    1. Agent nhan state, chon action
    2. Environment thuc thi action -> cap nhat cash, holdings
    3. t += 1
    4. next_state = state_space.get_state(t, cash, holdings)
    5. Tinh reward
    6. done = (t >= state_space.window_size - 1 + state_space.max_steps - 1)

Ket thuc khi:
    t >= state_space.n_days - 1
```

---

## Luu y khi tich hop

1. **Index `t` bat dau tu `window_size - 1`** (= 29), khong phai tu 0. Vi can 30 ngay lich su phia truoc.

2. **Clip [-5, 5]** duoc ap dung o mode `flatten`. Tat ca features da duoc normalize ve khoang nho, nhung van clip de phong outlier.

3. **Mode `sequential`** tra ve tuple, khong phai 1 array. Network can xu ly market qua LSTM/CNN rieng, roi concat voi portfolio truoc khi dua vao policy head.

4. **`get_state` tu dong lay gia** tu `self.close_prices[t]`, khong can truyen `prices` rieng.

5. **Du lieu phai duoc xu ly truoc** bang `DataProcessor.process()` va luu vao `data/processed/` truoc khi dung `StateSpace`.
