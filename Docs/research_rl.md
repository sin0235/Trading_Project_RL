# tôi đang có 1 project môn học RL về đề tài trading, 2 thuật toán để giải quyết bài toán - so sánh giữa 2 cách  , sử dụng dataset của việt nam, giúp tôi nghiên cứu các nội dung:

1. Mô tả bài toán 2. Mô tả dataset sẽ sử dụng 3. Thiết kế và mô tả ý tưởng thuật toán 4. Mô tả môi trường 5. Hướng triển khai thuật toán đảm bảo chi tiết, đủ dung lượng, có giải thích chi tiết, phù hợp với việt nam

Dưới đây là một khung nghiên cứu + thiết kế khá đầy đủ cho project RL trading dùng dữ liệu Việt Nam, với hai thuật toán so sánh (gợi ý: DQN vs PPO, vì trùng với các paper bạn gửi và có sẵn implementation trong stable-baselines3).

***

## 0. Gợi ý tổng thể cho project

- **Bài toán**: trading cổ phiếu/ETF Việt Nam theo dữ liệu lịch sử, agent chọn hành động mỗi ngày để tối đa hóa lợi nhuận (hoặc Sharpe).
- **Hai thuật toán so sánh**:
    - **DQN** (Deep Q-Network): value-based, discrete action, phù hợp single-stock / vị thế rời rạc ±1, 0.[^1][^2]
    - **PPO** (Proximal Policy Optimization): policy gradient, actor–critic, ổn định, hỗ trợ action liên tục (tỉ lệ phân bổ vốn), rất hay dùng trong trading thực nghiệm.[^3][^2]
- **Dataset Việt Nam**: nên chọn VN30 daily OHLCV 5–10 năm để có đủ data, ví dụ:
    - Kaggle *“Stock Prices \& Volume VN30 Index Vietnam”* (10 năm, 30 mã VN30).[^4]
    - Hoặc tự crawl từ Stockbiz / HSX, HNX (Historical Data).[^5]
- **Môi trường**: custom Gym environment (single-asset hoặc multi-asset), reward theo thay đổi giá trị portfolio sau phí giao dịch, không dùng margin, không short (phù hợp luật Việt Nam hiện tại).
- **So sánh**: train DQN và PPO trên cùng môi trường, cùng train/test split, so sánh các metric: cumulative return, Sharpe, max drawdown, win rate, v.v.

Các phần dưới lần lượt trả lời 5 mục bạn yêu cầu.

***

## 1. Mô tả bài toán trading dưới góc nhìn RL

### 1.1. Bài toán thực tế

- Bạn có dữ liệu lịch sử của **một hoặc nhiều cổ phiếu Việt Nam**, ví dụ 30 mã VN30 trên HOSE.
- Mỗi ngày giao dịch $t$, agent được quan sát một vector trạng thái (giá, volume, indicator, vị thế hiện tại, tiền mặt, v.v.).
- Agent chọn **hành động trading** (ví dụ: mua, bán, giữ hoặc điều chỉnh tỉ lệ phân bổ vốn).
- Sau khi hành động, portfolio thay đổi, sinh ra **reward** (lãi/lỗ sau phí).
- Mục tiêu: học chính sách $\pi(a \mid s)$ để **tối đa hóa lợi nhuận tích lũy** hoặc một hàm mục tiêu liên quan (Sharpe ratio, risk-adjusted return).[^2][^3]


### 1.2. Mô hình hóa thành MDP

Ta mô hình hóa bài toán như một **Markov Decision Process (MDP)**:[^2]

- **State $S_t$**: mô tả “ảnh chụp” thị trường và portfolio tại ngày $t$, ví dụ:
    - Dữ liệu giá: close, open, high, low, volume của 1–N mã trong $L$ ngày gần nhất.
    - Indicators: MACD, RSI, CCI, ADX…[^3][^2]
    - Thông tin portfolio: tiền mặt, số lượng cổ phiếu đang nắm giữ từng mã, giá trị portfolio, v.v.
- **Action $A_t$**:
    - Single-stock, discrete: $\{-1, 0, +1\}$ tương ứng Bán 1 đơn vị, Giữ, Mua 1 đơn vị (hoặc dùng action là target position: short full, flat, long full).[^1][^2]
    - Multi-stock + PPO: vector liên tục $[-1, 1]^N$ biểu diễn tỉ lệ phân bổ cho mỗi mã (PPO rất phù hợp).[^3]
    - Trong thực tế Việt Nam **không short selling phổ thông**, nên trong project có thể:
        - Hoặc **giới hạn action $\{0, +1\}$** (flat hoặc long).
        - Hoặc vẫn cho short để nghiên cứu RL (nói rõ trong báo cáo là giả định lý tưởng, không đúng 100% luật).
- **Transition**: lấy giá ngày $t+1$, cập nhật portfolio (tiền + số cổ phiếu), tính reward.
- **Reward $R_t$**:
    - Cách đơn giản, single-asset:

$$
R_t = \frac{V_{t+1} - V_t}{V_t}
$$

với $V_t$ là giá trị portfolio sau ngày $t$, đã trừ phí giao dịch.
    - Multi-asset: tương tự nhưng $V_t = \text{cash}_t + \sum_i h_{i,t} \cdot p_{i,t}$.[^2][^3]
    - Có thể **thêm transaction cost** (0.1–0.2% mỗi chiều buy/sell để gần với market Việt Nam).
- **Objective**:
    - Tối đa hóa tổng reward chiết khấu:

$$
\mathbb{E}\left[\sum_{t=0}^{T} \gamma^t R_t\right]
$$
    - Hoặc đưa Sharpe / Sortino vào reward (các paper cao cấp đã làm, nhưng cho project, tối đa hóa cumulative PnL là đủ).[^3][^2]


### 1.3. Baseline để so sánh

Để đánh giá RL có “xứng đáng” không, nên thêm 1–2 baseline:

- **Buy \& Hold**: mua VN30 Index hoặc 1 mã từ đầu đến cuối (benchmark phổ biến).[^2]
- **Chiến lược kỹ thuật đơn giản**: ví dụ MACD crossover hoặc time-series momentum (sign of 1-year return) như trong Zhang et al.[^2]

***

## 2. Mô tả dataset Việt Nam sẽ sử dụng

### 2.1. Nguồn dữ liệu phù hợp Việt Nam

Một số nguồn bạn có thể dùng:

1. **Kaggle – VN30**: *Stock Prices \& Volume VN30 Index Vietnam*
    - Gồm dữ liệu giá và khối lượng cho 30 mã VN30, khoảng 10 năm.[^4]
    - Thường có các cột: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`.
2. **Kaggle – Viet Nam Stock Market Prediction**
    - 30 mã, dữ liệu 2021, dùng từ package `vnquant`, có các file `price_train`, `finance_train`, `business_train`.[^6]
3. **Stockbiz.vn / HSX / HNX**
    - Trang `HistoricalIndices` hoặc `HistoricalPrices` cho từng mã, tải về CSV (VN-Index, HNX-Index, từng cổ phiếu).[^7][^5]

Cho project môn học, cách dễ nhất là **dùng Kaggle VN30** hoặc một subset (5–10 mã) để giảm độ phức tạp mà vẫn đúng “dữ liệu Việt Nam”.

### 2.2. Cấu trúc dữ liệu đề xuất

Giả sử bạn dùng VN30 Kaggle:

- **Mức độ thời gian**: daily (T+1 đủ cho RL low-frequency).
- **Các trường chính**:
    - `date` – ngày giao dịch.
    - `ticker` – mã cổ phiếu.
    - `open`, `high`, `low`, `close` – giá.
    - `volume` – khối lượng.
- **Tiền xử lý**:
    - Lọc khoảng thời gian đủ dài, ví dụ 2013–2023 (khoảng 10 năm).
    - Loại bỏ ngày nghỉ, chỉ giữ ngày có giao dịch đồng thời cho tất cả mã đã chọn.
    - Xử lý missing (forward fill nếu hợp lý).
    - Chuẩn hóa/scale feature (z-score hoặc MinMax cho các indicator).
    - Nếu dùng multi-asset: pivot dữ liệu thành dạng wide (`date` x features x ticker).


### 2.3. Xây dựng feature (state)

Dựa theo các paper bạn gửi, nhất là Zhang et al. (2019) và Zou et al. (2023):[^3][^2]

- **Giá \& returns**:
    - Normalized close price.
    - Returns trong các khung: 1 ngày, 5, 20, 60, 252 ngày.
- **Technical indicators**:
    - **MACD** (12–26–9 hoặc tương đương, chuẩn hóa theo volatility).[^3][^2]
    - **RSI** (14 hoặc 30 ngày).[^2][^3]
    - **CCI**, **ADX** nếu đi theo CLSTM-PPO paper.[^3]
- **Thông tin portfolio**:
    - Tiền mặt (chuẩn hóa theo vốn ban đầu).
    - Tỉ lệ vốn đang nằm trên từng mã (số cổ phiếu * giá / tổng vốn).
- **Cửa sổ thời gian**:
    - Lấy **L ngày gần nhất** làm 1 state, ví dụ $L = 30$ hoặc 60:
        - Chen (2019) dùng 20 ngày close làm state cho DQN/DRQN.[^1]
        - Zhang (2019) dùng 60 quan sát cho mỗi feature.[^2]
        - Zou (2023) test time-window LSTM = 5, 15, 30, 50 và thấy 30 ngày tốt nhất.[^3]

Trong project, L=30 là hợp lý: đủ dài để học xu hướng mà vẫn train được nhanh.

### 2.4. Chia tập train/validation/test

- Ví dụ với dữ liệu 2013–2023:
    - **Train**: 2013–2018.
    - **Validation**: 2019–2020 (tuning hyperparam).
    - **Test (out-of-sample)**: 2021–2023.
- Không shuffle theo thời gian; RL là time-series nên phải giữ thứ tự.

***

## 3. Thiết kế \& mô tả ý tưởng hai thuật toán

Giả sử bạn chọn:

- **Thuật toán 1**: DQN (Deep Q-Network).
- **Thuật toán 2**: PPO với LSTM feature extractor (lược giản từ CLSTM-PPO).[^3]


### 3.1. Thuật toán 1 – DQN cho trading

Tham khảo Chen (2019) và Zhang (2019):[^1][^2]

**Ý tưởng chính**:

- Học hàm giá trị hành động $Q_\theta(s, a)$ ≈ kỳ vọng tổng reward nếu chọn action $a$ trong state $s$ rồi theo policy greedy sau đó.
- Action discrete, rất hợp với **bài toán 1 mã hoặc ít mã**:
    - $a \in \{-1, 0, +1\}$ hoặc $\{0, +1\}$ (nếu muốn tuân luật VN không short).
    - Có thể giải thích là **vị thế mục tiêu** (target position): short full / flat / long full.[^2]
- Ở mỗi bước:
    - Dùng network dự đoán $Q(s_t, a)$ cho mọi action.
    - Chọn action theo $\epsilon$-greedy (exploration).
    - Lưu transition vào replay buffer, train bằng mini-batch.

**Thiết kế cụ thể cho project**:

- **State input**:
    - Nếu single-stock: vector kích thước $L \times d$ flattened thành 1 chiều hoặc đưa qua 1D-CNN / LSTM đơn giản.
    - Để đơn giản, bạn có thể flatten: dimension ≈ $L \times$ (số feature).
- **Network architecture** (MLP đơn giản, giống Chen 2019):[^1]
    - Dense(128) – ReLU
    - Dense(64) – ReLU
    - Dense(|A|) – linear (số node = số action).
- **Tricks ổn định** (theo Zhang 2019):[^2]
    - Fixed target network.
    - Double DQN.
    - Dueling architecture (tùy, nếu muốn advanced).
- **Reward**:
    - Dùng return theo ngày đã trừ phí.
- **Ưu điểm**:
    - Dễ implement (stable-baselines3 có sẵn DQN).
    - Phù hợp bài toán discrete, single-stock.

**Hạn chế**:

- Không tự nhiên cho action liên tục/tỉ lệ phân bổ danh mục (portfolio multi-stock).
- DQN tương đối nhạy tham số, training không ổn định nếu reward noisy (thị trường Việt Nam nhiễu cao).[^2]


### 3.2. Thuật toán 2 – PPO + LSTM (đơn giản hóa CLSTM-PPO)

Theo Zou et al. (2023): dùng LSTM để trích xuất đặc trưng chuỗi thời gian, sau đó dùng PPO để học chính sách trading đa tài sản.[^3]

**Ý tưởng chính**:

- PPO là **actor–critic**:
    - Actor $\pi_\theta(a \mid s)$: policy sinh action.
    - Critic $V_\phi(s)$: ước lượng giá trị state.
- Dùng **objective clipped surrogate** để update an toàn, tránh update quá mạnh làm hỏng policy.[^3]
- LSTM làm feature extractor:
    - Input: chuỗi $L$ state liên tiếp.
    - Output: feature vector cho PPO.

**Thiết kế cho Việt Nam (multi-stock VN30)**:

- **State**:
    - 181-dim state như Yang et al. \& Zou et al. (bạn không nhất thiết phải lên 181 dim, nhưng structure tương tự).[^3]
    - Thành phần:
        - Tiền mặt normalized.
        - Giá adjusted close 30 mã.
        - Số cổ phiếu nắm giữ 30 mã.
        - Indicators: MACD, RSI, CCI, ADX cho 30 mã.[^3]
- **Action space**:
    - Continuous $[-1, 1]^{N}$ với N = số mã:
        - $a_i > 0$: tăng vị thế mã i.
        - $a_i < 0$: giảm vị thế mã i.
    - Trong code, có thể map action sang số cổ phiếu mua/bán, giới hạn bởi tiền mặt và không short (clip hành động sao cho holdings ≥ 0).
- **Reward**:
    - Thay đổi giá trị portfolio (return) như trên.
    - Trừ phí giao dịch (0.1% mỗi lần buy/sell * giá trị giao dịch).
- **LSTM feature extractor**:
    - LSTM (hidden size 128) đọc chuỗi $L$ state gần nhất.
    - Lấy output cuối cùng (hoặc toàn chuỗi, tùy policy của SB3) làm feature đưa vào policy/value head.[^3]
- **Risk control – Turbulence index**:
    - Zou (2023) dùng **turbulence index** dựa trên covariance của returns để phát hiện market crash; nếu vượt ngưỡng 90th percentile thì agent dừng giao dịch.[^3]
    - Bạn có thể áp dụng tương tự cho VN30 (giai đoạn COVID-19, 2022/2023 crash…).

**Ưu điểm**:

- Hợp với **multi-stock VN30 portfolio**.
- PPO thường **ổn định và dễ tune** hơn các policy gradient khác, là standard trong nhiều bài báo và framework.[^2][^3]
- LSTM tận dụng **chuỗi thời gian Việt Nam vốn có xu hướng + regime** (sóng tăng/giảm kéo dài, sideway, v.v.).[^2][^3]

**Hạn chế**:

- Phức tạp hơn DQN; dễ overfit nếu data ít (VN dữ liệu ngắn hơn US).
- Cần nhiều tuning (learning rate, clip range, length episode, reward scaling).


### 3.3. So sánh kỳ vọng giữa DQN và PPO

- **DQN**:
    - Đơn giản, hợp bài toán 1–2 mã, discrete action.
    - Hay được thấy là “tốt nhất” trong một số nghiên cứu futures multi-asset khi action space vẫn discrete.[^2]
- **PPO**:
    - Tự nhiên hơn cho **phân bổ danh mục**; xử lý tốt action liên tục.
    - Trong CLSTM-PPO, PPO + LSTM outperform cả ensemble DRL (PPO+A2C+DDPG) trong nhiều thị trường, và **đặc biệt tốt ở thị trường mới nổi (Trung Quốc)** – khá tương đồng với Việt Nam.[^3]
- Bạn có thể đặt **giả thuyết** cho báo cáo:
> Với dữ liệu VN30 daily, PPO + LSTM sẽ perform tốt hơn DQN multi-asset cả về cumulative return và Sharpe, nhất là trong giai đoạn thị trường có xu hướng mạnh, trong khi DQN dễ gặp khó ở action space lớn và reward noisy.

***

## 4. Mô tả môi trường RL (environment)

Xây dựng theo chuẩn Gym để plug vào stable-baselines3.

### 4.1. Thiết kế tổng quát

**Lớp `VNTradingEnv(gym.Env)`** với các thành phần chính:

- `observation_space`: Box(low, high, shape=(state_dim,))
    - Có thể là vector flatten của:
        - L ngày feature cho từng mã (hoặc chỉ LSTM làm feature extractor).
        - cash, holdings, indicators.
- `action_space`:
    - DQN: `Discrete(n_actions)` với n_actions = 3 (sell, hold, buy) hoặc 2 (hold, buy).
    - PPO: `Box(low=-1, high=1, shape=(N_stocks,))`.
- `reset()`:
    - Đặt ngày index về `start_idx` của tập train.
    - Đặt cash = initial_capital (vd 1e6), holdings = 0.
    - Tính state đầu tiên (dựa trên L ngày trước đó).
- `step(action)`:

1. Từ action, tính số cổ phiếu mua/bán từng mã (giới hạn bởi cash, không cho holdings < 0 nếu không short).
2. Tính **transaction cost** = fee_rate × traded_value.
3. Cập nhật `cash`, `holdings`.
4. Tiến đến ngày tiếp theo, cập nhật giá.
5. Tính `portfolio_value_t+1`.
6. Reward = $(V_{t+1} - V_t)/V_t$ (hoặc số tuyệt đối).
7. Kiểm tra điều kiện `done`:
        - Hết data (đến cuối test/training window).
        - Optional: phá sản (portfolio < threshold).
8. Trả về `(next_state, reward, done, info)`.


### 4.2. Chú ý “phù hợp Việt Nam”

- **Không short selling** phổ thông:
    - Mặc định, **cấm holdings < 0**.
- **Không margin/leverage**:
    - Không cho tổng giá trị mua > cash hiện tại.
- **Phí giao dịch**:
    - Việt Nam có phí môi giới + thuế; chọn 1 tham số fee_rate (vd 0.001–0.002) để mô phỏng.
- **Time granularity**:
    - Dùng **daily**, phù hợp khung trading thực tế của cá nhân (T+2).
- **Turbulence \& sự kiện cực đoan**:
    - Giai đoạn 2020–2022 ở VN có nhiều cú sốc; có thể dùng turbulence threshold để agent “nghỉ giao dịch” khi biến động quá mạnh, giống Zou (2023).[^3]

***

## 5. Hướng triển khai thuật toán – chi tiết các bước

### 5.1. Stack công nghệ

- **Python**: 3.9+.
- **Thư viện chính**:
    - `pandas`, `numpy` – xử lý dữ liệu.
    - `ta` hoặc tự code MACD, RSI, CCI, ADX.[^2][^3]
    - `gymnasium` hoặc `gym` – define environment.
    - `stable-baselines3` – DQN, PPO.
    - `matplotlib` / `seaborn` – vẽ equity curve, histogram, v.v.


### 5.2. Quy trình từng bước

1. **Chuẩn bị dữ liệu Việt Nam**
    - Tải dataset VN30 từ Kaggle hoặc crawl từ Stockbiz/HOSE.[^5][^4]
    - Làm sạch:
        - Sort theo `date`.
        - Forward-fill missing nếu cần.
        - Chọn các mã có đủ lịch sử.
    - Tính indicator (MACD, RSI, CCI, ADX) cho từng mã.
    - Tạo cấu trúc dữ liệu thân thiện (ví dụ: 3D array: time × stock × features).
2. **Xây dựng hàm tạo state**
    - Input: index ngày $t$, window size $L$.
    - Output: vector state:
        - Stack feature của $t-L+1 \to t$ cho mọi mã.
        - Thêm cash / holdings normalized.
3. **Xây dựng môi trường `VNTradingEnv`**
    - Cài đặt `__init__`, `reset`, `step` như mô tả ở mục 4.
    - Cho phép chọn mode:
        - `mode="single"`: chỉ 1 mã; DQN.
        - `mode="multi"`: multi-stock; PPO.
4. **Huấn luyện DQN**
    - Dùng stable-baselines3:

```python
model = DQN(
    "MlpPolicy",
    env_single_stock,
    learning_rate=1e-3,
    buffer_size=100_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1_000,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
)
model.learn(total_timesteps=200_000)
```

    - Có thể thử:
        - Double DQN, dueling network (trong SB3 có `DQNPolicy` options).
        - Giảm gamma nếu bạn muốn agent quan tâm nhiều hơn reward ngắn hạn (như Chen, discount factor 0.79 cho daily trading).[^1]
5. **Huấn luyện PPO + LSTM**
    - Dùng `RecurrentPPO` hoặc custom feature extractor:
        - Trong SB3, bạn có thể define `policy_kwargs={"features_extractor_class": CustomLSTMExtractor, ...}` tương tự cách Zou dùng LSTM làm feature extractor.[^3]
    - Hyperparams (gợi ý từ CLSTM-PPO):[^3]
        - `learning_rate ≈ 3e-4`
        - `gamma = 0.99`
        - `n_steps ≈ 128`
        - `clip_range = 0.2`
        - `gae_lambda = 0.95`
        - `ent_coef ≈ 0.01`
        - `vf_coef ≈ 0.5`
    - Train nhiều epoch (vì multi-stock, state phức tạp):
        - `total_timesteps` vài trăm nghìn đến vài triệu, tùy resource.
6. **Đánh giá \& so sánh**
    - Chạy **backtest** trên tập test cho mỗi model (freeze weights, không update).
    - Ghi lại:
        - Equity curve (giá trị portfolio theo thời gian).
        - Số trade, win rate, average profit per trade (APPT).[^2][^3]
        - **Metrics**:
            - Cumulative return (CR).
            - Max earning rate (MER).
            - Max drawdown (MDD/MPB).
            - Sharpe ratio, Sortino (nếu muốn).
    - So sánh với:
        - Buy \& Hold VN30 (hoặc 1 mã benchmark).
        - Có thể so sánh với chiến lược đơn giản MACD/time-series momentum như Zhang et al.[^2]
7. **Phân tích kết quả – liên hệ với bối cảnh Việt Nam**
    - Phân tích xem:
        - DQN perform thế nào trên 1 mã bluechip (VD: VNM, VIC, HPG).
        - PPO multi-asset có tận dụng được tính **luân chuyển dòng tiền** giữa các mã VN30 không (giai đoạn 2020–2022, v.v.).
    - Liên hệ với đặc thù Việt Nam:
        - Thị trường mới nổi, biến động mạnh, news-driven: RL có tận dụng được trend không?
        - Giai đoạn crash (COVID, 2022) agent có giảm drawdown tốt hơn Buy \& Hold không?
    - So sánh định tính:
        - DQN có học ra policy overfit 1 mã?
        - PPO có xu hướng “đa dạng hóa” danh mục tốt hơn?
8. **Viết báo cáo cho môn học**
    - **Chương 1 – Giới thiệu \& mô tả bài toán**:
        - Nêu bối cảnh VN, vì sao RL trading thú vị.
        - Trình bày MDP, mục tiêu.
    - **Chương 2 – Dataset Việt Nam**:
        - Mô tả nguồn dữ liệu (VN30/Kaggle/Stockbiz), thời gian, số mã, các field.
        - Quy trình xử lý \& tính indicator.
    - **Chương 3 – Thuật toán \& thiết kế**:
        - Lý thuyết DQN, PPO (tóm tắt).
        - Mô tả state, action, reward cụ thể của bạn.
        - Giải thích vì sao chọn 2 thuật toán này (liên hệ với các paper: Chen/DQN, Zhang/DQN vs A2C, Zou/PPO+LSTM).[^1][^2][^3]
    - **Chương 4 – Môi trường \& triển khai**:
        - Mô tả environment (sơ đồ flow: data → env → agent → reward).
        - Trình bày các giả định phù hợp VN (no short, no margin, fee).
    - **Chương 5 – Kết quả \& thảo luận**:
        - Bảng so sánh các metric (CR, MDD, Sharpe…).
        - Biểu đồ equity curve.
        - Bình luận ưu/nhược điểm DQN vs PPO trên dữ liệu Việt Nam.
    - **Chương 6 – Kết luận \& hướng phát triển**:
        - Thêm sentiment (tin tức VN), fundamental indicators (Kaggle fundamental VN dataset).[^8][^9]
        - Thử các RL khác (SAC, A2C) hoặc MARL.

***
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19]</span>

<div align="center">⁂</div>

[^1]: chen2019.pdf

[^2]: 1911.10107v1.pdf

[^3]: 2212.02721v2.pdf

[^4]: https://www.kaggle.com/datasets/thangtranquang/stock-vn30-vietnam

[^5]: http://en.stockbiz.vn/HistoricalIndices.aspx

[^6]: https://www.kaggle.com/competitions/stock-market-prediction/data

[^7]: https://tradingeconomics.com/vnindex:ind

[^8]: ferreira2021.pdf

[^9]: https://www.kaggle.com/datasets/thaonguyentrang/fundamental-indicators-in-vietnam-stock-market

[^10]: https://www.scribd.com/document/738374511/Nhom18

[^11]: https://www.facebook.com/cung.AI.VN/posts/cẩn-thận-khi-mua-đồ-nvidiamột-số-ae-tech-đang-bức-xúc-vì-hàng-nvidia-đu/548946814807743/

[^12]: https://cole.vn/san-pham/khoa-hoc-ai-engineer-942

[^13]: https://vfcs.org.vn/wp-content/uploads/2025/09/BCKT-Xay-dung-TCVN-cacbon-rung_Final-approved.pdf

[^14]: https://hvnh.edu.vn/medias/tapchi/vi/04.2024/system/archivedate/1da61e7a_S%E1%BB%91%20263-%20Th%C3%A1ng%204.2024.pdf

[^15]: https://arxiv.org/html/2411.07585v1

[^16]: https://dost.hochiminhcity.gov.vn/documents/1422/3._Ky_yeu_Kinh_te.pdf

[^17]: https://tradingeconomics.com/vietnam/stock-market

[^18]: https://www.ijnrd.org/papers/IJNRD2508023.pdf

[^19]: https://tapchigiaothong.qltns.mediacdn.vn/481400261263945728/2025/7/20/tcxd-07-2025-17529768977321356171246.pdf

