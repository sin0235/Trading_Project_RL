<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# mô tả 2 thuật toán sẽ dùng

Dưới đây là mô tả chi tiết, đúng “style báo cáo môn học”, cho **2 thuật toán** rất hợp với project của bạn:

- **Thuật toán 1: DQN (Deep Q-Network)** – đại diện cho nhánh **value-based, critic-only**.[^1][^2]
- **Thuật toán 2: PPO (Proximal Policy Optimization)** kết hợp LSTM (theo hướng CLSTM-PPO) – đại diện cho nhánh **policy gradient, actor–critic**.[^2][^3]

Bạn có thể dùng chính hai phần này làm mục “Phương pháp” trong báo cáo.

***

## 1. Thuật toán 1 – DQN (Deep Q-Network)

### 1.1. Ý tưởng chung

DQN là thuật toán **học hàm giá trị hành động** $Q(s, a)$ bằng mạng nơ-ron sâu. Ý tưởng:

- Mô hình hóa trading như MDP:
    - State $s_t$: thông tin thị trường + danh mục tại ngày $t$.
    - Action $a_t$: hành động giao dịch (mua / giữ / bán).
    - Reward $r_t$: lãi/lỗ danh mục trong bước thời gian đó (sau phí).
- Mục tiêu: tìm hàm $Q_\theta(s,a)$ xấp xỉ **giá trị kỳ vọng** của tổng phần thưởng tương lai khi ta:
    - chọn action $a$ tại state $s$,
    - sau đó hành động theo chính sách greedy từ $Q$.

Từ $Q_\theta(s,a)$, chính sách được lấy đơn giản:

$$
\pi(s) = \arg\max_a Q_\theta(s,a)
$$

Trong trading: ở mỗi ngày, agent chọn hành động có $Q$ cao nhất (với $\epsilon$-greedy để khám phá).

### 1.2. Cập nhật Q bằng Bellman target

DQN bắt nguồn từ Q-learning cổ điển:

- Phương trình Bellman tối ưu cho $Q^\*(s,a)$:

$$
Q^\*(s,a) = \mathbb{E}\big[r + \gamma \max_{a'} Q^\*(s', a') \mid s, a\big]
$$

- Ta dùng mạng nơ-ron $Q_\theta(s,a)$ để xấp xỉ.
Target tại bước $t$:

$$
y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')
$$

Trong đó:

- $Q_{\theta^-}$ là **target network**, copy từ $Q_\theta$ sau mỗi vài ngàn bước để ổn định huấn luyện.[^1][^2]
- Loss:

$$
L(\theta) = \mathbb{E}[(y_t - Q_\theta(s_t, a_t))^2]
$$

Tối ưu $\theta$ bằng SGD/Adam để giảm sai số này.

### 1.3. Các kỹ thuật ổn định

DQN gốc rất dễ “nổ” khi dữ liệu nhiễu (thị trường tài chính). Trong các paper về trading, người ta thường dùng:

1. **Replay Buffer (Experience Replay)**:
    - Lưu các transition $(s_t, a_t, r_t, s_{t+1})$ vào buffer.
    - Train bằng cách **lấy minibatch ngẫu nhiên** từ buffer:
        - Giảm tương quan giữa mẫu liên tiếp.
        - Tăng hiệu quả sử dụng dữ liệu.
2. **Target Network**:
    - Dùng mạng tách biệt $Q_{\theta^-}$ để tính target $y_t$.
    - Cứ mỗi $C$ bước cập nhật $\theta^- \leftarrow \theta$.[^2][^1]
3. (Tuỳ chọn nâng cao) **Double DQN, Dueling DQN**:
    - Double DQN: giảm overestimation bias của $\max Q$.
    - Dueling: tách $Q(s,a)$ thành value $V(s)$ + advantage $A(s,a)$.
Cả hai đều từng được dùng trong trading futures/cổ phiếu.[^2]

### 1.4. Cách áp dụng DQN cho project trading Việt Nam

**Action space** (discrete, rất hợp với DQN):

- Với 1 mã cổ phiếu:
    - $A = \{-1, 0, +1\}$: Bán – Giữ – Mua (1 đơn vị / 1 lô).
    - Hoặc đơn giản: $A = \{0, 1, 2\}$ mapping sang SELL/HOLD/BUY.[^1][^2]
- Mỗi bước (mỗi ngày):
    - Agent dùng state $s_t$ (30 ngày gần nhất + portfolio).
    - Mạng DQN output 3 giá trị $Q(s_t, a)$ tương ứng 3 hành động.
    - Áp dụng $\epsilon$-greedy:
        - Với xác suất $\epsilon$: chọn random (explore).
        - Với xác suất $1 - \epsilon$: chọn action có $Q$ lớn nhất (exploit).

**State**:

- Bạn đã thiết kế: vector gồm:
    - Thông tin giá \& indicator cho 30 ngày × N mã.
    - Tỉ lệ tiền mặt \& tỉ lệ phân bổ từng mã (portfolio info).

**Reward**:

- Có thể dùng:

$$
r_t = \frac{V_{t+1} - V_t}{V_t}
$$

với $V_t$ là giá trị portfolio sau phí giao dịch.

**Ưu điểm DQN trong project**:

- Rõ ràng, dễ giải thích trong báo cáo (giống Chen 2019 dùng DQN/DRQN cho SPY ETF).[^1]
- Thích hợp để **so sánh “baseline RL”** với thuật toán mạnh hơn (PPO).
- Implementation có sẵn trong stable-baselines3.

**Hạn chế**:

- Action space phải **rời rạc** → không tự nhiên cho multi-stock portfolio trading với phân bổ vốn liên tục.[^2]
- Khá nhạy với hyperparameter và replay buffer, dễ overfit nếu data Việt Nam ngắn/nhiễu.

***

## 2. Thuật toán 2 – PPO (Proximal Policy Optimization, dạng actor–critic với LSTM)

### 2.1. Ý tưởng chung

PPO là thuật toán **policy gradient hiện đại**, thuộc nhóm **actor–critic**, rất phổ biến trong RL thực nghiệm (cả game lẫn tài chính).[^3][^2]

- **Actor** $\pi_\theta(a \mid s)$:
    - Policy sinh ra action trực tiếp;
    - Với action liên tục, thường là phân phối Gaussian (mean \& std do mạng sinh ra).
- **Critic** $V_\phi(s)$:
    - Ước lượng **value** của state $s$, hỗ trợ giảm phương sai cho gradient.

Khác với DQN:

- DQN học $Q(s,a)$ rồi mới suy ra policy.
- PPO **học chính sách trực tiếp** (policy-based), rất phù hợp với action **liên tục** như phân bổ vốn theo nhiều mã cổ phiếu.


### 2.2. Policy gradient và vấn đề “update quá mạnh”

Policy gradient cơ bản tối ưu:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\big[ \log \pi_\theta(a_t \mid s_t) \, A_t \big]
$$

Trong đó:

- $A_t$ là **advantage** (hành động tốt hơn hay tệ hơn kỳ vọng của state).
- Gradient update $\theta$ theo $\nabla_\theta J(\theta)$.

Vấn đề: nếu mỗi update thay đổi $\pi_\theta$ **quá mạnh**, có thể làm performance bị phá hủy (policy “nhảy” lung tung).

### 2.3. Ý tưởng “Proximal” của PPO

PPO giải vấn đề này bằng cách **giới hạn mức thay đổi policy** thông qua hàm mục tiêu “clipped”:

- Đầu tiên định nghĩa **tỉ lệ xác suất**:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

- Hàm mục tiêu clipped surrogate:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\big[ \min(
r_t(\theta) A_t,\;
\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t
) \big]
$$

Trong đó:

- $\epsilon$ thường ~ 0.1–0.2.[^3]
- Nếu $r_t(\theta)$ vượt ngoài $[1-\epsilon, 1+\epsilon]$, gradient sẽ bị **clamp**, tránh update quá mạnh.

Ngoài ra PPO còn có:

- Term **entropy bonus** để khuyến khích khám phá.
- Loss Critic $L_V(\phi)$ để train $V_\phi(s)$ bằng MSE giữa return thực tế và dự đoán.


### 2.4. Kết hợp LSTM – CLSTM-PPO cho chuỗi giá

Trong paper “A Novel Deep RL Based Automated Stock Trading System Using Cascaded LSTM – CLSTM-PPO” của Zou (2023):[^3]

- Dùng **LSTM** như **feature extractor** để xử lý chuỗi thời gian:
    - Input: chuỗi $T$ state gần nhất $(s_{t-T+1}, \dots, s_t)$.
    - LSTM học các pattern thời gian (trend, volatiltiy regime…).
    - Output: vector feature $f_t$ dùng làm input cho actor và critic.
- Sau đó dùng PPO để học:
    - Actor: policy trên không gian action (phân bổ vốn multi-stock).
    - Critic: value function trên feature $f_t$.

Với dữ liệu thị trường mới nổi (Trung Quốc), CLSTM-PPO cho kết quả tốt hơn cả Buy \& Hold và ensemble DRL (PPO+A2C+DDPG). Điều này rất gần bối cảnh Việt Nam.[^3]

### 2.5. Cách áp dụng PPO cho project trading Việt Nam

**Action space** (liên tục – điểm mạnh của PPO):

- Giả sử bạn trade $N$ mã (VD: 5 mã VN30).
- Define action là vector:

$$
a_t \in [-1, 1]^N
$$

Trong đó:

- $a_{t,i} > 0$: tăng vị thế (mua mã i).
- $a_{t,i} < 0$: giảm vị thế (bán mã i).
- Ở implementation:
    - Map $a_{t,i}$ → số cổ phiếu mua/bán giới hạn max\_trade\_amount.
    - Kiểm soát **không short**: không cho holdings < 0, không margin.

**State**:

- Dùng đúng state bạn đã thiết kế:
    - Thông tin giá + indicators cho 30 ngày × N mã.
    - Thông tin danh mục (cash_ratio, holdings_ratio).
- Phần LSTM:
    - Có thể:
        - Hoặc tự thêm LSTM trong policy.
        - Hoặc dùng custom features extractor (LSTM) giống CLSTM-PPO.[^3]

**Reward**:

- Tương tự DQN: return theo ngày (sau phí).
- Có thể thử thêm risk-adjusted reward sau nếu còn thời gian (Sharpe-like, penalty drawdown).

**Pipeline PPO trong project**:

1. Mỗi episode: duyệt qua 1 giai đoạn lịch sử (VD: 2015–2019).
2. Mỗi ngày:
    - Lấy state $s_t$ (nếu có LSTM: dùng chuỗi state).
    - Actor sinh action $a_t$.
    - Env thực hiện trade, tính reward $r_t$ và next state.
3. Lưu trajectories $(s_t, a_t, r_t, s_{t+1})$ theo batch.
4. Sau mỗi **N steps**, train PPO:
    - Tính advantage $A_t$ bằng GAE (Generalized Advantage Estimation).
    - Update $\theta, \phi$ bằng loss clipped objective + value loss + entropy loss.

**Ưu điểm PPO trong project**:

- Hợp với **multi-stock portfolio** (VN30 subset).
- Action liên tục → agent có thể học chiến lược **phân bổ vốn** chứ không chỉ mua/bán 1 mã đơn lẻ.[^2][^3]
- PPO tương đối **ổn định, ít “kén” hyperparameter** hơn các policy gradient khác, được dùng rất nhiều trong RL thực nghiệm.

**Hạn chế**:

- Mô hình phức tạp hơn DQN:
    - Nhiều tham số hơn (actor + critic + LSTM).
    - Dễ overfit nếu dữ liệu quá ít (VN data ngắn hơn US).
- Training tốn thời gian hơn (multi-asset, continuous control).

***

## 3. Lý do chọn và cách so sánh 2 thuật toán trong báo cáo

Khi viết **phần lựa chọn thuật toán \& so sánh**, bạn có thể nhấn mạnh:

1. **Tính đại diện**:
    - DQN: tiêu biểu cho **value-based, critic-only**, action discrete; đã được ứng dụng thành công cho trading SPY, futures…[^1][^2]
    - PPO: tiêu biểu cho **policy gradient, actor–critic** hiện đại, action continuous; CLSTM-PPO chứng minh hiệu quả trên nhiều thị trường (US, China, India, UK).[^3]
2. **Gắn với bài toán Việt Nam**:
    - Thị trường VN là **emerging market**, biến động mạnh, dữ liệu nhiễu:
        - DQN là baseline tốt cho **single-stock trading**.
        - PPO + LSTM phù hợp hơn để **bắt pattern thời gian** và **phân bổ danh mục** giữa nhiều mã VN30.[^2][^3]
3. **Kỳ vọng kết quả**:
    - DQN có thể:
        - Học được chiến lược trên 1–2 mã lớn (e.g., VNM, VIC) nhưng dễ bị ảnh hưởng noise.
    - PPO có khả năng:
        - Tận dụng diversification (nhiều mã),
        - Kiểm soát risk tốt hơn,
        - Cho Sharpe ratio \& cumulative return cao hơn trong backtest.
4. **Tiêu chí so sánh**:
    - Về **hiệu suất**:
        - Cumulative return.
        - Sharpe ratio.
        - Max drawdown.
        - Win rate, average profit per trade.[^2][^3]
    - Về **thực thi**:
        - Thời gian train.
        - Độ ổn định (variance kết quả qua nhiều seed).
    - Về **ý nghĩa thực tế**:
        - Có tôn trọng ràng buộc thị trường VN (no short, no margin, fees)?
        - Chiến lược có “hợp lý” khi quan sát chuỗi lệnh \& equity curve?

***


<div align="center">⁂</div>

[^1]: chen2019.pdf

[^2]: 1911.10107v1.pdf

[^3]: 2212.02721v2.pdf

