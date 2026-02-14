<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Action Space - Thiết kế và Phân tích Học thuật


***

## PHẦN 1: Nền tảng lý thuyết

### 1.1. Định nghĩa Action Space

Trong Reinforcement Learning, Action Space (không gian hành động) là tập hợp tất cả các hành động khả thi mà agent có thể thực hiện tại một state bất kỳ (Sutton \& Barto, 2018).

**Định nghĩa toán học:**

Cho một Markov Decision Process (MDP) được định nghĩa bởi tuple $(S, A, P, R, \gamma)$, trong đó:

- $S$: State space
- $A$: Action space
- $P$: Transition probability
- $R$: Reward function
- $\gamma$: Discount factor

Action space $A$ có thể được phân loại thành hai loại chính:

1. **Discrete Action Space**: $A = \{a_1, a_2, ..., a_n\}$ với $n$ hữu hạn
2. **Continuous Action Space**: $A \subseteq \mathbb{R}^d$ với $d$ là số chiều

### 1.2. Action Space trong Portfolio Trading

Trong bài toán portfolio trading với $N$ tài sản, action tại thời điểm $t$ biểu diễn quyết định giao dịch cho từng tài sản.

**Representation:**

$$
a_t = [a_t^1, a_t^2, ..., a_t^N]
$$

Với $a_t^i$ là quyết định giao dịch cho tài sản $i$.

***

## PHẦN 2: Discrete Action Space (DQN)

### 2.1. Thiết kế

Cho portfolio với $N$ tài sản, mỗi tài sản có $K$ hành động rời rạc.

**Định nghĩa:**

$$
A_{\text{discrete}} = \{0, 1, 2, ..., K \times N - 1\}
$$

Trong đề tài với $N=5$ mã cổ phiếu và $K=3$ loại hành động:

$$
|A_{\text{discrete}}| = K \times N = 3 \times 5 = 15
$$

**Mapping function:**

Định nghĩa hàm ánh xạ $f: A_{\text{discrete}} \to \{1, ..., N\} \times \{SELL, HOLD, BUY\}$

$$
f(a) = (i, t)
$$

Trong đó:

$$
\begin{aligned}
i &= \lfloor a / K \rfloor + 1 \quad \text{(stock index)} \\
t &= a \mod K \quad \text{(action type)}
\end{aligned}
$$

### 2.2. Trade Execution Function

Định nghĩa hàm thực thi giao dịch $g: A_{\text{discrete}} \to \mathbb{Z}^N$:

$$
g(a) = \mathbf{q} = [q_1, q_2, ..., q_N]
$$

Với:

$$
q_j = 
\begin{cases}
-Q_{\max} & \text{if } j = i \text{ and } t = 0 \text{ (SELL)} \\
0 & \text{if } j \neq i \text{ or } t = 1 \text{ (HOLD)} \\
+Q_{\max} & \text{if } j = i \text{ and } t = 2 \text{ (BUY)}
\end{cases}
$$

Trong đó $Q_{\max}$ là số lượng cổ phiếu tối đa cho mỗi giao dịch.

### 2.3. Mã giả - Discrete Action Decoding

```
Algorithm 1: Discrete Action Decoding
────────────────────────────────────────────────────────────
Input: action_index a ∈ {0, 1, ..., KN-1}
       n_stocks N
       max_shares Q_max
       action_types K = 3
Output: trade_amounts q ∈ Z^N

1:  Initialize q ← [0, 0, ..., 0] ∈ Z^N
2:  Compute stock_idx i ← ⌊a / K⌋
3:  Compute action_type t ← a mod K
4:  
5:  if t = 0 then                    // SELL
6:      q[i] ← -Q_max
7:  else if t = 1 then               // HOLD
8:      q[i] ← 0
9:  else if t = 2 then               // BUY
10:     q[i] ← +Q_max
11: end if
12: 
13: return q
```


### 2.4. Phân tích ưu nhược điểm

**Ưu điểm:**

1. **Computational efficiency**: Action space có kích thước hữu hạn, thuận lợi cho Q-learning
2. **Convergence properties**: DQN với discrete actions đã được chứng minh hội tụ (Mnih et al., 2015)
3. **Interpretability**: Mỗi action có nghĩa rõ ràng
4. **Exploration strategy**: Epsilon-greedy exploration đơn giản và hiệu quả

**Nhược điểm:**

1. **Sequential limitation**: Mỗi bước chỉ trade một tài sản
2. **Fixed position sizing**: Không thể điều chỉnh linh hoạt kích thước vị thế
3. **Scalability issues**: Với $N$ lớn, action space tăng tuyến tính
4. **Lack of realism**: Không phản ánh chính xác hành vi trading thực tế

***

## PHẦN 3: Continuous Action Space (PPO)

### 3.1. Thiết kế

**Định nghĩa:**

$$
A_{\text{continuous}} = [-1, 1]^N \subset \mathbb{R}^N
$$

Mỗi action là một vector $N$ chiều:

$$
\mathbf{a} = [a_1, a_2, ..., a_N]^T, \quad a_i \in [-1, 1]
$$

**Semantic interpretation:**

- $a_i = -1$: Bán tối đa tài sản $i$
- $a_i = 0$: Không giao dịch tài sản $i$
- $a_i = +1$: Mua tối đa tài sản $i$
- $a_i \in (-1, 0)$: Bán với tỷ lệ $|a_i|$
- $a_i \in (0, 1)$: Mua với tỷ lệ $a_i$


### 3.2. Scaling Function

Định nghĩa hàm scale $h: A_{\text{continuous}} \to \mathbb{Z}^N$:

$$
h(\mathbf{a}) = \mathbf{q}, \quad q_i = \lfloor a_i \times Q_{\max} \rceil
$$

Trong đó $\lfloor \cdot \rceil$ là hàm làm tròn đến số nguyên gần nhất.

### 3.3. Policy Parameterization

Trong PPO, policy được parameterized dưới dạng Gaussian distribution:

$$
\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))
$$

Trong đó:

- $\mu_\theta(s) \in \mathbb{R}^N$: Mean vector
- $\Sigma_\theta(s) \in \mathbb{R}^{N \times N}$: Covariance matrix

**Diagonal covariance assumption:**

Để đơn giản hóa, thường giả định covariance matrix là diagonal:

$$
\Sigma_\theta(s) = \text{diag}(\sigma_1^2(s), \sigma_2^2(s), ..., \sigma_N^2(s))
$$

### 3.4. Mã giả - Continuous Action Decoding

```
Algorithm 2: Continuous Action Decoding
────────────────────────────────────────────────────────────
Input: action_vector a ∈ R^N, where a_i ∈ [-1, 1]
       max_shares Q_max
Output: trade_amounts q ∈ Z^N

1:  Initialize q ∈ Z^N
2:  
3:  for i = 1 to N do
4:      // Clip to valid range
5:      a_i ← clip(a_i, -1, 1)
6:      
7:      // Scale to trade amount
8:      q_i ← a_i × Q_max
9:      
10:     // Round to integer
11:     q_i ← round(q_i)
12: end for
13: 
14: return q
```


### 3.5. Phân tích ưu nhược điểm

**Ưu điểm:**

1. **Parallel execution**: Có thể trade nhiều tài sản đồng thời
2. **Variable position sizing**: Linh hoạt trong việc điều chỉnh kích thước vị thế
3. **Continuous optimization**: PPO được thiết kế tối ưu cho continuous spaces (Schulman et al., 2017)
4. **Realistic representation**: Phản ánh chính xác hành vi portfolio rebalancing

**Nhược điểm:**

1. **Implementation complexity**: Phức tạp hơn trong việc implement và debug
2. **Sample inefficiency**: Thường cần nhiều samples hơn để học
3. **Constraint handling**: Cần xử lý constraints phức tạp (cash, holdings)
4. **Hyperparameter sensitivity**: Nhạy cảm với learning rate, entropy coefficient

***

## PHẦN 4: Constraint Handling

### 4.1. Formal Constraint Definition

Cho portfolio state tại thời điểm $t$:

- $C_t$: Cash (tiền mặt)
- $H_t = [h_t^1, h_t^2, ..., h_t^N]$: Holdings (số lượng nắm giữ)
- $P_t = [p_t^1, p_t^2, ..., p_t^N]$: Prices (giá hiện tại)
- $f$: Transaction fee rate

**Constraints:**

**C1 (Selling constraint):**

$$
\forall i: q_i < 0 \implies |q_i| \leq h_t^i
$$

**C2 (Buying constraint):**

$$
\sum_{i: q_i > 0} q_i \cdot p_t^i \cdot (1 + f) \leq C_t
$$

**C3 (Non-negative holdings):**

$$
\forall i: h_t^i + q_i \geq 0
$$

### 4.2. Constraint Projection

Định nghĩa constraint projection function $\Pi: \mathbb{Z}^N \to \mathbb{Z}^N$:

$$
\mathbf{q}^* = \Pi(\mathbf{q}) = \arg\min_{\mathbf{q}' \in \mathcal{F}} \|\mathbf{q}' - \mathbf{q}\|_2^2
$$

Trong đó $\mathcal{F}$ là feasible set thỏa mãn các constraints C1, C2, C3.

### 4.3. Mã giả - Constraint Application

```
Algorithm 3: Constraint Application
────────────────────────────────────────────────────────────
Input: trade_amounts q ∈ Z^N
       cash C_t
       holdings H_t ∈ Z^N
       prices P_t ∈ R^N
       fee_rate f
Output: constrained_amounts q* ∈ Z^N

1:  Initialize q* ← q
2:  
3:  // Constraint C1: Cannot sell more than holdings
4:  for i = 1 to N do
5:      if q*_i < 0 then
6:          q*_i ← max(q*_i, -H_t[i])
7:      end if
8:  end for
9:  
10: // Constraint C2: Cannot buy more than cash allows
11: buy_indices ← {i : q*_i > 0}
12: if buy_indices ≠ ∅ then
13:     buy_values ← {q*_i × P_t[i] : i ∈ buy_indices}
14:     buy_fees ← {v × f : v ∈ buy_values}
15:     total_cost ← Σ(buy_values + buy_fees)
16:     
17:     if total_cost > C_t then
18:         λ ← C_t / total_cost
19:         for i ∈ buy_indices do
20:             q*_i ← ⌊λ × q*_i⌋
21:         end for
22:     end if
23: end if
24: 
25: return q*
```


### 4.4. Constraint Handling Analysis

**Theoretical properties:**

1. **Feasibility**: $\Pi(\mathbf{q})$ luôn thỏa mãn tất cả constraints
2. **Proximity**: $\Pi(\mathbf{q})$ là điểm feasible gần $\mathbf{q}$ nhất
3. **Idempotency**: $\Pi(\Pi(\mathbf{q})) = \Pi(\mathbf{q})$

**Computational complexity:**

- Constraint C1: $O(N)$
- Constraint C2: $O(N)$
- Total: $O(N)$ - linear trong số lượng tài sản

***

## PHẦN 5: So sánh định lượng

### 5.1. Bảng so sánh chi tiết

| Tiêu chí | DQN (Discrete) | PPO (Continuous) |
| :-- | :-- | :-- |
| **Mathematical form** | $a \in \{0,...,14\}$ | $\mathbf{a} \in [-1,1]^5$ |
| **Dimension** | 1 (scalar) | 5 (vector) |
| **Cardinality** | 15 (finite) | $\infty$ (uncountable) |
| **Parallelism** | Sequential | Parallel |
| **Position sizing** | $\{0, Q_{\max}\}$ | $[0, Q_{\max}]$ |
| **Policy class** | $\pi: S \to \Delta(A)$ | $\pi: S \to \mathcal{N}(\mu, \Sigma)$ |
| **Value function** | $Q: S \times A \to \mathbb{R}$ | $V:S \to \mathbb{R}$ |
| **Convergence** | Proven (Mnih 2015) | Proven (Schulman 2017) |
| **Sample efficiency** | Higher | Lower |
| **Expressiveness** | Lower | Higher |

### 5.2. Complexity Analysis

**Space Complexity:**

DQN:

$$
\text{Space}(A_{\text{DQN}}) = O(K \times N)
$$

PPO:

$$
\text{Space}(A_{\text{PPO}}) = O(N)
$$

**Time Complexity per action:**

DQN decoding: $O(1)$
PPO decoding: $O(N)$
Constraint application: $O(N)$

***

## PHẦN 6: Integration với Environment

### 6.1. Environment Step Function

```
Algorithm 4: Environment Step
────────────────────────────────────────────────────────────
Input: action a (discrete hoặc continuous)
       current_state s_t
       mode ∈ {discrete, continuous}
Output: next_state s_{t+1}
        reward r_t
        done ∈ {True, False}
        info (dictionary)

1:  // Decode action
2:  if mode = discrete then
3:      q ← decode_discrete_action(a)
4:  else if mode = continuous then
5:      q ← decode_continuous_action(a)
6:  end if
7:  
8:  // Apply constraints
9:  q* ← apply_constraints(q, C_t, H_t, P_t, f)
10: 
11: // Execute trades
12: total_fee ← 0
13: for i = 1 to N do
14:     if q*_i ≠ 0 then
15:         success, fee ← execute_trade(i, q*_i, P_t[i])
16:         total_fee ← total_fee + fee
17:     end if
18: end for
19: 
20: // Update time
21: t ← t + 1
22: 
23: // Get new state
24: s_{t+1} ← get_state(t, C_t, H_t)
25: 
26: // Calculate reward
27: V_{t+1} ← C_t + Σ(H_t[i] × P_t[i])
28: r_t ← (V_{t+1} - V_t) / V_t
29: 
30: // Check termination
31: done ← (t ≥ T_max)
32: 
33: // Construct info
34: info ← {
35:     portfolio_value: V_{t+1},
36:     trades: q*,
37:     fees: total_fee
38: }
39: 
40: return (s_{t+1}, r_t, done, info)
```


### 6.2. Trade Execution Function

```
Algorithm 5: Trade Execution
────────────────────────────────────────────────────────────
Input: stock_index i
       shares q_i ∈ Z (q_i > 0: buy, q_i < 0: sell)
       price p_i
Output: success ∈ {True, False}
        fee ∈ R+

1:  trade_value ← |q_i| × p_i
2:  fee ← trade_value × f
3:  
4:  if q_i > 0 then                    // BUY
5:      total_cost ← trade_value + fee
6:      if total_cost > C_t then
7:          return (False, 0)
8:      end if
9:      C_t ← C_t - total_cost
10:     H_t[i] ← H_t[i] + q_i
11: 
12: else if q_i < 0 then               // SELL
13:     if |q_i| > H_t[i] then
14:         return (False, 0)
15:     end if
16:     C_t ← C_t + (trade_value - fee)
17:     H_t[i] ← H_t[i] + q_i          // q_i is negative
18: end if
19: 
20: return (True, fee)
```


***

## PHẦN 7: Thiết kế cho đề tài

### 7.1. Configuration cho DQN

```
DQN Action Space Configuration:
──────────────────────────────────────
Type:               Discrete
Total actions:      15
Actions per stock:  3 (SELL, HOLD, BUY)
Max shares/trade:   100
Constraint mode:    Post-action
```

**Justification:**

- Discrete space phù hợp với Q-learning framework
- 15 actions vừa đủ để exploration hiệu quả
- Fixed position sizing đơn giản hóa learning


### 7.2. Configuration cho PPO

```
PPO Action Space Configuration:
──────────────────────────────────────
Type:               Continuous
Dimension:          5
Range:              [-1, 1] per dimension
Max shares/trade:   100
Policy output:      Gaussian N(μ, σ²)
Constraint mode:    Post-action projection
```

**Justification:**

- Continuous space cho phép fine-grained control
- Parallel trading realistic hơn
- PPO được thiết kế tối ưu cho continuous actions


### 7.3. Hyperparameter Table

| Parameter | Symbol | DQN | PPO | Unit |
| :-- | :-- | :-- | :-- | :-- |
| Max shares per trade | $Q_{\max}$ | 100 | 100 | shares |
| Transaction fee rate | $f$ | 0.15 | 0.15 | % |
| Exploration param (start) | $\epsilon_0$ | 1.0 | N/A | - |
| Exploration param (end) | $\epsilon_f$ | 0.01 | N/A | - |
| Policy std (start) | $\sigma_0$ | N/A | 0.5 | - |
| Policy std (end) | $\sigma_f$ | N/A | 0.1 | - |


***

## PHẦN 8: Formal Notation Summary

**State Space:**

$$
S = \mathbb{R}^{d_s}, \quad d_s = L \times N \times F + (1 + N)
$$

**Action Spaces:**

$$
\begin{aligned}
A_{\text{DQN}} &= \{0, 1, ..., K \times N - 1\} \\
A_{\text{PPO}} &= [-1, 1]^N \subset \mathbb{R}^N
\end{aligned}
$$

**Decoding Functions:**

$$
\begin{aligned}
g: A_{\text{DQN}} &\to \mathbb{Z}^N \\
h: A_{\text{PPO}} &\to \mathbb{Z}^N
\end{aligned}
$$

**Constraint Projection:**

$$
\Pi: \mathbb{Z}^N \to \mathcal{F} \subset \mathbb{Z}^N
$$

**Policy Functions:**

$$
\begin{aligned}
\pi_{\text{DQN}}: S &\to \Delta(A_{\text{DQN}}) \\
\pi_{\text{PPO}}: S &\to \mathcal{N}(\mu(S), \Sigma(S))
\end{aligned}
$$

Trong đó $\Delta(A)$ là probability simplex trên $A$.

***

## Tham khảo

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
3. Sutton, R. S., \& Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
4. Zou, Y., et al. (2023). CLSTM-PPO: A novel deep reinforcement learning approach for stock market trading.
5. Zhang, Z., et al. (2019). Deep reinforcement learning for trading. Journal of Financial Data Science.
