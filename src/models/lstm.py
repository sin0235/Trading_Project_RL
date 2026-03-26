"""
Mạng neural LSTM cho RL Trading Agent.

Chứa 3 class chính:
    - LSTMFeatureExtractor: Backbone LSTM dùng chung
    - DRQNNetwork: Dueling Q-Network (cho DQN Agent)
    - PPOLSTMActorCritic: Actor-Critic (cho PPO Agent)

Input từ StateSpace (mode='sequential'):
    - market_state:    (batch, seq_len, n_stocks * n_features)
    - portfolio_state: (batch, portfolio_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np


# ============================================================
#  Khởi tạo trực giao
# ============================================================


def orthogonal_init(module, gain=1.0):
    """
    Khởi tạo trực giao.

    - Linear: trọng số trực giao, bias zero
    - LSTM: trực giao cho recurrent weights (weight_hh),
            xavier cho input weights (weight_ih),
            forget gate bias = 1.0 để cải thiện gradient flow
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
                # Bias cổng quên = 1.0 (Jozefowicz et al., 2015)
                # Thứ tự bias LSTM: [input_gate, forget_gate, cell_gate, output_gate]
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def _init_sequential(sequential: nn.Sequential, output_gain: float = 1.0):
    """
    Khởi tạo tất cả lớp Linear trong nn.Sequential.
    Các lớp trước ReLU dùng gain=sqrt(2), lớp c uối dùng output_gain.
    """
    linear_layers = [m for m in sequential.modules() if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        if i < len(linear_layers) - 1:
            # Các lớp ẩn (trước ReLU)
            orthogonal_init(layer, gain=np.sqrt(2))
        else:
            # Lớp đầu ra
            orthogonal_init(layer, gain=output_gain)


def _set_module_training_mode(module: nn.Module, training: bool) -> nn.Module:
    """
    Tương thích với môi trường torch thiếu nn.Module.train().

    Một số runtime downstream vẫn có eval() nhưng không expose train().
    Hàm này tự lan mode train/eval xuống toàn bộ cây module.
    """
    module.training = bool(training)
    for child in module.children():
        train_fn = getattr(type(child), "train", None)
        if callable(train_fn):
            child.train(training)
        else:
            _set_module_training_mode(child, training)
    return module


# ============================================================
#  Trích đặc trưng LSTM (Backbone)
# ============================================================


class LSTMFeatureExtractor(nn.Module):
    """
    Backbone LSTM xử lý chuỗi thời gian thị trường.

    Input:
        market_state: (batch, seq_len, input_size)
            trong đó input_size = n_stocks * n_features (vd: 16*7 = 112)
        hidden: tuple (h_0, c_0), mỗi cái shape (num_layers, batch, hidden_size)

    Output:
        features: (batch, hidden_size) — vector đặc trưng từ bước cuối cùng
        new_hidden: tuple (h_n, c_n)
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer norm để ổn định output LSTM
        self.layer_norm = nn.LayerNorm(hidden_size)
        orthogonal_init(self.lstm)

    def forward(self, market_state: torch.Tensor,
                hidden: tuple = None) -> tuple:
        """
        Args:
            market_state: (batch, seq_len, input_size)
            hidden: (h_0, c_0) hoặc None (sẽ tự khởi tạo bằng zeros)

        Returns:
            features: (batch, hidden_size)
            new_hidden: (h_n, c_n)
        """
        batch_size = market_state.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size, market_state.device)

        lstm_out, new_hidden = self.lstm(market_state, hidden)
        features = lstm_out[:, -1, :]
        features = self.layer_norm(features)

        return features, new_hidden

    def init_hidden(self, batch_size: int,
                    device: torch.device = None) -> tuple:
        """Khởi tạo hidden state bằng zeros."""
        if device is None:
            device = next(self.parameters()).device

        h_0 = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size, device=device)
        return (h_0, c_0)


# ============================================================
#  Tách Q(s,a) = V(s) + A(s,a) - mean(A)
#  V(s): giá trị trạng thái (thị trường tốt hay xấu)
#  A(s,a): lợi thế của hành động a so với trung bình
# ============================================================


class DRQNNetwork(nn.Module):
    """
    Dueling Deep Recurrent Q-Network cho Discrete Action Space.

    Tách thành 2 nhánh thay vì một Q-head phẳng:
        - Value stream: V(s) — giá trị trạng thái, không phụ thuộc action
        - Advantage stream: A(s,a) — lợi thế tương đối của từng action
        - Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    Dueling giúp agent hiểu "trạng thái này tốt hay xấu" (V) độc lập với
    "hành động nào tốt nhất" (A). Đặc biệt hữu ích trong trading vì nhiều
    trạng thái có giá trị tương tự bất kể hành động cụ thể.

    Architecture:
        market_state → LSTM → lstm_features (128)
                                    ↓ concat portfolio (17)
                              combined (145)
                                    ↓
                              Shared FC (145 → 128)
                              ↓               ↓
                         Value Stream    Advantage Stream
                              ↓               ↓
                           V(s) (1)      A(s,a) (n_actions)
                              ↓               ↓
                         Q(s,a) = V + A - mean(A)
    """

    def __init__(self, n_stocks: int, n_features: int,
                 seq_len: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 k: int = 3):
        """
        Args:
            n_stocks: Số lượng cổ phiếu (vd: 16)
            n_features: Số features mỗi cổ phiếu (vd: 7)
            seq_len: Độ dài chuỗi thời gian / window_size (vd: 30)
            hidden_size: Kích thước hidden state LSTM
            num_layers: Số lớp LSTM
            dropout: Dropout rate
            k: Số loại hành động mỗi cổ phiếu (3 = sell/hold/buy)
        """
        super().__init__()

        self.n_stocks = n_stocks
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.k = k
        self.n_actions = k * n_stocks

        input_size = n_stocks * n_features  # 16 * 7 = 112
        portfolio_dim = 1 + n_stocks        # 1 + 16 = 17
        combined_dim = hidden_size + portfolio_dim  # 128 + 17 = 145

        self.feature_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Nhánh giá trị: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Nhánh lợi thế: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.n_actions),
        )

        _init_sequential(self.shared_fc, output_gain=np.sqrt(2))
        _init_sequential(self.value_stream, output_gain=1.0)
        _init_sequential(self.advantage_stream, output_gain=0.01)

    def forward(self, market_state: torch.Tensor,
                portfolio_state: torch.Tensor,
                hidden: tuple = None) -> tuple:
        """
        Args:
            market_state: (batch, seq_len, n_stocks, n_features)
                          hoặc (batch, seq_len, n_stocks * n_features)
            portfolio_state: (batch, portfolio_dim)
            hidden: (h_0, c_0) hoặc None

        Returns:
            q_values: (batch, n_actions)
            new_hidden: (h_n, c_n)
        """
        if market_state.dim() == 4:
            batch, seq, stocks, feats = market_state.shape
            market_state = market_state.reshape(batch, seq, stocks * feats)

        lstm_features, new_hidden = self.feature_extractor(
            market_state, hidden
        )

        combined = torch.cat([lstm_features, portfolio_state], dim=-1)
        shared_out = self.shared_fc(combined)
        value = self.value_stream(shared_out)           # (batch, 1)
        advantage = self.advantage_stream(shared_out)   # (batch, n_actions)

        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values, new_hidden

    def init_hidden(self, batch_size: int,
                    device: torch.device = None) -> tuple:
        return self.feature_extractor.init_hidden(batch_size, device)

    def train(self, mode: bool = True):
        return _set_module_training_mode(self, mode)

    def eval(self):
        return self.train(False)

    def select_action(self, market_state: torch.Tensor,
                      portfolio_state: torch.Tensor,
                      hidden: tuple, epsilon: float = 0.0) -> tuple:
        """
        Chọn hành động theo epsilon-greedy.

        Args:
            market_state: (1, seq_len, input_size) — single observation
            portfolio_state: (1, portfolio_dim)
            hidden: (h, c)
            epsilon: xác suất chọn ngẫu nhiên

        Returns:
            action: int
            new_hidden: (h_n, c_n)
        """
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.n_actions)
            with torch.no_grad():
                _, new_hidden = self.forward(
                    market_state, portfolio_state, hidden
                )
            return action, new_hidden
        else:
            with torch.no_grad():
                q_values, new_hidden = self.forward(
                    market_state, portfolio_state, hidden
                )
                action = q_values.argmax(dim=-1).item()
            return action, new_hidden


class BranchingDRQNNetwork1(nn.Module):
    """
    Branching Dueling DRQN cho Multi-Discrete Action Space.

    Mỗi stock = 1 branch riêng:
        → mỗi branch có k actions (sell/hold/buy)

    Q(s,a) = V(s) + A_i(s,a_i) - mean(A_i)

    Tổng action space không còn là k * n_stocks (flat),
    mà là n_stocks nhánh độc lập.
    """

    def __init__(self, n_stocks: int, n_features: int,
                 seq_len: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 k: int = 3):

        super().__init__()

        self.n_stocks = n_stocks
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.k = k  # số action mỗi stock

        input_size = n_stocks * n_features
        portfolio_dim = 1 + n_stocks
        combined_dim = hidden_size + portfolio_dim

        # ===== LSTM =====
        self.feature_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ===== Shared =====
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ===== Value =====
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # ===== Branching Advantage =====
        self.advantage_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, k)
            )
            for _ in range(n_stocks)
        ])

        _init_sequential(self.shared_fc, output_gain=np.sqrt(2))
        _init_sequential(self.value_stream, output_gain=1.0)
        for adv in self.advantage_streams:
            _init_sequential(adv, output_gain=0.01)

    def forward(self, market_state, portfolio_state, hidden=None):
        """
        Returns:
            q_values: (batch, n_stocks, k)
        """

        if market_state.dim() == 4:
            b, t, s, f = market_state.shape
            market_state = market_state.reshape(b, t, s * f)

        lstm_features, new_hidden = self.feature_extractor(
            market_state, hidden
        )

        combined = torch.cat([lstm_features, portfolio_state], dim=-1)
        shared_out = self.shared_fc(combined)

        # Value
        value = self.value_stream(shared_out)  # (batch, 1)

        # Advantage per branch
        advantages = []
        for adv_head in self.advantage_streams:
            adv = adv_head(shared_out)  # (batch, k)
            advantages.append(adv)

        # stack → (batch, n_stocks, k)
        advantages = torch.stack(advantages, dim=1)

        # normalize advantage
        adv_mean = advantages.mean(dim=-1, keepdim=True)

        q_values = value.unsqueeze(1) + advantages - adv_mean

        return q_values, new_hidden

    def init_hidden(self, batch_size, device=None):
        return self.feature_extractor.init_hidden(batch_size, device)

    def select_action(self, market_state, portfolio_state,
                      hidden, epsilon=0.0):
        """
        Returns:
            actions: (n_stocks,) mỗi stock 1 action
        """

        if np.random.random() < epsilon:
            actions = np.random.randint(0, self.k, size=self.n_stocks)

            with torch.no_grad():
                _, new_hidden = self.forward(
                    market_state, portfolio_state, hidden
                )

            return actions, new_hidden

        else:
            with torch.no_grad():
                q_values, new_hidden = self.forward(
                    market_state, portfolio_state, hidden
                )

                # (1, n_stocks, k) → (n_stocks,)
                actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()

            return actions, new_hidden

class BranchingDRQNNetwork(nn.Module):
    """
    Branching Dueling DRQN cho Multi-Discrete Action Space.

    Mỗi stock = 1 branch riêng:
        → mỗi branch có k actions (sell/hold/buy)

    Q(s,a) = V(s) + A_i(s,a_i) - mean(A_i)

    Tổng action space không còn là k * n_stocks (flat),
    mà là n_stocks nhánh độc lập.
    """

    def __init__(self, n_stocks: int, n_features: int,
                 seq_len: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 k: int = 3):

        super().__init__()

        self.n_stocks = n_stocks
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.k = k  # số action mỗi stock

        input_size = n_stocks * n_features
        portfolio_dim = 1 + n_stocks
        combined_dim = hidden_size + portfolio_dim

        # ===== LSTM =====
        self.feature_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ===== Shared =====
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ===== Value =====
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # ===== Branching Advantage =====
        self.advantage_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, k)
            )
            for _ in range(n_stocks)
        ])

        _init_sequential(self.shared_fc, output_gain=np.sqrt(2))
        _init_sequential(self.value_stream, output_gain=1.0)
        for adv in self.advantage_streams:
            _init_sequential(adv, output_gain=0.01)

    def forward(self, market_state, portfolio_state, hidden=None):
        """
        Returns:
            q_values: (batch, n_stocks, k)
        """

        if market_state.dim() == 4:
            b, t, s, f = market_state.shape
            market_state = market_state.reshape(b, t, s * f)

        lstm_features, new_hidden = self.feature_extractor(
            market_state, hidden
        )

        combined = torch.cat([lstm_features, portfolio_state], dim=-1)
        shared_out = self.shared_fc(combined)

        # Value
        value = self.value_stream(shared_out)  # (batch, 1)

        # Advantage per branch
        advantages = []
        for adv_head in self.advantage_streams:
            adv = adv_head(shared_out)  # (batch, k)
            advantages.append(adv)

        # stack → (batch, n_stocks, k)
        advantages = torch.stack(advantages, dim=1)

        # normalize advantage
        adv_mean = advantages.mean(dim=-1, keepdim=True)

        q_values = value.unsqueeze(1) + advantages - adv_mean

        return q_values, new_hidden

    def init_hidden(self, batch_size, device=None):
        return self.feature_extractor.init_hidden(batch_size, device)

    def select_action(self, market_state, portfolio_state,
                      hidden, epsilon=0.0):
        """
        Returns:
            actions: (n_stocks,) mỗi stock 1 action
        """

        if np.random.random() < epsilon:
            actions = np.random.randint(0, self.k, size=self.n_stocks)

            with torch.no_grad():
                _, new_hidden = self.forward(
                    market_state, portfolio_state, hidden
                )

            return actions, new_hidden

        else:
            with torch.no_grad():
                q_values, new_hidden = self.forward(
                    market_state, portfolio_state, hidden
                )

                # (1, n_stocks, k) → (n_stocks,)
                actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()

            return actions, new_hidden

# ============================================================
#  Branching DRQN cho vector action per-stock
# ============================================================


class BranchingDRQNNetwork(nn.Module):
    """
    Branching Dueling DRQN.

    Mỗi cổ phiếu là một branch riêng với k action rời rạc (sell/hold/buy).
    Output Q-value có shape (batch, n_stocks, k), phù hợp với env discrete
    đang hỗ trợ vector action độ dài n_stocks.
    """

    def __init__(self, n_stocks: int, n_features: int,
                 seq_len: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 k: int = 3, max_positions: int = 3):
        super().__init__()

        self.n_stocks = n_stocks
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.k = k
        self.max_positions = max_positions
        input_size = n_stocks * n_features
        portfolio_dim = 1 + n_stocks
        combined_dim = hidden_size + portfolio_dim

        self.feature_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # thêm 1 layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.advantage_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),  # shallow hơn
                nn.ReLU(),
                nn.Linear(hidden_size // 2, self.k),
            )
            for _ in range(self.n_stocks)
        ])

        _init_sequential(self.shared_fc, output_gain=np.sqrt(2))
        _init_sequential(self.value_stream, output_gain=1.0)
        for stream in self.advantage_streams:
            _init_sequential(stream, output_gain=0.01)

    def forward(self, market_state: torch.Tensor,
                portfolio_state: torch.Tensor,
                hidden: tuple = None) -> tuple:
        if market_state.dim() == 4:
            batch, seq, stocks, feats = market_state.shape
            market_state = market_state.reshape(batch, seq, stocks * feats)

        lstm_features, new_hidden = self.feature_extractor(market_state, hidden)
        combined = torch.cat([lstm_features, portfolio_state], dim=-1)
        shared_out = self.shared_fc(combined)

        value = self.value_stream(shared_out).unsqueeze(-1)  # (batch, 1, 1)
        advantages = torch.stack(
            [stream(shared_out) for stream in self.advantage_streams],
            dim=1,
        )  # (batch, n_stocks, k)
        q_values = value + advantages - advantages.mean(dim=-1, keepdim=True)
        return q_values, new_hidden

    def init_hidden(self, batch_size: int,
                    device: torch.device = None) -> tuple:
        return self.feature_extractor.init_hidden(batch_size, device)

    def train(self, mode: bool = True):
        return _set_module_training_mode(self, mode)

    def eval(self):
        return self.train(False)

    def select_action(self, market_state: torch.Tensor,
                      portfolio_state: torch.Tensor,
                      hidden: tuple,
                      epsilon: float = 0.0) -> tuple:
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.k, size=(self.n_stocks,), dtype=np.int64)
            with torch.no_grad():
                _, new_hidden = self.forward(market_state, portfolio_state, hidden)
            return action, new_hidden

        with torch.no_grad():
            q_values, new_hidden = self.forward(market_state, portfolio_state, hidden)
            # action = q_values.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
            action = self.select_action_constrained(q_values, portfolio_state, self.max_positions).squeeze(0).cpu().numpy().astype(np.int64)
        return action, new_hidden
    def select_action_constrained(self, q_values, portfolio_state, max_positions=3):
        raw_action = q_values.argmax(dim=-1).squeeze(0)
        
        # Nếu số lệnh mua vượt ngưỡng → chỉ giữ top-k theo Q-value
        buy_mask = (raw_action == 2)  # action 2 = buy
        if buy_mask.sum() > max_positions:
            buy_q = q_values.squeeze(0)[:, 2]  # Q-value của action buy
            buy_q[~buy_mask] = -torch.inf
            _, top_k = buy_q.topk(max_positions)
            constrained = raw_action.clone()
            constrained[buy_mask] = 1  # reset về hold
            constrained[top_k] = 2    # chỉ mua top-k
            return constrained
        return raw_action


# ============================================================
#  Mạng PPO Actor-Critic
# ============================================================


class PPOLSTMActorCritic(nn.Module):
    """
    PPO Actor-Critic với LSTM cho Continuous Action Space.

    Actor: xuất tham số concentration cho phân phối Dirichlet trên simplex.
    Action luôn có tổng bằng 1, phù hợp trực tiếp với vector tỷ trọng mục tiêu
    (N stocks + 1 cash) mà environment thực thi.

    Có thể cố định total concentration của Dirichlet để sampling trong lúc train
    bám sát mean-action hơn. Điều này đặc biệt hữu ích khi deploy/eval dùng
    deterministic mean-action, tránh mismatch lớn giữa train-time sample và
    inference-time allocation.

    Architecture:
        market_state → LSTM → lstm_features (128)
                                    ↓ concat portfolio (17)
                              combined (145)
                              ↓                ↓
                         Actor Head       Critic Head
                              ↓                ↓
              concentration (n_stocks + 1)   value (1)
    """

    CONCENTRATION_MIN = 1e-3
    CONCENTRATION_MAX = 100.0
    ACTION_EPS = 1e-6

    def __init__(self, n_stocks: int, n_features: int,
                 seq_len: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 log_std_init: float = -0.5,
                 dirichlet_total_concentration: float = 0.0):
        """
        Args:
            n_stocks: Số lượng cổ phiếu (vd: 16)
            n_features: Số features mỗi cổ phiếu (vd: 7)
            seq_len: Độ dài chuỗi thời gian / window_size
            hidden_size: Kích thước hidden state LSTM
            num_layers: Số lớp LSTM
            dropout: Dropout rate
            log_std_init: Giữ lại để tương thích ngược với config cũ. Không còn dùng
                          sau khi policy chuyển sang Dirichlet trên simplex.
            dirichlet_total_concentration:
                Nếu > 0, normalize actor output về mean weights rồi scale về
                tổng concentration cố định này. Khi = 0, dùng legacy mode:
                mỗi chiều alpha_i được học tự do.
        """
        super().__init__()

        self.n_stocks = n_stocks
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.log_std_init = log_std_init
        self.dirichlet_total_concentration = float(dirichlet_total_concentration)
        if self.dirichlet_total_concentration < 0.0:
            raise ValueError("dirichlet_total_concentration phải >= 0.")

        input_size = n_stocks * n_features
        portfolio_dim = 1 + n_stocks
        combined_dim = hidden_size + portfolio_dim

        self.feature_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Actor: logits cho concentration của Dirichlet
        self.actor_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_stocks + 1),
        )

        # Critic: đầu ra giá trị trạng thái V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Actor: gain vừa phải để policy có thể diverge khỏi initial bias
        _init_sequential(self.actor_head, output_gain=0.1)
        _init_sequential(self.critic_head, output_gain=1.0)
        self._set_initial_dirichlet_bias(target_concentration=3.0)

    def _set_initial_dirichlet_bias(self, target_concentration: float) -> None:
        """Khởi tạo bias cuối để concentration ban đầu xấp xỉ 1.0 (gần uniform)."""
        if target_concentration <= self.CONCENTRATION_MIN:
            raise ValueError("target_concentration must be > CONCENTRATION_MIN")
        last_linear = [m for m in self.actor_head.modules() if isinstance(m, nn.Linear)][-1]
        target = target_concentration - self.CONCENTRATION_MIN
        bias_value = float(np.log(np.expm1(target)))
        with torch.no_grad():
            last_linear.bias.fill_(bias_value)

    def _get_policy_dist(self, concentration: torch.Tensor) -> Dirichlet:
        safe_concentration = torch.nan_to_num(
            concentration,
            nan=self.CONCENTRATION_MIN,
            posinf=self.CONCENTRATION_MAX,
            neginf=self.CONCENTRATION_MIN,
        )
        safe_concentration = torch.clamp(
            safe_concentration,
            min=self.CONCENTRATION_MIN,
            max=self.CONCENTRATION_MAX,
        )
        return Dirichlet(safe_concentration)

    def forward(self, market_state: torch.Tensor,
                portfolio_state: torch.Tensor,
                hidden: tuple = None) -> tuple:
        """
        Args:
            market_state: (batch, seq_len, n_stocks, n_features)
                          hoặc (batch, seq_len, n_stocks * n_features)
            portfolio_state: (batch, portfolio_dim)
            hidden: (h_0, c_0) hoặc None

        Returns:
            concentration: (batch, n_stocks + 1) — tham số Dirichlet > 0
            value: (batch, 1)
            new_hidden: (h_n, c_n)
        """
        if market_state.dim() == 4:
            batch, seq, stocks, feats = market_state.shape
            market_state = market_state.reshape(batch, seq, stocks * feats)

        lstm_features, new_hidden = self.feature_extractor(
            market_state, hidden
        )
        combined = torch.cat([lstm_features, portfolio_state], dim=-1)

        concentration_logits = self.actor_head(combined)
        concentration_logits = torch.nan_to_num(
            concentration_logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        concentration_scores = F.softplus(concentration_logits) + self.CONCENTRATION_MIN
        concentration_scores = torch.nan_to_num(
            concentration_scores,
            nan=self.CONCENTRATION_MIN,
            posinf=self.CONCENTRATION_MAX,
            neginf=self.CONCENTRATION_MIN,
        )
        concentration_scores = torch.clamp(
            concentration_scores,
            min=self.CONCENTRATION_MIN,
            max=self.CONCENTRATION_MAX,
        )
        if self.dirichlet_total_concentration > 0.0:
            concentration = concentration_scores / concentration_scores.sum(
                dim=-1, keepdim=True
            ).clamp_min(self.ACTION_EPS)
            concentration = concentration * self.dirichlet_total_concentration
        else:
            concentration = concentration_scores

        # Critic
        value = self.critic_head(combined)

        return concentration, value, new_hidden

    def init_hidden(self, batch_size: int,
                    device: torch.device = None) -> tuple:
        return self.feature_extractor.init_hidden(batch_size, device)

    def train(self, mode: bool = True):
        return _set_module_training_mode(self, mode)

    def eval(self):
        return self.train(False)

    def get_action(self, market_state: torch.Tensor,
                   portfolio_state: torch.Tensor,
                   hidden: tuple = None) -> tuple:
        """
        Lấy mẫu action từ policy (khi tương tác với environment).
        Action được lấy trực tiếp từ Dirichlet nên luôn nằm trên simplex.

        Returns:
            action: (batch, n_stocks + 1) — vector tỷ trọng tổng bằng 1
            action_for_buffer: (batch, n_stocks + 1) — lưu vào buffer để PPO update
            log_prob: (batch,) — log_prob của action theo Dirichlet
            value: (batch, 1)
            new_hidden: (h_n, c_n)
        """
        concentration, value, new_hidden = self.forward(
            market_state, portfolio_state, hidden
        )

        dist = self._get_policy_dist(concentration)
        action = dist.rsample()
        action = torch.clamp(action, min=self.ACTION_EPS)
        action = action / action.sum(dim=-1, keepdim=True)
        log_prob = dist.log_prob(action)

        return action, action.detach().clone(), log_prob, value, new_hidden

    def evaluate_actions(self, market_state: torch.Tensor,
                         portfolio_state: torch.Tensor,
                         actions: torch.Tensor,
                         hidden: tuple = None) -> tuple:
        """
        Đánh giá lại actions đã thực hiện (dùng trong PPO update).
        Nhận action simplex đã thực hiện để log_prob nhất quán với get_action().

        Lưu ý hidden state: hidden=None sẽ dùng zero-init và mất context thời gian.
        Trong PPO training nên lưu hidden state đầu mỗi rollout segment và truyền vào.

        Args:
            market_state: (batch, seq_len, input_size)
            portfolio_state: (batch, portfolio_dim)
            actions: (batch, n_stocks + 1) — action simplex từ get_action()
            hidden: (h_0, c_0) — nên là stored hidden state

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
            values: (batch, 1)
            new_hidden: (h_n, c_n)
        """
        concentration, values, new_hidden = self.forward(
            market_state, portfolio_state, hidden
        )

        dist = self._get_policy_dist(concentration)
        actions = torch.clamp(actions, min=self.ACTION_EPS)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values, new_hidden
