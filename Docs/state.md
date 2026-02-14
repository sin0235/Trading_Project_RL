<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Hướng dẫn chi tiết xây dựng State Space cho DQN và PPO

State Space là **tim mạch** của environment RL. Dưới đây là hướng dẫn từng bước với code production-ready.

***

## Thiết kế State Space

### 1.1. State Space chung (dùng cho cả DQN và PPO)

**Nguyên tắc:** State phải chứa đủ thông tin để agent đưa ra quyết định tốt, nhưng không quá phức tạp gây overfit.

#### **Cấu trúc State gồm 2 phần:**

```
State = [Market Information] + [Portfolio Information]
```

**A. Market Information (Thông tin thị trường)**

Với mỗi mã cổ phiếu, tại mỗi thời điểm trong window:


| Feature | Mô tả | Lý do chọn |
| :-- | :-- | :-- |
| `close_norm` | Giá đóng cửa chuẩn hóa | Xu hướng giá |
| `return_1d` | Return 1 ngày | Momentum ngắn hạn |
| `return_5d` | Return 5 ngày | Momentum trung hạn |
| `macd` | MACD indicator | Xu hướng \& momentum |
| `rsi` | RSI indicator | Overbought/oversold |
| `volume_norm` | Volume chuẩn hóa | Thanh khoản |

→ **6 features/mã/ngày**

**B. Portfolio Information (Thông tin danh mục)**


| Feature | Công thức | Ý nghĩa |
| :-- | :-- | :-- |
| `cash_ratio` | cash / portfolio_value | Tỉ lệ tiền mặt còn lại |
| `holdings_ratio[i]` | (holdings[i] × price[i]) / portfolio_value | Tỉ lệ vốn ở mỗi mã |

→ **1 + N features** (với N = số mã)

#### **Tổng State Dimension:**

$$
\text{State Dim} = (\text{window\_size} \times N_{\text{stocks}} \times N_{\text{features}}) + (1 + N_{\text{stocks}})
$$

**Ví dụ cụ thể:**

- Window = 30 ngày
- N stocks = 5 mã
- N features = 6
- Portfolio = 1 + 5

→ **State Dim = 30 × 5 × 6 + 6 = 906 chiều**

### 1.2. Sự khác biệt giữa DQN và PPO về State

| Khía cạnh | DQN | PPO | Ghi chú |
| :-- | :-- | :-- | :-- |
| **State structure** | Giống nhau | Giống nhau | Đều dùng 906-dim vector |
| **Preprocessing** | Flatten to 1D | Flatten to 1D | Giống nhau |
| **LSTM support** | Không cần | Có thể dùng | PPO+LSTM cho sequential patterns |
| **State history** | 1 state/step | 1 state/step (hoặc sequence) | LSTM cần sequence |

**Kết luận:** State Space **cơ bản giống nhau**. Điểm khác chính:

- **DQN**: Dùng state flatten đơn giản
- **PPO+LSTM**: Có thể dùng sequence of states (sliding window)

***