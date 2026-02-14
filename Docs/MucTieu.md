1. Mục đích tổng quát (Big Picture)
Nghiên cứu và đánh giá khả năng ứng dụng Deep Reinforcement Learning (DRL) vào bài toán giao dịch cổ phiếu tự động trên thị trường chứng khoán Việt Nam.

Tại sao đề tài này có ý nghĩa?
Về mặt công nghệ:

Deep RL đã thành công trong gaming (AlphaGo, Atari), robotics, nhưng ứng dụng vào tài chính Việt Nam còn rất hạn chế.
​

Các nghiên cứu chủ yếu tập trung vào thị trường phát triển (US, EU, Trung Quốc) - ít research riêng cho VN.

Về mặt thực tiễn:

Thị trường VN là emerging market: biến động cao, thanh khoản thấp hơn, nhiều yếu tố phi lý tính (news-driven, nhà đầu tư cá nhân chiếm đa số).

Chiến lược trading truyền thống (buy-and-hold, technical analysis đơn giản) thường không tối ưu trong môi trường này.

RL có thể học được chiến lược thích ứng (adaptive strategy) từ dữ liệu lịch sử mà không cần giả định về phân phối giá.

Về mặt giáo dục/học thuật:

Áp dụng kiến thức RL vào bài toán thực tế phức tạp.

So sánh 2 nhánh chính của RL (value-based vs policy-based) trên cùng 1 môi trường.

2. Mục đích cụ thể (Specific Objectives)
Đề tài hướng đến 5 mục tiêu cụ thể:

Mục tiêu 1: Xây dựng môi trường mô phỏng giao dịch cổ phiếu Việt Nam
Nội dung:

Thu thập & tiền xử lý dữ liệu lịch sử từ thị trường VN (VN30 hoặc subset).

Thiết kế State Space (không gian trạng thái) phản ánh:

Thông tin thị trường: giá, volume, chỉ báo kỹ thuật (MACD, RSI, CCI, ADX) theo chuỗi thời gian.

Thông tin danh mục: tiền mặt, vị thế cổ phiếu hiện tại.

Thiết kế Action Space (không gian hành động):

Discrete (mua/bán/giữ) cho DQN.

Continuous (tỉ lệ phân bổ vốn) cho PPO.

Thiết kế Reward Function (hàm phần thưởng):

Tối đa hóa lợi nhuận sau phí giao dịch.

(Nâng cao: tối ưu risk-adjusted return như Sharpe ratio).

Đảm bảo tuân thủ ràng buộc thị trường VN:

Không short selling (không bán khống).

Không margin (không vay ký quỹ).

Phí giao dịch thực tế (~0.15% mỗi lệnh).

Đầu ra: Môi trường RL tương thích với OpenAI Gym / Stable-Baselines3.

Mục tiêu 2: Triển khai thuật toán DQN (Deep Q-Network)
Nội dung:

Áp dụng DQN - thuật toán value-based, critic-only:

Học hàm giá trị hành động 
Q
(
s
,
a
)
Q(s,a) để đánh giá "tốt/xấu" của mỗi hành động tại mỗi trạng thái.

Sử dụng experience replay & target network để ổn định training.
​

Test trên:

Single-stock (1 mã bluechip VN) hoặc

Multi-stock với action discrete cho từng mã.

Đầu ra:

Agent DQN được train trên dữ liệu VN 2015-2020 (train), test 2021-2023.

Performance metrics: cumulative return, Sharpe, max drawdown.

Mục tiêu 3: Triển khai thuật toán PPO với LSTM
Nội dung:

Áp dụng PPO - thuật toán policy gradient, actor-critic:
​

Học chính sách 
π
(
a
∣
s
)
π(a∣s) trực tiếp, phù hợp với action liên tục (phân bổ vốn multi-stock).

Dùng clipped objective để đảm bảo update ổn định.

Kết hợp LSTM làm feature extractor:
​

Xử lý chuỗi thời gian (30 ngày lịch sử).

Phát hiện pattern thời gian (trend, volatility regime) mà MLP không bắt được.

(Tuỳ chọn) Áp dụng turbulence index để agent dừng giao dịch khi thị trường crash.
​

Đầu ra:

Agent PPO+LSTM train trên multi-stock VN30 subset.

So sánh với DQN và baseline (Buy & Hold).

Mục tiêu 4: So sánh hiệu suất 2 thuật toán trên dữ liệu Việt Nam
Nội dung:

Backtest cả 2 agent trên cùng test set (out-of-sample 2021-2023).

So sánh theo nhiều tiêu chí:

Return: Cumulative return, annualized return.

Risk: Max drawdown, volatility.

Risk-adjusted: Sharpe ratio, Sortino ratio.

Trading behavior: Win rate, avg profit per trade, số lệnh.

So sánh với baseline:

Buy & Hold VN30-Index.

(Tuỳ chọn) Chiến lược kỹ thuật đơn giản: MACD crossover, time-series momentum.
​

Đầu ra:

Bảng so sánh metrics.

Biểu đồ equity curve (đường vốn theo thời gian).

Phân tích điểm mạnh/yếu từng thuật toán.

Mục tiêu 5: Đánh giá tính khả thi & đề xuất cải tiến
Nội dung:

Phân tích tính thực tiễn:

Agent có học được chiến lược "hợp lý" không?

Có bị overfit vào training set không?

Có chịu được sự kiện cực đoan (COVID-19, market crash 2022) không?

Hạn chế của phương pháp hiện tại:

Data Việt Nam ngắn hơn US/China → dễ overfit.

Chưa tích hợp sentiment (tin tức), fundamental (báo cáo tài chính).

Chưa xét đến chi phí cơ hội, slippage (trượt giá).

Hướng phát triển:

Tích hợp sentiment analysis từ tin tức VN (VNExpress, CafeF).
​

Thêm fundamental indicators (P/E, ROE, debt ratio).
​

Test thêm thuật toán khác (SAC, A2C, multi-agent RL).

Mở rộng lên toàn bộ VN30 hoặc thị trường rộng hơn.

Đầu ra:

Chương thảo luận & kết luận trong báo cáo.

Roadmap cho nghiên cứu tiếp theo (nếu làm luận văn tốt nghiệp).

3. Câu hỏi nghiên cứu chính (Research Questions)
Để làm rõ mục đích, bạn có thể đặt 3-4 câu hỏi nghiên cứu:

RQ1: Liệu Deep Reinforcement Learning có thể học được chiến lược giao dịch sinh lời trên thị trường chứng khoán Việt Nam?

RQ2: Thuật toán nào (DQN hay PPO+LSTM) phù hợp hơn với đặc điểm thị trường VN (biến động cao, thanh khoản thấp, emerging market)?

RQ3: Agent RL có vượt trội hơn chiến lược Buy & Hold và các phương pháp kỹ thuật truyền thống không (xét cả return và risk)?

RQ4: Các yếu tố nào trong thiết kế môi trường RL (state, action, reward) ảnh hưởng mạnh nhất đến hiệu suất agent?

4. Phạm vi & giới hạn đề tài
Để định vị rõ, cần nói:

Phạm vi
Thị trường: Cổ phiếu niêm yết trên HOSE (VN30 subset, 5-10 mã).

Thời gian: Dữ liệu lịch sử 2015-2023 (8-10 năm).

Tần suất giao dịch: Daily (T+1) - không làm intraday.

Loại trading:

Long-only (không short).

Không leverage.

Thuật toán: DQN & PPO (2 đại diện cho value-based và policy-based).

Giới hạn
Không xét:

Derivatives (futures, options).

High-frequency trading (HFT).

Transaction cost ngoài phí môi giới (slippage, market impact).

Giả định đơn giản hóa:

Agent giao dịch với khối lượng nhỏ → không ảnh hưởng giá thị trường.

Có thể mua/bán tại giá close của ngày (thực tế có thể trượt giá).

Không xét tax (thuế TNCN từ giao dịch CP).

5. Ý nghĩa & đóng góp của đề tài
5.1. Ý nghĩa khoa học
Đóng góp cho literature về RL in finance:

Nghiên cứu đầu tiên (hoặc trong số rất ít) so sánh DQN vs PPO+LSTM trên dữ liệu Việt Nam.

Mở rộng kiến thức về khả năng generalization của DRL sang emerging markets.

Xác minh các findings từ paper quốc tế:

Zhang et al. (2019): DQN có thể beat baseline time-series momentum trên futures.
​

Zou et al. (2023): PPO+LSTM outperform ensemble DRL, đặc biệt tốt trên thị trường mới nổi (China → VN tương đồng?).
​

5.2. Ý nghĩa thực tiễn
Cho nhà đầu tư cá nhân:

Cung cấp tool tự động hóa giao dịch, giảm cảm xúc/thiên kiến.

Có thể backtest chiến lược trước khi bỏ vốn thật.

Cho tổ chức tài chính (fund, securities firms):

Tham khảo để phát triển algorithmic trading system.

Áp dụng RL vào portfolio management, risk control.

Cho cộng đồng RL/AI Việt Nam:

Cung cấp codebase mở (nếu public) để nghiên cứu tiếp.

Tăng nhận thức về tiềm năng AI trong tài chính VN.

5.3. Ý nghĩa giáo dục
Sinh viên (bạn và nhóm):

Nắm vững lý thuyết RL (MDP, value-based, policy-based).

Thực hành end-to-end ML project: data → model → evaluation.

Biết cách đọc và replicate paper quốc tế.

6. Cách viết mục "Mục đích nghiên cứu" trong báo cáo
Dưới đây là mẫu đoạn văn bạn có thể tham khảo:

1.2. Mục đích nghiên cứu
Đề tài này nhằm nghiên cứu và đánh giá khả năng ứng dụng Deep Reinforcement Learning vào bài toán giao dịch cổ phiếu tự động trên thị trường chứng khoán Việt Nam, với các mục tiêu cụ thể sau:

Xây dựng môi trường mô phỏng giao dịch cổ phiếu phù hợp với đặc thù thị trường Việt Nam, bao gồm:

Thu thập và tiền xử lý dữ liệu lịch sử VN30 từ nguồn vnstock.

Thiết kế không gian trạng thái (state space) tích hợp thông tin giá, chỉ báo kỹ thuật và danh mục đầu tư.

Thiết kế không gian hành động (action space) phù hợp với từng thuật toán (discrete cho DQN, continuous cho PPO).

Thiết kế hàm phần thưởng (reward function) tối đa hóa lợi nhuận sau phí giao dịch, tuân thủ các ràng buộc: không short selling, không margin, phí giao dịch thực tế.

Triển khai và huấn luyện hai thuật toán Deep RL đại diện:

DQN (Deep Q-Network): thuật toán value-based, học hàm giá trị hành động 
Q
(
s
,
a
)
Q(s,a) để ra quyết định giao dịch với action space rời rạc.

PPO (Proximal Policy Optimization) kết hợp LSTM: thuật toán policy gradient với actor-critic, tận dụng LSTM để xử lý chuỗi thời gian và hỗ trợ action space liên tục (phân bổ vốn đa tài sản).
​

Đánh giá và so sánh hiệu suất của hai thuật toán trên dữ liệu out-of-sample (2021-2023), theo các tiêu chí:

Lợi nhuận tích lũy (cumulative return), tỷ suất lợi nhuận hàng năm (annualized return).

Rủi ro: độ sụt giảm tối đa (max drawdown), độ biến động (volatility).

Hiệu suất điều chỉnh rủi ro: Sharpe ratio, Sortino ratio.

So sánh với baseline: chiến lược Buy & Hold VN30-Index và các phương pháp kỹ thuật đơn giản.

Phân tích tính khả thi và đề xuất hướng cải tiến:

Đánh giá khả năng thực thi thực tế của các chiến lược RL học được.

Xác định hạn chế (overfitting, sensitivity to hyperparameters, data scarcity).

Đề xuất hướng phát triển: tích hợp sentiment analysis, fundamental indicators, mở rộng sang nhiều tài sản hơn.

Thông qua việc thực hiện các mục tiêu trên, đề tài kỳ vọng trả lời câu hỏi nghiên cứu chính: Liệu Deep Reinforcement Learning có thể học được chiến lược giao dịch hiệu quả trên thị trường chứng khoán Việt Nam, và thuật toán nào (DQN hay PPO+LSTM) phù hợp hơn với đặc điểm thị trường này?