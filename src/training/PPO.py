import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.environment.trading_env import TradingEnv
from src.constants import TICKERS, FEATURES, WINDOW_SIZE, DATA_PATH


def load_processed_data(data_path: str, tickers: list, train_ratio: float = 0.8) -> tuple:
    """
    Load dữ liệu đã processed và split thành train/test theo thời gian
    Returns: (train_data_dict, test_data_dict) where dict[ticker] = df
    """
    data_dict = {}
    for ticker in tickers:
        file_path = os.path.join(data_path, f"{ticker}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['time'])
            df = df.sort_values('time').reset_index(drop=True)
            data_dict[ticker] = df
        else:
            print(f"Warning: {file_path} not found")

    if not data_dict:
        raise ValueError("No data files found")

    # Split theo thời gian cho mỗi ticker
    train_data = {}
    test_data = {}
    for ticker, df in data_dict.items():
        n_train = int(len(df) * train_ratio)
        train_data[ticker] = df.iloc[:n_train].copy()
        test_data[ticker] = df.iloc[n_train:].copy()

    return train_data, test_data


def train_ppo(total_timesteps: int = 100000, save_path: str = "saved_models/ppo_model", train_ratio: float = 0.8):
    """
    Hàm training PPO với dữ liệu từ data/processed, split train/test
    """
    # 1. Load và split dữ liệu
    print("Loading and splitting data...")
    train_data, test_data = load_processed_data(DATA_PATH, TICKERS, train_ratio)
    print(f"Train data: {len(train_data[TICKERS[0]])} days per ticker")
    print(f"Test data: {len(test_data[TICKERS[0]])} days per ticker")

    # 2. Khởi tạo môi trường training
    print("Initializing training environment...")
    env = TradingEnv(
        tickers=TICKERS,
        mode="continuous",  # PPO sử dụng continuous action space
        initial_balance=1_000_000_000,  # 1 tỷ VNĐ
        fee_rate=0.0015,  # Phí giao dịch
        reward_type="simple",
        window_size=WINDOW_SIZE,
        data_dict=train_data,
        features=FEATURES,
        max_steps=100,
        random_start=True
    )

    # Wrap env cho stable-baselines3
    env = DummyVecEnv([lambda: env])

    # 3. Thiết lập siêu tham số PPO
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    # 4. Khởi tạo model PPO
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./results/",
        **ppo_params
    )

    # 5. Thiết lập callback để evaluate trên test data
    eval_env = TradingEnv(
        tickers=TICKERS,
        mode="continuous",
        initial_balance=1_000_000_000,
        fee_rate=0.0015,
        reward_type="simple",
        window_size=WINDOW_SIZE,
        data_dict=test_data,
        features=FEATURES,
        max_steps=100,
        random_start=True
    )
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./saved_models/best_ppo/",
        log_path="./results/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # 6. Training
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    # 7. Lưu model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model


def evaluate_ppo(model_path: str, test_data_dict: dict = None, n_episodes: int = 10):
    """
    Hàm đánh giá model PPO trên test data
    Tính các metrics: total return, Sharpe ratio, max drawdown, win rate
    """
    if test_data_dict is None:
        # Nếu không cung cấp, load và split từ toàn bộ data (giả sử test là 20% cuối)
        _, test_data_dict = load_processed_data(DATA_PATH, TICKERS, 0.8)

    # Load model
    model = PPO.load(model_path)

    # Khởi tạo eval env
    eval_env = TradingEnv(
        tickers=TICKERS,
        mode="continuous",
        initial_balance=1_000_000_000,
        fee_rate=0.0015,
        reward_type="simple",
        window_size=WINDOW_SIZE,
        data_dict=test_data_dict,
        features=FEATURES,
        max_steps=100,
        random_start=True
    )

    returns = []
    portfolio_values = []

    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_returns = []
        episode_values = [eval_env.portfolio_value]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_returns.append(reward)
            episode_values.append(eval_env.portfolio_value)
            done = done or truncated

        total_return = (eval_env.portfolio_value - eval_env.initial_balance) / eval_env.initial_balance
        returns.append(total_return)
        portfolio_values.append(episode_values)

        print(f"Episode {episode+1}: Total Return = {total_return:.4f}")

    # Tính metrics
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0

    # Max drawdown
    max_drawdowns = []
    for values in portfolio_values:
        peak = max(values)
        drawdown = (peak - min(values)) / peak
        max_drawdowns.append(drawdown)
    avg_max_drawdown = np.mean(max_drawdowns)

    # Win rate
    win_rate = np.mean([1 if r > 0 else 0 for r in returns])

    print("\nEvaluation Results:")
    print(f"Average Total Return: {avg_return:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Average Max Drawdown: {avg_max_drawdown:.4f}")
    print(f"Win Rate: {win_rate:.4f}")

    # # Dọn dẹp nếu tạo temp
    # if test_data_path.startswith("./temp"):
    #     shutil.rmtree(test_data_path, ignore_errors=True)

    return {
        "avg_return": avg_return,
        "sharpe_ratio": sharpe_ratio,
        "avg_max_drawdown": avg_max_drawdown,
        "win_rate": win_rate
    }


if __name__ == "__main__":
    # Chạy training
    trained_model = train_ppo(total_timesteps=50000)

    # Load test data để đánh giá
    _, test_data = load_processed_data(DATA_PATH, TICKERS, 0.8)
    eval_results = evaluate_ppo("saved_models/ppo_model", test_data)