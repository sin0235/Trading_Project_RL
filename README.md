# Reinforcement Learning for Portfolio Trading

An academic project that compares reinforcement learning agents for portfolio allocation on Vietnamese stock-market data.

## What is included

- PPO-LSTM, DDQ-LSTM, and Branching DDQ-LSTM agents
- A configurable trading environment with transaction fees and risk-aware rewards
- Market features including returns, MACD, RSI, ADX, volume, and volatility
- Time-based train, validation, and test splits
- Training, comparison, and data-analysis notebooks

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

The notebooks in `notebooks/` provide the easiest entry point. Training can also be started directly:

```bash
python -m src.training.PPO --config Conf/ppo_conf.yaml
python -m src.training.DDQ --config Conf/ddq_conf.yaml
python -m src.training.BranchingDDQ --config Conf/branching_ddq_conf.yaml
```

Run the lightweight logic checks with:

```bash
python -m unittest discover -s tests
```

## Project structure

```text
Conf/          experiment and environment configuration
data/          processed market data
notebooks/     training, comparison, and analysis workflows
src/agents/    reinforcement learning agents
src/environment/ trading environment, actions, states, and rewards
src/training/  training entry points
tests/         data, environment, reward, and configuration checks
```

> This project is for research and education, not financial advice.
