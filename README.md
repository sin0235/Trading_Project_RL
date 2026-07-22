<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=180&section=header&text=RL%20Portfolio%20Trading&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=PPO%20%E2%80%A2%20DDQ%20%E2%80%A2%20Branching%20DDQ%20%E2%80%A2%20Vietnamese%20Stocks&descAlignY=56&descSize=17" width="100%"/>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-Agents-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
    <img src="https://img.shields.io/badge/Task-Portfolio%20Allocation-6A5ACD?style=for-the-badge" alt="Task"/>
  </p>
</div>

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

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>
  <em>Reinforcement learning experiments for portfolio allocation.</em>
</div>
