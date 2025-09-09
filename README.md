# 📈 Momentum Rotation Trading Bot

A Python-based **ETF momentum rotation strategy** that automatically rotates into the strongest-trending ETFs and moves to a defensive asset when momentum is weak.  

Supports:
- **Backtesting** with Yahoo Finance (`yfinance`)
- **Live trading** with Alpaca (paper/live accounts)
- **Profiles** for different risk levels (`conservative`, `balanced`, `aggressive`)
- **Optional SMS alerts** via Twilio
- **GitHub Actions automation**

---

## 🚀 Features

- **Momentum strategy**: Picks ETFs with the strongest blended momentum (1, 3, 6, 12 month lookbacks depending on profile).  
- **Risk management**:
  - Risk-off asset (SHY or GLD depending on profile)
  - Optional per-position stop-loss and portfolio circuit breaker
- **Profiles**:
  - 🟢 `conservative`: broad ETFs, defensive SHY, slower momentum  
  - 🟡 `balanced`: growth tilt (QQQ, XLK, ARKK), GLD risk-off, 2 winners, stops enabled  
  - 🔴 `aggressive`: leveraged ETFs (TQQQ, SOXL, ARKK), Top-1 only, high risk/high reward  
- **Modes**:
  - `backtest`: test strategies on historical data
  - `live`: run trades via Alpaca (supports one-shot or continuous loop)
- **Automation**: Works with **GitHub Actions** to run on a schedule
- **Notifications**: Optional SMS alerts using Twilio

---

## 🛠 Setup

### 1. Clone and install dependencies
```bash
git clone https://github.com/yourname/stock_bot.git
cd stock_bot
pip install -r requirements.txt
```

### 2. Configure secrets  
Create a `.env` file (or use GitHub Secrets):
```ini
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets

# Optional: Twilio SMS alerts
TWILIO_ACCOUNT_SID=xxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_FROM=+1234567890
TWILIO_TO=+1987654321
```

---

## 📊 Backtesting

Run a backtest with `yfinance`:

```bash
# Balanced profile, weekly rotation since 2016
python momentum_rotation.py --profile balanced backtest --start 2016-01-01 --freq weekly

# Aggressive profile, monthly rotation since 2016
python momentum_rotation.py --profile aggressive backtest --start 2016-01-01 --freq monthly
```

Outputs:
- Console stats:
  - Final equity
  - Return %
  - Sharpe ratio
  - Max drawdown
- CSV equity curve: `equity_weekly.csv` or `equity_monthly.csv`

---

## 🤖 Live Trading with Alpaca

Run in **paper** or **live** trading mode:

```bash
# Balanced profile, weekly rebalance, one-shot (for Actions)
python momentum_rotation.py --profile balanced live --freq weekly --once

# Aggressive profile, monthly rebalance, continuous loop
python momentum_rotation.py --profile aggressive live --freq monthly
```

Options:
- `--once`: Run one pass (good for scheduled GitHub Actions)
- `--force`: Run even if market closed (for testing)
- `--after-hours`: If market is closed, queue LIMIT orders with extended hours

---

## ⚡ GitHub Actions Automation

Example workflow (`.github/workflows/run.yml`):

```yaml
name: Momentum Rotation Bot

on:
  schedule:
    - cron: "35 14 * * 1-5"   # Every weekday at 10:35 ET
  workflow_dispatch:

jobs:
  trade:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.11
      - run: pip install -r requirements.txt
      - name: Restore state
        uses: actions/cache@v4
        with:
          path: rotation_state.json
          key: rotation-state-${{ runner.os }}-${{ github.ref_name }}
          restore-keys: |
            rotation-state-
      - name: Run rotation
        run: python momentum_rotation.py --profile balanced live --freq weekly --once
        env:
          APCA_API_KEY_ID: ${{ secrets.APCA_API_KEY_ID }}
          APCA_API_SECRET_KEY: ${{ secrets.APCA_API_SECRET_KEY }}
          APCA_API_BASE_URL: https://paper-api.alpaca.markets
          TWILIO_ACCOUNT_SID: ${{ secrets.TWILIO_ACCOUNT_SID }}
          TWILIO_AUTH_TOKEN: ${{ secrets.TWILIO_AUTH_TOKEN }}
          TWILIO_FROM: ${{ secrets.TWILIO_FROM }}
          TWILIO_TO: ${{ secrets.TWILIO_TO }}
```

---

## 📈 Profiles Summary

| Profile        | Universe                              | Risk-off | Top N | Lookbacks | Stops       | Risk/Reward |
|----------------|---------------------------------------|----------|-------|-----------|-------------|-------------|
| Conservative   | SPY, QQQ, IWM, EFA, EEM               | SHY      | 2     | [3,6,12]  | Disabled    | Low risk    |
| Balanced       | QQQ, XLK, SPY, ARKK, IWM              | GLD      | 2     | [1,3,6]   | –10% / –15% | Moderate    |
| Aggressive     | QQQ, TQQQ, SOXL, ARKK, SPY            | GLD      | 1     | [1,3]     | –12% / –20% | High risk   |

---

## ⚠️ Disclaimer

This bot is for **educational purposes only**.  
Past performance ≠ future results. Trading involves risk.  
Use **paper trading** first before committing real money.  
