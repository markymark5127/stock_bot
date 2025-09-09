# 📈 Momentum Rotation Trading Bot

A Python-based **ETF momentum rotation strategy** that rotates into the **strongest-trending ETFs** on a schedule (weekly or monthly) and moves to a **defensive asset** when momentum is weak.

- ✅ **Backtesting** (Yahoo Finance / `yfinance`)
- ✅ **Live trading** (Alpaca paper/live)
- ✅ **Profiles** for different risk levels (`conservative`, `balanced`, `aggressive`)
- ✅ **Optional SMS** alerts (Twilio)
- ✅ **GitHub Actions** automation
- ✅ **Compare profiles** helper script

---

## 🔧 How it works (high-level)

1. Compute a **blended momentum score** for each ETF in the *universe* using past returns (1/3/6/12 months depending on profile).
2. Pick the **Top N** ETFs with **positive** momentum (or use risk-off if none).
3. Rebalance **weekly** (first trading day of the week) or **monthly** (first trading day of the month).
4. In live mode, place orders via **Alpaca**; in backtests, simulate with **Yahoo Finance** data.

---

## 🧪 Profiles

| Profile        | Universe                              | Risk-off | Top N | Lookbacks | Stops (per/portfolio) | Risk/Reward |
|----------------|---------------------------------------|----------|-------|-----------|------------------------|-------------|
| **conservative** | SPY, QQQ, IWM, EFA, EEM             | SHY      | 2     | [3,6,12]  | disabled/disabled      | Low         |
| **balanced**     | QQQ, XLK, SPY, ARKK, IWM            | GLD      | 2     | [1,3,6]   | –10% / –15%            | Medium      |
| **aggressive**   | QQQ, TQQQ, SOXL, ARKK, SPY          | GLD      | 1     | [1,3]     | –12% / –20%            | High        |

> You can switch profiles at runtime with `--profile`.

---

## 📦 Installation

```bash
git clone https://github.com/yourname/stock_bot.git
cd stock_bot
pip install -r requirements.txt
```

Create a `.env` (or use GitHub Secrets):

```ini
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_FEED=iex  # use free IEX feed (prevents SIP paywall errors)

# Optional: Twilio SMS alerts
TWILIO_ACCOUNT_SID=xxx
TWILIO_AUTH_TOKEN=xxx
TWILIO_FROM=+1234567890
TWILIO_TO=+1987654321
```

---

## 🚀 Commands & Flags (full reference)

The script has **two modes**: `backtest` and `live`. There’s also a **global `--profile`**.

### Global

- `--profile {conservative|balanced|aggressive}`  
  Selects a preset universe, lookbacks, stops, etc.  
  **Default:** `balanced`.

### Backtest mode

```bash
python momentum_rotation.py --profile balanced backtest [flags...]
```

**Flags:**
- `--start YYYY-MM-DD`  
  Start date for historical data (**required** for meaningful tests).
- `--end YYYY-MM-DD`  
  End date (optional; defaults to latest).
- `--freq {weekly|monthly}`  
  Rebalance cadence for the simulation. Default: profile default (often `weekly`).
- `--topn N`  
  **Override** the profile’s `TOP_N` (e.g., try `--topn 1` vs `--topn 2`).
- `--months a,b,c`  
  **Override** lookbacks (e.g., `--months 1,3,6`).
- `--abs X`  
  **Override** momentum threshold (e.g., `--abs 0.0` means only positive momentum is eligible).

**Outputs:**
- Console stats (final equity, return %, Sharpe, max drawdown).
- `equity_weekly.csv` or `equity_monthly.csv` with the equity curve.

### Live mode

```bash
python momentum_rotation.py --profile balanced live [flags...]
```

**Flags:**
- `--freq {weekly|monthly}`  
  Live rebalance schedule.
- `--once`  
  Run **one pass** and exit (ideal for GitHub Actions).
- `--force`  
  Run even if the market is closed (testing only; **orders may not fill**).
- `--after-hours`  
  If market is closed, the bot places **LIMIT** orders at the last price and sets `extended_hours=True` so they can queue for AH sessions.

**Notes:**
- Live mode uses:
  - **Alpaca IEX** data (`ALPACA_DATA_FEED=iex`) for free historical/latest bars.
  - A local `rotation_state.json` to remember the last period key, entries, and equity.
- The script will **not rebalance more than once per period** thanks to `rotation_state.json`.  
  In CI (GitHub Actions), the workflow **caches** this file to persist across runs.

---

## 📊 Examples

### Backtests
```bash
# Compare profiles (weekly) over a long window
python momentum_rotation.py --profile conservative backtest --start 2016-01-01 --freq weekly
python momentum_rotation.py --profile balanced     backtest --start 2016-01-01 --freq weekly
python momentum_rotation.py --profile aggressive   backtest --start 2016-01-01 --freq weekly

# Shorter, reactive lookbacks and Top-1 override on balanced
python momentum_rotation.py --profile balanced backtest --start 2020-01-01 --freq weekly --months 1,3 --topn 1

# Monthly cadence (smoother)
python momentum_rotation.py --profile balanced backtest --start 2016-01-01 --freq monthly
```

### Live (paper trading)
```bash
# Weekly, one-shot (run manually or via Actions)
python momentum_rotation.py --profile balanced live --freq weekly --once

# Monthly, continuous loop (checks every 60s; exits immediately if no rebalance day)
python momentum_rotation.py --profile aggressive live --freq monthly
```

> For **paper vs live trading**: use the appropriate Alpaca endpoint in `APCA_API_BASE_URL`.  
> Paper: `https://paper-api.alpaca.markets` (recommended!)  
> Live:  `https://api.alpaca.markets` (real money)

---

## 🤝 GitHub Actions (automation)

Example workflow (`.github/workflows/run.yml`) that runs **every weekday** at 10:35 ET:

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
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: |
          pip install --upgrade pip
          pip install -r requirements.txt
      - name: Create .env
        run: |
          echo "APCA_API_KEY_ID=${{ secrets.APCA_API_KEY_ID }}" >> .env
          echo "APCA_API_SECRET_KEY=${{ secrets.APCA_API_SECRET_KEY }}" >> .env
          echo "APCA_API_BASE_URL=https://paper-api.alpaca.markets" >> .env
          echo "ALPACA_DATA_FEED=iex" >> .env
          if [ -n "${{ secrets.TWILIO_ACCOUNT_SID }}" ]; then
            echo "TWILIO_ACCOUNT_SID=${{ secrets.TWILIO_ACCOUNT_SID }}" >> .env
            echo "TWILIO_AUTH_TOKEN=${{ secrets.TWILIO_AUTH_TOKEN }}" >> .env
            echo "TWILIO_FROM=${{ secrets.TWILIO_FROM }}" >> .env
            echo "TWILIO_TO=${{ secrets.TWILIO_TO }}" >> .env
          fi
      - name: Restore state
        uses: actions/cache@v4
        with:
          path: rotation_state.json
          key: rotation-state-${{ runner.os }}-${{ github.ref_name }}
          restore-keys: |
            rotation-state-
      - name: Run rotation (weekly)
        run: python momentum_rotation.py --profile balanced live --freq weekly --once
      - name: Save state
        if: always()
        uses: actions/cache@v4
        with:
          path: rotation_state.json
          key: rotation-state-${{ runner.os }}-${{ github.ref_name }}-${{ github.run_id }}
```

---

## 📊 Compare profiles (helper script)

Use the included helper to run all profiles and produce a **comparison CSV + chart**.

```bash
python compare_backtests.py --start 2016-01-01 --freq weekly
# Optional end date:
python compare_backtests.py --start 2020-01-01 --end 2025-09-08 --freq monthly
```

Outputs:
- `comparison_summary.csv` — final equity, return %, Sharpe, max drawdown for each profile  
- `comparison_equity.png` — all equity curves on one plot

> The script simply runs `momentum_rotation.py --profile <x> backtest ...` for each profile, then aggregates results.

---

## 📣 SMS notifications (optional)

Set these in `.env` (or GitHub Secrets) to receive texts for rebalances and stops:
```
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM=+1234567890
TWILIO_TO=+1987654321
```

---

## 🧠 Tips

- **First run** after changing profiles/params: consider deleting `rotation_state.json` so the new schedule/period key starts fresh.
- **IEX vs SIP**: free accounts must use `ALPACA_DATA_FEED=iex`. The script already defaults to IEX and passes `feed=...` in all calls.
- **Paper first**: always validate live trading logic in **paper** before switching to **live**.

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and does not constitute financial advice.  
Trading involves risk, including the loss of principal. Use **paper trading** before real funds.  
Past performance is **not** indicative of future results.
