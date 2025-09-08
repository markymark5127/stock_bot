# Momentum Rotation Bot (Monthly/Weekly)

Rules-based ETF rotation using blended momentum (3/6/12 months).
- **Backtest** with yfinance
- **Live (Paper)** via Alpaca
- **Runs on GitHub Actions** with `--once` (fast exit)
- Optional **SMS alerts** via Twilio

## Setup

1) Create repo; copy files.
2) Add repo **Secrets**:
   - `APCA_API_KEY_ID`
   - `APCA_API_SECRET_KEY`
   - *(Optional SMS)* `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `TWILIO_TO`

3) Commit & push. Workflow runs on schedule.

## Local Backtest
```bash
pip install -r requirements.txt
python momentum_rotation.py backtest --start 2012-01-01 --freq monthly
python momentum_rotation.py backtest --start 2012-01-01 --freq weekly
