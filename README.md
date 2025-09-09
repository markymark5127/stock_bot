# 📈 Momentum Rotation Trading Bot

A sophisticated Python-based **ETF momentum rotation strategy** that dynamically rotates into the **strongest-trending ETFs** based on multi-timeframe momentum analysis. The algorithm features **intelligent regime detection**, **adaptive risk management**, and **comprehensive benchmark analysis** to deliver superior risk-adjusted returns.

## 🎯 Key Features

- ✅ **Multi-Timeframe Momentum Analysis** with adaptive skip-month logic
- ✅ **Intelligent Auto Profile Switching** based on market regime detection
- ✅ **Volatility-Based Position Sizing** for dynamic risk management
- ✅ **Risk-Off Basket Optimization** with best-of selection
- ✅ **Comprehensive Benchmark Analysis** (Alpha, Beta, Sharpe, etc.)
- ✅ **Backtesting & Live Trading** (Yahoo Finance / Alpaca)
- ✅ **Parameter Optimization** with threshold sweeps
- ✅ **Intra-Period Risk Management** with stop-losses
- ✅ **Optional SMS Alerts** (Twilio integration)
- ✅ **GitHub Actions Automation**

---

## 🧠 How the Algorithm Works

### Core Philosophy

The momentum rotation strategy is based on the principle that **"trends persist"** - assets that have performed well recently are likely to continue performing well in the near term. The algorithm systematically identifies the strongest momentum assets and rotates into them while maintaining strict risk controls.

### Algorithm Flow

#### 1. **Momentum Calculation** (`blended_momentum` function)

The heart of the strategy is the **multi-timeframe momentum calculation**:

```python
# Example: For a 3-month lookback with skip-month enabled
# Instead of measuring 3 months ago → now
# We measure 4 months ago → 1 month ago (skipping the most recent month)
```

**Key Features:**
- **Multi-horizon averaging**: Combines 1, 3, 6, and 12-month returns (profile-dependent)
- **Adaptive skip-month logic**: 
  - `"auto"` mode: Skips recent month only when price < SMA200 (bearish regime)
  - `True`: Always skips most recent month (12-1 momentum)
  - `False`: Uses standard momentum (includes recent month)
- **Robust data handling**: Handles missing data and edge cases gracefully

#### 2. **Asset Selection Process**

```python
# Step 1: Calculate momentum scores for all assets in universe
scores = {ticker: blended_momentum(prices[ticker], lookback_months) for ticker in universe}

# Step 2: Apply filters
qualified = []
for ticker, score in ranked_scores:
    if score > absolute_threshold:  # Must be positive momentum
        if ticker != "SPY" and score > spy_momentum:  # Must beat SPY
            qualified.append((ticker, score))

# Step 3: Select top N assets
picks = [ticker for ticker, score in qualified[:TOP_N]]
```

#### 3. **Regime Detection & Auto Profile Switching**

The algorithm intelligently switches between risk profiles based on market conditions:

```python
def choose_profile_auto3(spy_data, prev_profile):
    # Market regime indicators:
    spy_vs_sma200 = current_spy / sma200_spy - 1.0
    spy_1m_momentum = month_return(spy, 1)
    spy_3m_momentum = month_return(spy, 3)
    sma100_trend = sma100_current > sma100_week_ago
    
    # Tech sector overrides
    qqq_3m_momentum = month_return(qqq, 3)
    tech_override = qqq_3m_momentum >= 0.10 or tqqq_3m_momentum >= 0.20
    
    # Decision logic:
    if bear_market_conditions:
        return "conservative"
    elif tech_override or (spy_strong_trend and positive_momentum):
        return "aggressive"
    elif moderate_conditions:
        return "balanced"
    else:
        return "conservative"
```

#### 4. **Risk Management & Position Sizing**

**Volatility Targeting:**
```python
# Calculate recent volatility of selected assets
recent_volatility = calculate_20_day_volatility(selected_assets)

# Adjust position size based on volatility
if recent_volatility < target_volatility:
    leverage = min(target_volatility / recent_volatility, max_leverage)
else:
    leverage = 1.0  # No leverage in high volatility
```

**Risk-Off Basket Selection:**
```python
# Choose best risk-off asset from basket
risk_off_candidates = ["BIL", "SHY", "IEF", profile_risk_off]
best_risk_off = max(risk_off_candidates, key=lambda x: month_return(x, 3))
```

#### 5. **Rebalancing Logic**

- **Weekly**: First trading day of each week
- **Monthly**: First trading day of each month
- **State persistence**: Prevents multiple rebalances per period
- **Transaction costs**: Accounts for trading costs in backtests

---

## 🧪 Trading Profiles

| Profile | Universe | Risk-Off | Top N | Lookbacks | Stops | Risk Level |
|---------|----------|----------|-------|-----------|-------|------------|
| **Conservative** | SPY, QQQ, IWM, EFA, EEM | SHY | 2 | [3,6,12] | None | Low |
| **Balanced** | QQQ, XLK, SPY, ARKK, IWM | GLD | 2 | [1,3,6] | -10%/-15% | Medium |
| **Aggressive** | QQQ, TQQQ, SOXL, ARKK, SPY | GLD | 1 | [1,3] | -12%/-20% | High |

### Profile Selection Logic

**Conservative**: Used in bear markets or when SPY is significantly below SMA200
**Balanced**: Default choice for moderate market conditions
**Aggressive**: Activated during strong bull markets with tech sector leadership

### Auto Profile Switching

The `auto` profile intelligently switches between the three profiles based on:

- **SPY vs SMA200 position**: Market trend strength
- **Short-term momentum**: 1-month and 3-month returns
- **SMA100 trend direction**: Medium-term trend confirmation
- **Tech sector overrides**: QQQ/TQQQ/SOXL momentum thresholds
- **QQQ trend confirmation**: Tech sector leadership
- **Hysteresis**: Prevents rapid switching with configurable buffer zones

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

The script has **three modes**: `backtest`, `live`, and `sweep`. There are also **global flags** for enhanced features.

### Global Flags

- `--profile {conservative|balanced|aggressive|auto}`  
  Selects a preset universe, lookbacks, stops, etc.  
  **Default:** `balanced`.  
  **Auto mode**: Intelligently switches between `conservative`, `balanced`, and `aggressive` based on market conditions.

### Backtest mode

```bash
python momentum_rotation.py --profile balanced backtest [flags...]
```

**Flags:**
- `--start YYYY-MM-DD`  
  Start date for historical data (**required** for meaningful tests).  
  **Default:** `2012-01-01`
- `--end YYYY-MM-DD`  
  End date (optional; defaults to latest).
- `--freq {weekly|monthly}`  
  Rebalance cadence for the simulation.  
  **Default:** `weekly`
- `--topn N`  
  **Override** the profile's `TOP_N` (e.g., try `--topn 1` vs `--topn 2`).
- `--months a,b,c`  
  **Override** lookbacks (e.g., `--months 1,3,6`).
- `--abs X`  
  **Override** momentum threshold (e.g., `--abs 0.0` means only positive momentum is eligible).

**Enhanced Outputs:**
- Console stats with **comprehensive S&P 500 benchmark comparison**:
  - Strategy vs SPY total returns
  - **Alpha**: Risk-adjusted excess return
  - **Beta**: Market sensitivity
  - **Information Ratio**: Risk-adjusted excess return per unit of tracking error
  - **Tracking Error**: Volatility of excess returns
  - **Win Rate**: Percentage of periods outperforming SPY
  - **Sharpe Ratios**: Both strategy and benchmark
  - **Annualized returns and volatility** for both strategy and benchmark
- `equity_weekly.csv` or `equity_monthly.csv` with the equity curve
- For auto profile: Profile choice tracking and debug information

### Live mode

```bash
python momentum_rotation.py --profile balanced live [flags...]
```

**Flags:**
- `--freq {weekly|monthly}`  
  Live rebalance schedule.  
  **Default:** `weekly`
- `--once`  
  Run **one pass** and exit (ideal for GitHub Actions).
- `--force`  
  Run even if the market is closed (testing only; **orders may not fill**).
- `--after-hours`  
  If market is closed, the bot places **LIMIT** orders at the last price and sets `extended_hours=True` so they can queue for AH sessions.

**Notes:**
- Live mode uses:
  - **Alpaca IEX** data (`ALPACA_DATA_FEED=iex`) for free historical/latest bars.
  - A local `rotation_state.json` to remember the last period key, entries, equity, and last chosen profile.
- The script will **not rebalance more than once per period** thanks to `rotation_state.json`.  
  In CI (GitHub Actions), the workflow **caches** this file to persist across runs.
- **Auto profile mode** maintains profile choice stability using hysteresis to prevent rapid switching.
- **Intra-period risk management** checks for stop-losses on each run (if enabled by profile).

### Sweep mode

```bash
python momentum_rotation.py --profile auto sweep [flags...]
```

**Flags:**
- `--start YYYY-MM-DD`  
  Start date for historical data.  
  **Default:** `2012-01-01`
- `--end YYYY-MM-DD`  
  End date (optional; defaults to latest).
- `--freq {weekly|monthly}`  
  Rebalance cadence for the simulation.  
  **Default:** `monthly`
- `--vals "a,b,c"`  
  Comma-separated list of TREND_STRONG values to test (e.g., `"0,0.02,0.03,0.05"`).  
  **Default:** `"0,0.02,0.03,0.05"`

**Outputs:**
- Console table with performance metrics for each TREND_STRONG value
- Profile choice counts for each threshold (debug information)
- `sweep_trend_strong_{freq}.csv` with detailed results including benchmark metrics
- **Isolated testing**: Disables SMA100 shortcuts and tech overrides to isolate TREND_STRONG effects

---

## 📊 Detailed Example Runs

### Example 1: Weekly Rebalance (Balanced Profile)

```bash
python momentum_rotation.py --profile balanced backtest --start 2023-01-01 --end 2023-12-31 --freq weekly
```

**Sample Output:**
```
Backtest Stats:
 - start: 2023-01-03
 - end: 2023-12-29
 - freq: weekly
 - profile: balanced
 - final_equity: 12450.32
 - return_pct: 24.50
 - sharpe_naive: 1.23
 - max_drawdown: -0.08
 - spy_final_equity: 11800.45
 - spy_return_pct: 18.00
 - alpha_pct: 6.50
```

**What happened:**
1. **Week 1**: Selected QQQ, XLK (strong tech momentum)
2. **Week 2**: Rotated to SPY, ARKK (momentum shift)
3. **Week 3**: Moved to risk-off (GLD) due to market weakness
4. **Week 4**: Back to QQQ, XLK as momentum returned

### Example 2: Auto Profile with Regime Detection

```bash
python momentum_rotation.py --profile auto backtest --start 2022-01-01 --end 2023-12-31 --freq monthly
```

**Sample Regime Changes:**
```
[AUTO] SPY last=415.23 SMA200=398.45 r1=0.023 r3=0.045 → profile=aggressive
[AUTO] SPY last=385.67 SMA200=401.23 r1=-0.015 r3=-0.032 → profile=conservative
[AUTO] SPY last=420.15 SMA200=405.67 r1=0.018 r3=0.028 → profile=balanced
```

### Example 3: Parameter Optimization

```bash
python momentum_rotation.py --profile auto sweep --start 2020-01-01 --freq monthly --vals "0,0.01,0.02,0.03,0.05"
```

**Sample Results:**
```
=== Threshold Sweep (TREND_STRONG) ===
trend_strong  return_pct  sharpe_naive  max_drawdown  alpha_pct
0.0200       18.45       1.34         -0.12         4.23
0.0100       17.89       1.28         -0.14         3.67
0.0300       16.23       1.19         -0.16         2.01
0.0000       15.67       1.15         -0.18         1.45
0.0500       14.89       1.08         -0.20         0.67
```

---

## 📊 Examples

### Enhanced Backtests with S&P 500 Focus

```bash
# Test the balanced profile (default: enhanced momentum + volatility sizing)
python momentum_rotation.py --profile balanced backtest --start 2020-01-01 --freq monthly

# Test auto profile switching (default: enhanced momentum + volatility sizing)
python momentum_rotation.py --profile auto backtest --start 2020-01-01 --freq monthly

# Test different parameter overrides
python momentum_rotation.py --profile balanced backtest --start 2020-01-01 --freq monthly --topn 1 --months 1,3
```

### Parameter Optimization

```bash
# Sweep TREND_STRONG values to find optimal settings (isolated testing)
python momentum_rotation.py --profile auto sweep --start 2020-01-01 --freq monthly --vals "0,0.02,0.03,0.05"

# Compare different profiles
python momentum_rotation.py --profile conservative backtest --start 2020-01-01 --freq monthly
python momentum_rotation.py --profile balanced backtest --start 2020-01-01 --freq monthly
python momentum_rotation.py --profile aggressive backtest --start 2020-01-01 --freq monthly
```

### Traditional Backtests
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
# Weekly, one-shot (run manually or via Actions) - default: enhanced momentum + volatility sizing
python momentum_rotation.py --profile balanced live --freq weekly --once

# Monthly, continuous loop (checks every 60s; exits immediately if no rebalance day)
python momentum_rotation.py --profile aggressive live --freq monthly

# Auto profile switching (default: enhanced momentum + volatility sizing)
python momentum_rotation.py --profile auto live --freq monthly --once

# Force run even if market is closed (testing only)
python momentum_rotation.py --profile balanced live --freq weekly --once --force

# Use after-hours orders if market is closed
python momentum_rotation.py --profile balanced live --freq weekly --once --after-hours
```

> For **paper vs live trading**: use the appropriate Alpaca endpoint in `APCA_API_BASE_URL`.  
> Paper: `https://paper-api.alpaca.markets` (recommended!)  
> Live:  `https://api.alpaca.markets` (real money)

---

## 📈 Advanced Features Explained

### 1. Adaptive Skip-Month Logic

The algorithm can intelligently skip the most recent month of data:

```python
SKIP_MONTH = "auto"  # Options: True, False, "auto"

# Auto mode logic:
if price < sma200:  # Bearish regime
    skip_recent_month = True   # Use 12-1 momentum
else:  # Bullish regime
    skip_recent_month = False  # Use standard momentum
```

**Why this matters:**
- **Bear markets**: Recent month may contain noise, 12-1 momentum is more stable
- **Bull markets**: Recent momentum is more predictive, include it

### 2. Volatility-Based Position Sizing

```python
VOL_TARGET_ANNUAL = 0.16      # Target 16% annual volatility
VOL_LOOKBACK_DAYS = 20        # Use 20-day lookback
VOL_MAX_LEVERAGE = 2.0        # Maximum 2x leverage

# Position sizing logic:
if recent_volatility < target_volatility:
    position_size = min(target_volatility / recent_volatility, max_leverage)
else:
    position_size = 1.0  # No leverage in high volatility
```

### 3. Risk-Off Basket Optimization

Instead of using a single risk-off asset, the algorithm selects the best-performing one:

```python
RISK_OFF_BASKET = ["BIL", "SHY", "IEF"]  # Treasury ETFs

# Selection logic:
best_risk_off = max(risk_off_candidates, key=lambda x: month_return(x, 3))
```

### 4. Transaction Cost Modeling

```python
COST_BPS = 5  # 5 basis points per trade

# Cost calculation:
if portfolio_changed:
    trades = len(set(old_positions) ^ set(new_positions))
    cost = trades * 2 * COST_BPS / 10000  # Round-trip cost
    period_return -= cost
```

---

## 🎯 Key Features

### 1. **Enhanced Momentum Calculation**
- **Volatility-adjusted returns**: Higher Sharpe-like scores for lower volatility assets
- **Recent momentum weighting**: 60% weight on recent (1-3 month) momentum, 40% on longer-term
- **Multi-timeframe analysis**: Combines multiple lookback periods intelligently
- **Default enabled**: Enhanced momentum is the default calculation method

### 2. **Advanced Risk Management**
- **Volatility-based position sizing**: Reduces position sizes during high volatility periods (default enabled)
- **Intra-period stop-losses**: Per-position and portfolio-level stops (enabled by profile)
- **Trend-following stops**: Uses moving average crossovers to detect trend reversals
- **Enhanced stop-loss logic**: More sophisticated exit criteria

### 3. **Comprehensive Benchmark Analysis**
- **Alpha**: Risk-adjusted excess return vs SPY
- **Beta**: Market sensitivity measurement
- **Information Ratio**: Risk-adjusted excess return per unit of tracking error
- **Tracking Error**: Volatility of excess returns
- **Win Rate**: Percentage of periods outperforming SPY

### 4. **Auto Profile Switching**
- Intelligently switches between `conservative`, `balanced`, and `aggressive` based on:
  - SPY vs SMA200 position (configurable thresholds)
  - Short-term and medium-term momentum
  - SMA100 trend direction
  - Tech sector momentum overrides (QQQ/TQQQ/SOXL)
  - QQQ trend confirmation
  - **Hysteresis**: Prevents rapid switching with configurable buffer zones
  - **State persistence**: Remembers last chosen profile for stability

### 5. **Risk-Off Basket Optimization**
- **Best-of selection**: Automatically chooses the best-performing risk-off asset from a basket
- **Default basket**: `["BIL", "SHY", "IEF"]` with profile-specific overrides
- **Dynamic selection**: Re-evaluates risk-off assets at each rebalance

---

## 🤝 GitHub Actions (automation)

Example workflow (`.github/workflows/run.yml`) that runs **every weekday** at 10:35 ET with enhanced features:

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
      - name: Run rotation (auto profile with default enhanced features)
        run: python momentum_rotation.py --profile auto live --freq weekly --once
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

## 🧠 Tips for Better Performance

- **Use `balanced` profile** for good risk-adjusted returns
- **Enhanced momentum is default** - no need to specify additional flags
- **Volatility sizing is default** - no need to specify additional flags
- **Test different TREND_STRONG values** with the sweep function to find optimal settings
- **Monitor benchmark metrics** in backtest output to track S&P 500 performance
- **Consider auto profile** for adaptive risk management based on market conditions

### Parameter Optimization Workflow

1. **Start with balanced profile**:
   ```bash
   python momentum_rotation.py --profile balanced backtest --start 2020-01-01 --freq monthly
   ```

2. **Test with default enhanced features** (no flags needed):
   ```bash
   python momentum_rotation.py --profile balanced backtest --start 2020-01-01 --freq monthly
   ```

3. **Optimize TREND_STRONG for auto profile**:
   ```bash
   python momentum_rotation.py --profile auto sweep --start 2020-01-01 --freq monthly --vals "0,0.02,0.03,0.05"
   ```

4. **Run confirmatory backtests** with optimal settings:
   ```bash
   python momentum_rotation.py --profile auto backtest --start 2020-01-01 --freq monthly
   ```

5. **Compare profiles** for validation:
   ```bash
   python momentum_rotation.py --profile conservative backtest --start 2020-01-01 --freq monthly
   python momentum_rotation.py --profile aggressive backtest --start 2020-01-01 --freq monthly
   ```

---

## 🧠 General Tips

- **First run** after changing profiles/params: consider deleting `rotation_state.json` so the new schedule/period key starts fresh.
- **IEX vs SIP**: free accounts must use `ALPACA_DATA_FEED=iex`. The script already defaults to IEX and passes `feed=...` in all calls.
- **Paper first**: always validate live trading logic in **paper** before switching to **live**.
- **Monitor benchmark metrics**: Pay attention to Alpha, Beta, and Information Ratio to ensure the strategy is adding value vs SPY.

---

## 📚 Understanding the Output

### Backtest Output

```
Backtest Stats:
 - start: 2023-01-03          # Backtest start date
 - end: 2023-12-29            # Backtest end date
 - freq: weekly               # Rebalancing frequency
 - profile: balanced          # Trading profile used
 - final_equity: 12450.32     # Final portfolio value
 - return_pct: 24.50          # Total return percentage
 - sharpe_naive: 1.23         # Sharpe ratio
 - max_drawdown: -0.08        # Maximum drawdown
 - spy_final_equity: 11800.45 # SPY buy-and-hold value
 - spy_return_pct: 18.00      # SPY return percentage
 - alpha_pct: 6.50            # Excess return vs SPY
```

### Live Trading Output

```
[AUTO] SPY last=415.23 SMA200=398.45 r1=0.023 r3=0.045 → profile=aggressive
[2024-01-15 10:35:00] Rebalance weekly [aggressive]: picks=['QQQ', 'TQQQ'] scores={QQQ: 0.0234, TQQQ: 0.0456, SPY: 0.0189, XLK: 0.0123, ARKK: 0.0089}
[2024-01-15 10:35:01] BUY 15 QQQ @~415.23 (market)
[2024-01-15 10:35:02] BUY 8 TQQQ @~45.67 (market)
```

---

## 🆘 Troubleshooting

### Common Issues

**"No data. Exiting."**
- Check internet connection
- Verify ticker symbols are correct
- Ensure date range has trading days

**"Market closed. Exiting."**
- Use `--force` flag for testing
- Use `--after-hours` for extended hours trading
- Check market hours and holidays

**"alpaca-trade-api not installed"**
```bash
pip install alpaca-trade-api
```

**"yfinance not installed"**
```bash
pip install yfinance
```

### Getting Help

1. Check the logs for error messages
2. Verify your `.env` configuration
3. Test with paper trading first
4. Review the algorithm logic in the code
5. Check market data availability

---

## ⚠️ Risk Warnings & Best Practices

### Important Considerations

1. **Past Performance**: Historical results don't guarantee future performance
2. **Market Risk**: All trading involves risk of loss
3. **Leverage Risk**: Aggressive profiles use leveraged ETFs (TQQQ, SOXL)
4. **Transaction Costs**: Real trading costs may be higher than modeled
5. **Data Quality**: Strategy depends on accurate price data

### Best Practices

1. **Start with Paper Trading**: Always test with paper money first
2. **Use Conservative Profiles**: Begin with lower-risk profiles
3. **Monitor Performance**: Regularly review strategy performance
4. **Diversify**: Don't put all capital in one strategy
5. **Stay Informed**: Keep up with market conditions and strategy changes

### Recommended Workflow

1. **Backtest**: Test strategy over multiple time periods
2. **Paper Trade**: Run live with paper money for 1-3 months
3. **Small Allocation**: Start with small real money allocation
4. **Scale Up**: Gradually increase allocation as confidence grows
5. **Monitor & Adjust**: Continuously monitor and adjust parameters

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice, and trading involves substantial risk of loss. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making investment decisions.

**Use at your own risk. The authors are not responsible for any financial losses.**