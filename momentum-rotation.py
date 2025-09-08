import os, math, json, argparse, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Optional: Only needed for LIVE mode
try:
    from alpaca_trade_api import REST, TimeFrame
except:
    REST = None
    TimeFrame = None

# Optional: Only needed for BACKTEST mode
try:
    import yfinance as yf
except:
    yf = None

# ======== CONFIG (edit as you like) ========
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM"]  # risk-on candidates (liquid, broad)
RISK_OFF = "SHY"                                 # short-term Treasuries; fallback
TOP_N = 2                                        # hold best N each month
LOOKBACK_MONTHS = [3, 6, 12]                     # blended momentum
ABS_MOM_THRESHOLD = 0.0                          # require blended > 0, else move to RISK_OFF
REBALANCE_DAY = 1                                # rebalance on the 1st trading day after month change
WEIGHTS = "equal"                                # "equal" or "vol_target" (equal is simplest)
STATE_FILE = "rotation_state.json"               # stores last rebalance month for LIVE mode
# Position sizing for LIVE mode (paper)
MAX_PORTFOLIO_PCT_PER_TICKER = 1.0 / TOP_N       # equal split
MIN_DOLLARS_PER_ORDER = 50

# Alpaca polling
POLL_SECONDS = 60

# Backtest defaults
BT_START = "2012-01-01"
BT_END   = None  # None = today
# ===========================================

def blended_momentum(prices: pd.Series, months_list):
    """Compute blended momentum = average of N-month total returns."""
    rets = []
    for m in months_list:
        # total return over m months ≈ price_t / price_(t - m_months) - 1
        if len(prices) < (m+1):
            return np.nan
        ret = prices.iloc[-1] / prices.shift(m).dropna().iloc[-1] - 1.0
        rets.append(ret)
    return float(np.mean(rets))

def month_changed(ts_prev: pd.Timestamp, ts_now: pd.Timestamp, tz="America/New_York"):
    a = ts_prev.tz_convert(tz)
    b = ts_now.tz_convert(tz)
    return a.month != b.month or a.year != b.year

def first_trading_day_of_month(df_daily: pd.DataFrame):
    """Given a DF of daily bars, return the first index for each month."""
    idx = df_daily.index
    marker = []
    seen = set()
    for t in idx:
        key = (t.year, t.month)
        if key not in seen:
            marker.append(t)
            seen.add(key)
    return set(marker)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_rebalance_month": ""}

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

# ----------------------- BACKTEST -----------------------

def get_history_yf(tickers, start, end):
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")

def backtest_rotation(universe, risk_off, start=BT_START, end=BT_END,
                      top_n=TOP_N, months_list=LOOKBACK_MONTHS, abs_thresh=ABS_MOM_THRESHOLD,
                      weights="equal"):
    # Pull close prices (daily)
    prices = get_history_yf(universe + [risk_off], start, end)
    prices = prices.dropna()

    # Month-end markers (use last available trading day per month)
    month_ends = prices.resample("M").last().index

    # Portfolio state
    equity = 10000.0
    equity_curve = []
    positions = {}  # ticker -> shares

    # Precompute blended momentums monthly
    for i, me in enumerate(month_ends):
        # slice up to this month end
        hist = prices.loc[:me]
        if len(hist) < max(months_list)*21 + 10:  # ~21 trading days per month
            # not enough lookback yet
            equity_curve.append((me, equity))
            continue

        # compute blended momentum for all risk-on candidates
        scores = {}
        for t in universe:
            s = blended_momentum(hist[t].dropna(), months_list)
            scores[t] = s

        # pick top N that pass absolute momentum
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        picks = [t for t, s in ranked if s is not None and s > abs_thresh][:top_n]

        if len(picks) == 0:
            # move entire portfolio to risk-off
            target = [risk_off]
        else:
            target = picks

        # Compute next month's period (rebalance occurs at this month-end close -> hold to next month-end)
        if i < len(month_ends) - 1:
            nxt = month_ends[i+1]
        else:
            nxt = prices.index[-1]

        # simulate holding target from me->nxt
        p0 = prices.loc[me, target].mean() if len(target) > 1 else prices.loc[me, target[0]]
        p1 = prices.loc[nxt, target].mean() if len(target) > 1 else prices.loc[nxt, target[0]]
        ret = float(p1 / p0 - 1.0)
        equity *= (1.0 + ret)
        equity_curve.append((nxt, equity))

    curve = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    daily = curve["equity"].pct_change().dropna()
    sharpe = np.sqrt(252) * daily.mean() / (daily.std() + 1e-12)
    max_dd = (curve["equity"] / curve["equity"].cummax() - 1).min()

    stats = {
        "start": str(curve.index[0].date()) if len(curve) else start,
        "end": str(curve.index[-1].date()) if len(curve) else str(pd.Timestamp.today().date()),
        "final_equity": float(curve["equity"].iloc[-1]) if len(curve) else 10000.0,
        "return_pct": float((curve["equity"].iloc[-1] / 10000.0 - 1.0) * 100) if len(curve) else 0.0,
        "sharpe_naive": float(sharpe),
        "max_drawdown": float(max_dd),
        "top_n": top_n,
        "lookback_months": months_list,
        "abs_threshold": abs_thresh
    }
    return curve, stats

# ----------------------- LIVE (PAPER) -----------------------

def alpaca_client():
    load_dotenv()
    if REST is None:
        raise RuntimeError("alpaca-trade-api not installed. pip install alpaca-trade-api")
    return REST(
        key_id=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY"),
        base_url=os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    )

def get_daily_closes_alpaca(api, tickers, days=600, tz="America/New_York"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    frames = []
    for t in tickers:
        bars = api.get_bars(t, TimeFrame.Day, start.isoformat(), end.isoformat()).df
        if bars is None or bars.empty:
            continue
        bars = bars.tz_convert(tz)
        closes = bars["close"].rename(t)
        frames.append(closes)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).dropna(how="all")
    return df

def list_positions(api):
    pos = {}
    try:
        for p in api.list_positions():
            pos[p.symbol] = float(p.qty)
    except Exception:
        pass
    return pos

def get_equity(api):
    try:
        return float(api.get_account().equity)
    except Exception:
        return 0.0

def place_target_portfolio(api, targets, equity):
    """
    targets: dict ticker -> target dollar allocation
    Makes simple market orders to approximate targets.
    """
    for sym, dollars in targets.items():
        if dollars < MIN_DOLLARS_PER_ORDER:
            continue
        # get last price
        bar = api.get_latest_trade(sym)
        price = float(bar.price)
        qty = int(dollars // price)
        if qty <= 0:
            continue
        # Check current qty first to compute delta
        curr_qty = 0.0
        try:
            p = api.get_position(sym)
            curr_qty = float(p.qty)
        except Exception:
            curr_qty = 0.0

        target_qty = qty
        delta = target_qty - curr_qty
        if abs(delta) < 1:
            continue
        side = "buy" if delta > 0 else "sell"
        api.submit_order(symbol=sym, qty=abs(int(delta)), side=side, type="market", time_in_force="day")
        print(f"[{datetime.now()}] {side.upper()} {abs(int(delta))} {sym} @~{price:.2f}")

def liquidate_others(api, keep_symbols):
    current = list_positions(api)
    for sym, q in current.items():
        if sym not in keep_symbols and q > 0:
            try:
                api.close_position(sym)
                print(f"[{datetime.now()}] Closed position {sym}")
            except Exception as e:
                print("Close error:", sym, e)

def live_rotation_loop():
    api = alpaca_client()
    state = load_state()
    print("Starting Monthly Momentum Rotation (paper)… Ctrl+C to stop.")
    while True:
        try:
            clock = api.get_clock()
            if not clock.is_open:
                time.sleep(300)
                continue

            # Pull daily closes for universe + risk-off
            tickers = UNIVERSE + [RISK_OFF]
            closes = get_daily_closes_alpaca(api, tickers, days=800)
            if closes.empty:
                time.sleep(POLL_SECONDS); continue

            now = closes.index[-1].tz_convert("America/New_York")
            last_month_key = state.get("last_rebalance_month", "")
            this_month_key = f"{now.year}-{now.month:02d}"

            # Only rebalance on the first trading day of a new month
            first_days = first_trading_day_of_month(closes)
            is_first_trading_day = closes.index[-1] in first_days

            if this_month_key != last_month_key and is_first_trading_day:
                # compute momentum scores
                scores = {}
                for t in UNIVERSE:
                    s = blended_momentum(closes[t].dropna(), LOOKBACK_MONTHS)
                    scores[t] = s
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                picks = [t for t, s in ranked if s is not None and s > ABS_MOM_THRESHOLD][:TOP_N]
                targets = picks if len(picks) else [RISK_OFF]
                print(f"[{datetime.now()}] Rebalance: picks={picks if picks else [RISK_OFF]} scores={scores}")

                # build dollar targets
                equity = get_equity(api)
                alloc_each = equity * MAX_PORTFOLIO_PCT_PER_TICKER
                target_map = {sym: alloc_each for sym in targets}
                # Place orders and clean up others
                place_target_portfolio(api, target_map, equity)
                liquidate_others(api, set(targets))

                state["last_rebalance_month"] = this_month_key
                save_state(state)
            else:
                # nothing to do—wait
                pass

        except KeyboardInterrupt:
            print("Exiting…")
            break
        except Exception as e:
            print("Error:", e)
        time.sleep(POLL_SECONDS)

# ----------------------- CLI -----------------------

def main():
    p = argparse.ArgumentParser(description="Monthly Momentum Rotation: backtest or live (paper).")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run local backtest with yfinance.")
    bt.add_argument("--start", default=BT_START)
    bt.add_argument("--end", default=BT_END)
    bt.add_argument("--topn", type=int, default=TOP_N)
    bt.add_argument("--months", default="3,6,12", help="Comma list, e.g. 3,6,12")
    bt.add_argument("--abs", type=float, default=ABS_MOM_THRESHOLD, help="Absolute momentum threshold (e.g. 0.0)")

    live = sub.add_parser("live", help="Run Alpaca paper trading monthly rotation loop.")

    args = p.parse_args()

    if args.cmd == "backtest":
        months = [int(x.strip()) for x in args.months.split(",") if x.strip()]
        curve, stats = backtest_rotation(UNIVERSE, RISK_OFF, start=args.start, end=args.end,
                                         top_n=args.topn, months_list=months,
                                         abs_thresh=args.abs, weights=WEIGHTS)
        print("Backtest Stats:")
        for k, v in stats.items():
            print(f" - {k}: {v}")
        # Optional: write CSV
        out = "backtest_equity.csv"
        curve.to_csv(out)
        print(f"Equity curve saved -> {out}")

    elif args.cmd == "live":
        live_rotation_loop()

if __name__ == "__main__":
    main()
