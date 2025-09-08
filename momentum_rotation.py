import os, json, argparse, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Optional (LIVE only)
try:
    from alpaca_trade_api import REST, TimeFrame
except Exception:
    REST = None
# Optional (BACKTEST only)
try:
    import yfinance as yf
except Exception:
    yf = None

# ======== CONFIG (Balanced-but-spicy) ========
# Growth tilt with some breadth; GLD as risk-off to keep some upside when stocks are weak
UNIVERSE     = ["QQQ", "XLK", "SPY", "ARKK", "IWM"]  # high-quality growth + broad + innovation + small caps
RISK_OFF     = "GLD"                                # less defensive than SHY; tends to help in risk-off regimes
TOP_N        = 2                                     # diversify across top 2 winners
LOOKBACK_M   = [1, 3, 6]                             # faster momentum; reacts quicker to reversals
ABS_THRESH   = 0.0                                   # require blended momentum > 0 to be risk-on
STATE_FILE   = "rotation_state.json"
POLL_SECONDS = 60
MIN_DOLLARS_PER_ORDER = 50
MAX_PORTFOLIO_PCT_PER_TICKER = 1.0 / TOP_N
TZ = "America/New_York"

# Alpaca feed (free default = iex)
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()

# Optional intra-period protection (ENABLED for this profile)
INTRA_MONTH_STOP_PCT = -0.10     # per-position stop: cut if −10% from period entry
PORTFOLIO_DD_STOP_PCT = -0.15    # portfolio circuit breaker: move to GLD if equity −15% from period start

# Backtest defaults
BT_START = "2012-01-01"
BT_END   = None  # today
# ============================================

# ---------- Utilities / State ----------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_rebalance_key": "", "month_entries": {}, "month_start_equity": None}

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

def blended_momentum(prices: pd.Series, months_list):
    rets = []
    for m in months_list:
        if len(prices) < 60:  # guard
            return np.nan
        target_time = prices.index[-1] - pd.DateOffset(months=m)
        past_idx = prices.index.searchsorted(target_time)
        if past_idx <= 0 or past_idx >= len(prices):
            return np.nan
        p0 = prices.iloc[past_idx]
        p1 = prices.iloc[-1]
        if p0 <= 0:
            return np.nan
        rets.append(float(p1 / p0 - 1.0))
    return float(np.nanmean(rets))

# ---------- BACKTEST ----------
def get_history_yf(tickers, start, end):
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")

def period_end_indices(prices: pd.DataFrame, freq: str):
    """Return target period-end timestamps using resample ('M' or 'W-FRI')."""
    return prices.resample(freq).last().index

def rotation_backtest(universe, risk_off, start, end, top_n, months_list, abs_thresh, freq):
    prices = get_history_yf(universe + [risk_off], start, end).dropna()
    prices = prices.sort_index()
    ends = period_end_indices(prices, "M" if freq == "monthly" else "W-FRI")

    # Helper: last available trading day <= ts
    def last_trading_on_or_before(ts: pd.Timestamp):
        idx = prices.index[prices.index <= ts]
        return idx.max() if len(idx) else None

    # Helper: get level at ts* (on/before), averaged if multiple tickers
    def level_at(ts: pd.Timestamp, tickers):
        ts2 = last_trading_on_or_before(ts)
        if ts2 is None:
            return np.nan
        if isinstance(tickers, (list, tuple)):
            s = prices.loc[ts2, tickers].dropna()
            return float(s.mean()) if len(s) else np.nan
        else:
            return float(prices.loc[ts2, tickers])

    equity = 10000.0
    curve = []

    # Iterate over rebalance points
    for i, t_end in enumerate(ends):
        # Use actual trading day for the period end
        t0 = last_trading_on_or_before(t_end)
        if t0 is None:
            continue

        hist = prices.loc[:t0]
        # ensure enough history
        if len(hist) < 250:
            curve.append((t0, equity))
            continue

        # Compute blended momentum scores
        scores = {t: blended_momentum(hist[t].dropna(), months_list) for t in universe}
        ranked = sorted(scores.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -9e9), reverse=True)
        picks = [t for t, s in ranked if s == s and s > abs_thresh][:top_n]
        targets = picks if picks else [risk_off]

        # Next period end (or last available data)
        t_next_nominal = ends[i + 1] if i < len(ends) - 1 else prices.index[-1]
        t1 = last_trading_on_or_before(t_next_nominal)
        if t1 is None or t1 <= t0:
            curve.append((t0, equity))
            continue

        p0 = level_at(t0, targets)
        p1 = level_at(t1, targets)
        if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
            curve.append((t1, equity))
            continue

        ret = float(p1 / p0 - 1.0)
        equity *= (1.0 + ret)
        curve.append((t1, equity))

    curve = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date")
    daily = curve["equity"].pct_change().dropna()
    sharpe = float(np.sqrt(252) * (daily.mean() / (daily.std() + 1e-12))) if len(daily) else 0.0
    max_dd = float((curve["equity"] / curve["equity"].cummax() - 1).min()) if len(curve) else 0.0
    stats = {
        "start": str(curve.index[0].date()) if len(curve) else start,
        "end":   str(curve.index[-1].date()) if len(curve) else (end or str(pd.Timestamp.today().date())),
        "final_equity": float(curve["equity"].iloc[-1]) if len(curve) else 10000.0,
        "return_pct": float((curve["equity"].iloc[-1] / 10000.0 - 1.0) * 100) if len(curve) else 0.0,
        "sharpe_naive": sharpe,
        "max_drawdown": max_dd,
        "freq": freq,
        "top_n": top_n,
        "lookback_months": months_list,
        "abs_threshold": abs_thresh,
    }
    return curve, stats

# ---------- LIVE ----------
def alpaca_client():
    load_dotenv()
    if REST is None:
        raise RuntimeError("alpaca-trade-api not installed. pip install alpaca-trade-api")
    return REST(
        key_id=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY"),
        base_url=os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    )

def get_daily_closes_alpaca(api, tickers, days=800):
    end = datetime.now(timezone.utc); start = end - timedelta(days=days)
    frames = []
    for t in tickers:
        bars = api.get_bars(
            t,
            TimeFrame.Day,
            start.isoformat(),
            end.isoformat(),
            feed=ALPACA_DATA_FEED
        ).df
        if bars is None or bars.empty: continue
        bars = bars.tz_convert(TZ)
        closes = bars["close"].rename(t)
        frames.append(closes)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")

def is_first_trading_of_month(index: pd.DatetimeIndex):
    ft, seen = [], set()
    for t in index:
        key = (t.year, t.month)
        if key not in seen:
            ft.append(t); seen.add(key)
    return set(ft)

def is_first_trading_of_week(index: pd.DatetimeIndex):
    ft, seen = [], set()
    for t in index:
        key = (t.isocalendar().year, int(t.isocalendar().week))
        if key not in seen:
            ft.append(t); seen.add(key)
    return set(ft)

def list_positions(api):
    pos = {}
    try:
        for p in api.list_positions():
            pos[p.symbol] = float(p.qty)
    except Exception:
        pass
    return pos

def get_equity(api):
    try: return float(api.get_account().equity)
    except Exception: return 0.0

def latest_price(api, sym):
    return float(api.get_latest_trade(sym, feed=ALPACA_DATA_FEED).price)

def place_target_portfolio(api, targets, equity, allow_after_hours=False):
    """
    Place orders to roughly reach target dollar allocations.
    If allow_after_hours==True (market closed), use LIMIT orders at last price with extended_hours=True.
    Otherwise, use MARKET orders (RTH).
    """
    for sym, dollars in targets.items():
        if dollars < MIN_DOLLARS_PER_ORDER: continue
        last = latest_price(api, sym)
        qty_target = int(dollars // last)
        if qty_target <= 0: continue

        curr_qty = 0.0
        try: curr_qty = float(api.get_position(sym).qty)
        except Exception: pass

        delta = qty_target - curr_qty
        if abs(delta) < 1: continue

        side = "buy" if delta > 0 else "sell"
        order_kwargs = {
            "symbol": sym,
            "qty": abs(int(delta)),
            "side": side,
            "time_in_force": "day",
        }

        if allow_after_hours:
            order_kwargs.update({
                "type": "limit",
                "limit_price": round(last, 2),
                "extended_hours": True,
            })
        else:
            order_kwargs.update({"type": "market"})

        api.submit_order(**order_kwargs)
        print(f"[{datetime.now()}] {side.upper()} {abs(int(delta))} {sym} @~{last:.2f} ({'AH limit' if allow_after_hours else 'market'})")

def liquidate_others(api, keep_symbols, allow_after_hours=False):
    held = list_positions(api)
    for sym, q in held.items():
        if sym not in keep_symbols and q > 0:
            try:
                if allow_after_hours:
                    last = latest_price(api, sym)
                    api.submit_order(
                        symbol=sym, qty=int(q), side="sell",
                        type="limit", limit_price=round(last, 2),
                        time_in_force="day", extended_hours=True
                    )
                else:
                    api.close_position(sym)
                print(f"[{datetime.now()}] Closed {sym} ({'AH' if allow_after_hours else 'RTH'})")
            except Exception as e:
                print("Close error:", sym, e)

def compute_picks(closes: pd.DataFrame):
    scores = {t: blended_momentum(closes[t].dropna(), LOOKBACK_M) for t in UNIVERSE}
    ranked = sorted(scores.items(), key=lambda kv: (kv[1] if kv[1]==kv[1] else -9e9), reverse=True)
    picks = [t for t,s in ranked if s==s and s > ABS_THRESH][:TOP_N]
    return (picks if picks else [RISK_OFF]), scores

def make_rebalance_key(now: pd.Timestamp, freq: str):
    if freq == "monthly": return f"M-{now.year}-{now.month:02d}"
    iso = now.isocalendar(); return f"W-{iso.year}-{int(iso.week):02d}"

# ---------- SMS (Twilio optional) ----------
def send_sms(msg: str):
    try:
        from twilio.rest import Client
        load_dotenv()
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        tok = os.getenv("TWILIO_AUTH_TOKEN")
        frm = os.getenv("TWILIO_FROM")
        to  = os.getenv("TWILIO_TO")
        if not (sid and tok and frm and to):
            return
        Client(sid, tok).messages.create(body=msg[:1500], from_=frm, to=to)
    except Exception as e:
        print("SMS error:", e)

# ---------- Live once ----------
def live_once(freq: str, force: bool=False, after_hours: bool=False):
    api   = alpaca_client()
    state = load_state()

    clock = api.get_clock()
    try:
        print(f"Alpaca clock: is_open={clock.is_open} now={clock.timestamp} next_open={clock.next_open} next_close={clock.next_close}")
    except Exception:
        print(f"Alpaca clock: is_open={getattr(clock, 'is_open', None)}")

    market_open = bool(getattr(clock, "is_open", False))
    if (not market_open) and (not force) and (not after_hours):
        print("Market closed. Exiting (use --force or --after-hours to bypass).")
        return

    closes = get_daily_closes_alpaca(api, UNIVERSE + [RISK_OFF], days=800)
    if closes.empty:
        print("No data. Exiting.")
        return

    now = closes.index[-1]
    firsts = is_first_trading_of_month(closes.index) if freq=="monthly" else is_first_trading_of_week(closes.index)
    key_now = make_rebalance_key(now, freq)

    # Intra-period risk checks (enabled)
    if (INTRA_MONTH_STOP_PCT or PORTFOLIO_DD_STOP_PCT):
        try:
            if PORTFOLIO_DD_STOP_PCT and state.get("month_start_equity"):
                eq0 = state["month_start_equity"]
                eq  = get_equity(api)
                if eq0 and eq:
                    dd = (eq - eq0) / eq0
                    if dd <= PORTFOLIO_DD_STOP_PCT:
                        liquidate_others(api, keep_symbols=set([RISK_OFF]), allow_after_hours=(after_hours and not market_open))
                        equity = get_equity(api)
                        place_target_portfolio(api, {RISK_OFF: equity}, equity, allow_after_hours=(after_hours and not market_open))
                        send_sms(f"PORTFOLIO STOP triggered ({dd:.2%}). Moved to {RISK_OFF}.")

            if INTRA_MONTH_STOP_PCT and state.get("month_entries"):
                held = list_positions(api)
                for sym, qty in held.items():
                    if sym == RISK_OFF or qty <= 0: continue
                    entry = state["month_entries"].get(sym)
                    if not entry: continue
                    last = latest_price(api, sym)
                    pnl = (last - entry) / entry
                    if pnl <= INTRA_MONTH_STOP_PCT:
                        try:
                            if after_hours and not market_open:
                                api.submit_order(
                                    symbol=sym, qty=int(qty), side="sell",
                                    type="limit", limit_price=round(last, 2),
                                    time_in_force="day", extended_hours=True
                                )
                            else:
                                api.close_position(sym)
                            send_sms(f"STOP {sym} at {pnl:.2%} → shifting to {RISK_OFF}")
                        except Exception as e:
                            print("Stop close error:", e)
        except Exception as e:
            print("Intra-period check error:", e)

    # Rebalance only on first trading day of the period and only once per period
    if (now in firsts) and (state.get("last_rebalance_key") != key_now):
        targets, scores = compute_picks(closes)
        print(f"[{datetime.now()}] Rebalance {freq}: picks={targets}  scores={ {k: (None if v!=v else round(v,4)) for k,v in scores.items()} }")
        send_sms(f"Rotation {freq} rebalance -> {targets}")

        equity = get_equity(api)
        alloc_each = equity * MAX_PORTFOLIO_PCT_PER_TICKER
        ah_allowed = (after_hours and not market_open)
        place_target_portfolio(api, {sym: alloc_each for sym in targets}, equity, allow_after_hours=ah_allowed)
        liquidate_others(api, set(targets), allow_after_hours=ah_allowed)

        # record entries for per-position stops
        try:
            state["month_entries"] = {sym: latest_price(api, sym) for sym in targets}
        except Exception:
            state["month_entries"] = {}
        state["month_start_equity"] = equity
        state["last_rebalance_key"] = key_now
        save_state(state)
    else:
        print(f"No {freq} rebalance needed today. Exiting.")

# ---------- Loop ----------
def live_loop(freq: str, force: bool=False, after_hours: bool=False):
    while True:
        try:
            live_once(freq, force=force, after_hours=after_hours)
        except Exception as e:
            print("Error:", e)
        time.sleep(POLL_SECONDS)

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Momentum Rotation (monthly or weekly): backtest or live.")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run local backtest with yfinance.")
    bt.add_argument("--start", default=BT_START)
    bt.add_argument("--end", default=BT_END)
    bt.add_argument("--topn", type=int, default=TOP_N)
    bt.add_argument("--months", default="1,3,6")
    bt.add_argument("--abs", type=float, default=ABS_THRESH)
    bt.add_argument("--freq", choices=["monthly","weekly"], default="weekly")

    live = sub.add_parser("live", help="Run Alpaca paper/live.")
    live.add_argument("--freq", choices=["monthly","weekly"], default="weekly")
    live.add_argument("--once", action="store_true", help="Run a single pass (ideal for Actions)")
    live.add_argument("--force", action="store_true", help="Run even if market is closed (for testing)")
    live.add_argument("--after-hours", action="store_true",
                      help="If market is closed, queue LIMIT orders eligible for extended hours")

    args = p.parse_args()

    if args.cmd == "backtest":
        months = [int(x.strip()) for x in args.months.split(",") if x.strip()]
        curve, stats = rotation_backtest(UNIVERSE, RISK_OFF,
                                         start=args.start, end=args.end,
                                         top_n=args.topn, months_list=months,
                                         abs_thresh=args.abs, freq=args.freq)
        print("Backtest Stats:")
        for k,v in stats.items(): print(f" - {k}: {v}")
        out = f"equity_{args.freq}.csv"
        curve.to_csv(out)
        print(f"Saved equity curve -> {out}")
    else:
        if args.once:
            live_once(args.freq, force=args.force, after_hours=args.after_hours)
        else:
            live_loop(args.freq, force=args.force, after_hours=args.after_hours)

if __name__ == "__main__":
    main()
