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

# ======== GLOBAL DEFAULTS (overridden by profile/apply_profile) ========
UNIVERSE     = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
RISK_OFF     = "SHY"
TOP_N        = 2
LOOKBACK_M   = [3, 6, 12]
ABS_THRESH   = 0.0
STATE_FILE   = "rotation_state.json"
POLL_SECONDS = 60
MIN_DOLLARS_PER_ORDER = 50
MAX_PORTFOLIO_PCT_PER_TICKER = 1.0 / TOP_N
TZ = "America/New_York"

# --- Enhancements toggles ---
RELATIVE_MOM_BENCH = "SPY"   # None or "SPY"
SKIP_MONTH = True            # use 12-1 momentum if True (skip most recent month)
VOL_TARGET_ANNUAL = 0.16     # set None to disable vol targeting
VOL_LOOKBACK_DAYS = 20
VOL_MAX_LEVERAGE = 2.0       # cap risky sleeve scaling
REBALANCE_BAND = 0.10        # 10% band; set 0 to disable
COST_BPS = 5                 # per-trade cost in basis points (e.g., 5 = 0.05%)

# Risk-off basket (best-of at each rebalance if multiple given)
RISK_OFF_BASKET = ["BIL", "SHY", "IEF"]   # override/augment profile RISK_OFF

# Alpaca feed (free default = iex)
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()

# Optional intra-period protection
INTRA_MONTH_STOP_PCT = None
PORTFOLIO_DD_STOP_PCT = None

# Backtest defaults
BT_START = "2012-01-01"
BT_END   = None  # today
# ================================================================

# ----------------- PROFILES -----------------
PROFILES = {
    # Classic, slow & steady
    "conservative": {
        "UNIVERSE": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
        "RISK_OFF": "SHY",
        "TOP_N": 2,
        "LOOKBACK_M": [3, 6, 12],
        "ABS_THRESH": 0.0,
        "INTRA_MONTH_STOP_PCT": None,
        "PORTFOLIO_DD_STOP_PCT": None,
    },
    # Balanced-but-spicy
    "balanced": {
        "UNIVERSE": ["QQQ", "XLK", "SPY", "ARKK", "IWM"],
        "RISK_OFF": "GLD",
        "TOP_N": 2,
        "LOOKBACK_M": [1, 3, 6],
        "ABS_THRESH": 0.0,
        "INTRA_MONTH_STOP_PCT": -0.10,
        "PORTFOLIO_DD_STOP_PCT": -0.15,
    },
    # Max aggression
    "aggressive": {
        "UNIVERSE": ["QQQ", "TQQQ", "SOXL", "ARKK", "SPY"],
        "RISK_OFF": "GLD",
        "TOP_N": 1,
        "LOOKBACK_M": [1, 3],
        "ABS_THRESH": 0.0,
        "INTRA_MONTH_STOP_PCT": -0.12,
        "PORTFOLIO_DD_STOP_PCT": -0.20,
    },
}

# Regime thresholds & hysteresis (lean AGGRESSIVE faster)
AUTO_HYSTERESIS = 0.002   # 0.2% buffer
TREND_STRONG    = 0.00    # SPY ≥ SMA200 -> strong
TREND_OK        = -0.03   # within -3% of SMA200 -> "ok"

def apply_profile(name: str):
    """Override module-level config from a profile."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from: {', '.join(PROFILES)}")
    cfg = PROFILES[name]
    globals()["UNIVERSE"] = cfg["UNIVERSE"]
    globals()["RISK_OFF"] = cfg["RISK_OFF"]
    globals()["TOP_N"] = cfg["TOP_N"]
    globals()["LOOKBACK_M"] = cfg["LOOKBACK_M"]
    globals()["ABS_THRESH"] = cfg["ABS_THRESH"]
    globals()["INTRA_MONTH_STOP_PCT"] = cfg["INTRA_MONTH_STOP_PCT"]
    globals()["PORTFOLIO_DD_STOP_PCT"] = cfg["PORTFOLIO_DD_STOP_PCT"]
    globals()["MAX_PORTFOLIO_PCT_PER_TICKER"] = 1.0 / max(1, globals()["TOP_N"])

# ---------- Utilities / State ----------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_rebalance_key": "",
        "month_entries": {},
        "month_start_equity": None,
        "last_profile": None,   # remember last chosen profile for live hysteresis
    }

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

# ---------- Helpers for returns & benchmark ----------
def equal_weight_period_return(prices: pd.DataFrame, tickers, t0, t1) -> float:
    """Equal-weight return between t0 and t1 over the specified tickers."""
    ticks = [t for t in tickers if t in prices.columns]
    if not ticks:
        return np.nan
    p0 = prices.loc[t0, ticks].astype(float)
    p1 = prices.loc[t1, ticks].astype(float)
    p0 = p0[p0 > 0].dropna()
    p1 = p1[p1 > 0].dropna()
    common = p0.index.intersection(p1.index)
    if len(common) == 0:
        return np.nan
    rets = (p1[common] / p0[common] - 1.0).values
    return float(np.nanmean(rets))

def buy_and_hold_curve(prices: pd.Series, start_equity=10000.0) -> pd.DataFrame:
    """Buy & hold curve for a single series (e.g., SPY)."""
    s = prices.dropna().astype(float)
    if s.empty:
        return pd.DataFrame(columns=["equity"])
    base = float(s.iloc[0])
    eq = start_equity * (s / base)
    return eq.to_frame("equity")

def series_sharpe_dd(curve: pd.DataFrame, freq: str):
    """Naive Sharpe annualized by period frequency ('monthly' or 'weekly'); also max DD."""
    if curve.empty or "equity" not in curve:
        return 0.0, 0.0
    period_rets = curve["equity"].pct_change().dropna()
    if not len(period_rets):
        return 0.0, 0.0
    ann_k = np.sqrt(12) if freq == "monthly" else (np.sqrt(52) if freq == "weekly" else np.sqrt(252))
    sharpe = float(ann_k * (period_rets.mean() / (period_rets.std() + 1e-12)))
    max_dd = float((curve["equity"] / curve["equity"].cummax() - 1).min())
    return sharpe, max_dd

def blended_momentum(prices: pd.Series, months_list, skip_month=SKIP_MONTH):
    if not months_list:
        return np.nan
    max_m = max(months_list) + (1 if skip_month else 0)
    min_needed = 22 * max_m + 5
    s = prices.dropna()
    if len(s) < min_needed:
        return np.nan

    def ret_months(m):
        # if skip_month: we measure from (m+1) months ago to 1 month ago
        if skip_month:
            end_t = s.index[-1] - pd.DateOffset(months=1)
            start_t = s.index[-1] - pd.DateOffset(months=m+1)
        else:
            end_t = s.index[-1]
            start_t = s.index[-1] - pd.DateOffset(months=m)
        i0 = s.index.searchsorted(start_t)
        i1 = s.index.searchsorted(end_t)
        if i0 <= 0 or i1 <= 0 or i0 >= len(s) or i1 >= len(s) or i1 <= i0:
            return np.nan
        p0, p1 = float(s.iloc[i0]), float(s.iloc[i1])
        return (p1 / p0 - 1.0) if p0 > 0 else np.nan

    rets = [ret_months(m) for m in months_list]
    rets = [r for r in rets if np.isfinite(r)]
    return float(np.nanmean(rets)) if rets else np.nan

def sma(series: pd.Series, window=200):
    s = series.dropna()
    if len(s) < window: return np.nan
    return float(s.tail(window).mean())

def month_return(series: pd.Series, months=1):
    s = series.dropna()
    if s.empty: return np.nan
    t0 = s.index[-1] - pd.DateOffset(months=months)
    idx = s.index.searchsorted(t0)
    if idx <= 0 or idx >= len(s): return np.nan
    p0 = float(s.iloc[idx]); p1 = float(s.iloc[-1])
    return (p1 / p0 - 1.0) if p0 > 0 else np.nan

# ---------- BACKTEST ----------
def get_history_yf(tickers, start, end):
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")

def period_end_indices(prices: pd.DataFrame, freq: str):
    """Return target period-end timestamps using resample ('ME' month-end, 'W-FRI')."""
    freq_norm = "ME" if freq in ("M","monthly") else ("W-FRI" if freq in ("W-FRI","weekly") else freq)
    return prices.resample(freq_norm).last().index

def choose_profile_auto3(hist_prices: pd.DataFrame, prev_profile: str | None = None,
                         hyst=AUTO_HYSTERESIS):
    """
    Decide among 'aggressive' | 'balanced' | 'conservative' using:
      - SPY vs SMA200
      - SPY 1m/3m momentum (any positive)
      - SMA100 slope
      - Tech overrides (QQQ/TQQQ/SOXL)
      - QQQ above its SMA200 confirmation
      - Bear failsafe
    """
    if "SPY" not in hist_prices.columns:
        return "balanced", np.nan, np.nan, np.nan, np.nan

    spy = hist_prices["SPY"].dropna()
    if spy.empty:
        return "balanced", np.nan, np.nan, np.nan, np.nan

    sma200 = sma(spy, 200)
    sma100 = sma(spy, 100)
    last   = float(spy.iloc[-1])
    if not np.isfinite(sma200) or sma200 <= 0:
        return "balanced", last, sma200, np.nan, np.nan

    pct_above = last / sma200 - 1.0
    r1 = month_return(spy, 1)
    r3 = month_return(spy, 3)
    any_mom_pos = (np.isfinite(r1) and r1 > 0) or (np.isfinite(r3) and r3 > 0)

    # Simple SMA100 slope: last vs ~1 week earlier
    sma100_prev = np.nan
    try:
        if len(spy) >= 105:
            sma_series = spy.rolling(100).mean()
            sma100_prev = float(sma_series.iloc[-6])
    except Exception:
        pass
    sma100_rising = (np.isfinite(sma100) and np.isfinite(sma100_prev) and sma100 > sma100_prev)

    # Tech momentum overrides (looser)
    override_aggr = False
    if "QQQ" in hist_prices.columns:
        qqq = hist_prices["QQQ"].dropna()
        r3_qqq = month_return(qqq, 3)
        if np.isfinite(r3_qqq) and r3_qqq >= 0.10:
            override_aggr = True

    for lev_t in ("TQQQ", "SOXL"):
        if lev_t in hist_prices.columns:
            s = hist_prices[lev_t].dropna()
            r3_lev = month_return(s, 3)
            if np.isfinite(r3_lev) and r3_lev >= 0.20:
                override_aggr = True

    # QQQ SMA200 confirmation
    qqq_trend_confirm = False
    if "QQQ" in hist_prices.columns:
        qqq = hist_prices["QQQ"].dropna()
        qqq_sma200 = sma(qqq, 200)
        if np.isfinite(qqq_sma200):
            if float(qqq.iloc[-1]) / qqq_sma200 - 1.0 > (0.0 + hyst):
                qqq_trend_confirm = True

    # Bear failsafe
    if (pct_above < (TREND_OK - 0.02)) and (np.isfinite(r3) and r3 < -0.04):
        desired = "conservative"
    else:
        if override_aggr:
            desired = "aggressive"
        elif qqq_trend_confirm and any_mom_pos:
            desired = "aggressive"
        elif (pct_above > (TREND_STRONG + hyst) and any_mom_pos) or (sma100_rising and any_mom_pos):
            desired = "aggressive"
        elif pct_above > (TREND_OK - hyst) and (np.isfinite(r3) and r3 >= -0.02):
            desired = "balanced"
        else:
            desired = "conservative"

        # Hysteresis stabilization near borders
        if prev_profile:
            if TREND_STRONG - hyst <= pct_above <= TREND_STRONG + hyst:
                desired = prev_profile
            if TREND_OK - hyst <= pct_above <= TREND_OK + hyst:
                desired = prev_profile

    return desired, last, sma200, r1, r3


def rotation_backtest(profile_name, start, end, freq,
                      topn_override=None, months_override=None, abs_override=None):
    """
    Backtest with either a fixed profile or 'auto' that switches by regime
    (aggressive/balanced/conservative).

    Optional overrides:
      - topn_override: int or None
      - months_override: list[int] or None
      - abs_override: float or None
    Returns (curve, stats).
    """
    # Build union of tickers we may need
    if profile_name == "auto":
        tickers = set()
        for p in ("aggressive", "balanced", "conservative"):
            tickers |= set(PROFILES[p]["UNIVERSE"])
            tickers |= {PROFILES[p]["RISK_OFF"]}
        tickers |= {"SPY"}  # regime reference & benchmark
    else:
        cfg = PROFILES[profile_name]
        tickers = set(cfg["UNIVERSE"]) | {cfg["RISK_OFF"], "SPY"}

    # include risk-off basket so we can evaluate best-of
    tickers |= set(RISK_OFF_BASKET)

    prices = get_history_yf(sorted(tickers), start, end).dropna()
    prices = prices.sort_index()
    ends = period_end_indices(prices, "ME" if freq == "monthly" else "W-FRI")

    equity = 10000.0
    curve = []

    # track previous period's targets for bands/costs
    prev_targets = []

    for i, t_end in enumerate(ends):
        # last trading day on/before period end
        idx = prices.index[prices.index <= t_end]
        if len(idx) == 0:
            continue
        t0 = idx.max()
        hist = prices.loc[:t0]

        # Require some history
        if len(hist) < 250:
            curve.append((t0, equity))
            continue

        # Choose profile at this time
        if profile_name == "auto":
            chosen, last_spy, sma200, r1, r3 = choose_profile_auto3(hist, prev_profile=None)
            cfg = PROFILES[chosen]
        else:
            chosen = profile_name
            cfg = PROFILES[profile_name]

        universe   = [t for t in cfg["UNIVERSE"] if t in hist.columns]
        risk_off   = cfg["RISK_OFF"] if cfg["RISK_OFF"] in hist.columns else None

        # ---- APPLY OVERRIDES (if provided) ----
        top_n      = topn_override if topn_override is not None else cfg["TOP_N"]
        months     = months_override if months_override is not None else cfg["LOOKBACK_M"]
        abs_thresh = abs_override if abs_override is not None else cfg["ABS_THRESH"]
        # --------------------------------------

        if len(universe) == 0:
            curve.append((t0, equity))
            continue

        # Compute scores in the chosen universe
        scores = {t: blended_momentum(hist[t].dropna(), months) for t in universe}
        ranked = sorted(scores.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -9e9), reverse=True)

        # Relative momentum threshold vs SPY (and absolute)
        rel_gate = -1e9
        if RELATIVE_MOM_BENCH and RELATIVE_MOM_BENCH in hist.columns:
            rel_gate = blended_momentum(hist[RELATIVE_MOM_BENCH].dropna(), months)

        qualified = []
        for t, s in ranked:
            if not np.isfinite(s):
                continue
            if s <= abs_thresh:
                continue
            # Only gate *non-SPY* vs SPY
            if RELATIVE_MOM_BENCH and t != RELATIVE_MOM_BENCH and np.isfinite(rel_gate) and s <= rel_gate:
                continue
            qualified.append((t, s))


        picks = [t for t, _ in qualified[:top_n]]

        # --- NEW: default to SPY when its trend is positive ---
        if not picks and "SPY" in hist.columns:
            spy = hist["SPY"].dropna()
            spy_ok = False
            if len(spy) >= 200:
                spy_ok = (spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1.0) > 0 and month_return(spy, 3) > 0
            if spy_ok:
                picks = ["SPY"]


        # Risk-off: best-of basket if available, otherwise profile's RISK_OFF
        risk_cands = [r for r in (RISK_OFF_BASKET + ([risk_off] if risk_off else [])) if r in hist.columns]
        if not risk_cands:
            targets = picks if picks else ([risk_off] if risk_off else [])
        else:
            # choose best risk-off by 3m momentum
            ro_scores = {r: month_return(hist[r].dropna(), 3) for r in risk_cands}
            best_ro = max(ro_scores.items(), key=lambda kv: (-1e9 if not np.isfinite(kv[1]) else kv[1]))[0]
            targets = picks if picks else [best_ro]

        # Next period end (or last available)
        t_next_nominal = ends[i + 1] if i < len(ends) - 1 else prices.index[-1]
        idx2 = prices.index[prices.index <= t_next_nominal]
        if len(idx2) == 0 or idx2.max() <= t0:
            curve.append((t0, equity))
            continue
        t1 = idx2.max()

        # Rebalance band: compare set change
        changed = set(targets) != set(prev_targets)

        # Compute risky sleeve return
        risky_ret = equal_weight_period_return(prices, targets, t0, t1) if targets else np.nan
        risky_ret = 0.0 if not np.isfinite(risky_ret) else float(risky_ret)

        # Risk-off choice for this period (best-of again, using t0 history)
        risk_cands = [r for r in (RISK_OFF_BASKET + ([risk_off] if risk_off else [])) if r in prices.columns]
        ro = risk_cands[0] if risk_cands else None
        if risk_cands and len(risk_cands) > 1:
            hist_t0 = prices.loc[:t0]
            ro_scores = {r: month_return(hist_t0[r].dropna(), 3) for r in risk_cands}
            ro = max(ro_scores.items(), key=lambda kv: (-1e9 if not np.isfinite(kv[1]) else kv[1]))[0]
        riskoff_ret = equal_weight_period_return(prices, [ro], t0, t1) if ro else 0.0
        riskoff_ret = 0.0 if not np.isfinite(riskoff_ret) else float(riskoff_ret)

        # Volatility targeting on risky sleeve
        w_risky = 1.0
        if VOL_TARGET_ANNUAL is not None and targets:
            look_idx = prices.index[prices.index <= t0]
            look_idx = look_idx[-(VOL_LOOKBACK_DAYS+1):]
            if len(look_idx) >= 2:
                sub = prices.loc[look_idx, [t for t in targets if t in prices.columns]].dropna(axis=1, how="any")
                if not sub.empty and sub.shape[1] > 0:
                    sub_ret = sub.pct_change().dropna()
                    ew_ret = sub_ret.mean(axis=1)
                    ann_vol = float(ew_ret.std() * np.sqrt(252)) if len(ew_ret) else np.nan
                    if np.isfinite(ann_vol) and ann_vol > 1e-8:
                        ratio = VOL_TARGET_ANNUAL / ann_vol
                        # Only lever up when calm; never de-lever below 1.0
                        if ratio > 1.0:
                            w_risky = min(ratio, VOL_MAX_LEVERAGE)
                        else:
                            w_risky = 1.0

        # if leveraging, no "safe" sleeve; otherwise fill remaining with risk-off
        w_safe = 0.0 if w_risky > 1.0 else (1.0 - w_risky)


        period_ret = w_risky * risky_ret + w_safe * riskoff_ret

        # Rebalance band & rough costs
        if changed and COST_BPS:
            n_changed = len(set(prev_targets) ^ set(targets))  # symmetric diff
            legs = max(1, n_changed) * 2                      # exit+enter per changed name
            period_ret -= (COST_BPS / 10000.0) * legs

        equity *= (1.0 + period_ret)
        prev_targets = targets
        curve.append((t1, equity))

    # --- Strategy stats
    curve = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date")
    sharpe, max_dd = series_sharpe_dd(curve, freq=freq)

    # --- SPY benchmark over same window
    if len(curve):
        spy_series = prices["SPY"].loc[curve.index.min():curve.index.max()].dropna()
        spy_curve = buy_and_hold_curve(spy_series, start_equity=10000.0).reindex(curve.index).ffill()
    else:
        spy_curve = pd.DataFrame(columns=["equity"])
    spy_sharpe, spy_dd = series_sharpe_dd(spy_curve, freq=freq)

    stats = {
        "start": str(curve.index[0].date()) if len(curve) else start,
        "end":   str(curve.index[-1].date()) if len(curve) else (end or str(pd.Timestamp.today().date())),
        "final_equity": float(curve["equity"].iloc[-1]) if len(curve) else 10000.0,
        "return_pct": float((curve["equity"].iloc[-1] / 10000.0 - 1.0) * 100) if len(curve) else 0.0,
        "sharpe_naive": sharpe,
        "max_drawdown": max_dd,
        "freq": freq,
        "profile": profile_name,
        # ---- Benchmark add-ons ----
        "spy_final_equity": float(spy_curve["equity"].iloc[-1]) if len(spy_curve) else 10000.0,
        "spy_return_pct": float((spy_curve["equity"].iloc[-1] / 10000.0 - 1.0) * 100) if len(spy_curve) else 0.0,
        "spy_sharpe_naive": spy_sharpe,
        "spy_max_drawdown": spy_dd,
        "alpha_pct": (
            float((curve["equity"].iloc[-1] - 10000.0) - (spy_curve["equity"].iloc[-1] - 10000.0)) / 100.0
            if len(curve) and len(spy_curve) else 0.0
        ),
    }
    return curve, stats

def run_threshold_sweep(vals, start, end, freq):
    """
    Sweep TREND_STRONG over 'vals' (list of floats), run auto backtests,
    and print/save a results table.
    """
    global TREND_STRONG
    results = []

    original_trend_strong = TREND_STRONG
    try:
        for v in vals:
            TREND_STRONG = float(v)
            curve, stats = rotation_backtest(
                "auto", start=start, end=end, freq=freq,
                topn_override=None, months_override=None, abs_override=None
            )
            results.append({
                "trend_strong": v,
                "start": stats["start"],
                "end": stats["end"],
                "freq": stats["freq"],
                "final_equity": stats["final_equity"],
                "return_pct": stats["return_pct"],
                "sharpe_naive": stats["sharpe_naive"],
                "max_drawdown": stats["max_drawdown"],
                "spy_final_equity": stats.get("spy_final_equity"),
                "spy_return_pct": stats.get("spy_return_pct"),
                "alpha_pct": stats.get("alpha_pct"),
            })

        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by="return_pct", ascending=False)
        print("\n=== Threshold Sweep (TREND_STRONG) ===")
        with pd.option_context('display.float_format', '{:.4f}'.format):
            print(df_sorted.to_string(index=False))

        out_csv = f"sweep_trend_strong_{freq}.csv"
        df_sorted.to_csv(out_csv, index=False)
        print(f"\nSaved sweep results -> {out_csv}")
        return df_sorted
    finally:
        TREND_STRONG = original_trend_strong


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
    # scores (skip-month logic already in blended_momentum)
    scores = {t: blended_momentum(closes[t].dropna(), LOOKBACK_M) for t in UNIVERSE}
    ranked = sorted(scores.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -9e9), reverse=True)

    # relative momentum gate vs SPY
    rel_gate = -1e9
    if RELATIVE_MOM_BENCH and RELATIVE_MOM_BENCH in closes.columns:
        rel_gate = blended_momentum(closes[RELATIVE_MOM_BENCH].dropna(), LOOKBACK_M)

    qualified = []
    for t, s in ranked:
        if not np.isfinite(s): continue
        if s <= ABS_THRESH: continue
        if RELATIVE_MOM_BENCH and t != RELATIVE_MOM_BENCH and np.isfinite(rel_gate) and s <= rel_gate: continue 
        qualified.append((t, s))


    picks = [t for t, _ in qualified[:TOP_N]]

    # NEW: SPY fallback
    if not picks and "SPY" in closes.columns:
        spy = closes["SPY"].dropna()
        spy_ok = False
        if len(spy) >= 200:
            spy_ok = (spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1.0) > 0 and month_return(spy, 3) > 0
        if spy_ok:
            picks = ["SPY"]


    # best-of risk-off basket
    risk_cands = [r for r in (RISK_OFF_BASKET + [RISK_OFF]) if r in closes.columns]
    if risk_cands:
        ro_scores = {r: month_return(closes[r].dropna(), 3) for r in risk_cands}
        best_ro = max(ro_scores.items(), key=lambda kv: (-1e9 if not np.isfinite(kv[1]) else kv[1]))[0]
    else:
        best_ro = RISK_OFF

    return (picks if picks else [best_ro]), scores

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

# ---------- Live once with auto profile support ----------
def live_once(freq: str, force: bool=False, after_hours: bool=False, profile_name: str="balanced"):
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

    # Build ticker set (include risk-off basket)
    if profile_name == "auto":
        need = set()
        for p in ("aggressive","balanced","conservative"):
            need |= set(PROFILES[p]["UNIVERSE"]); need.add(PROFILES[p]["RISK_OFF"])
        need.add("SPY")
        need |= set(RISK_OFF_BASKET)
        tickers = sorted(need)
    else:
        tickers = sorted(set(PROFILES[profile_name]["UNIVERSE"])
                         | {PROFILES[profile_name]["RISK_OFF"], "SPY"}
                         | set(RISK_OFF_BASKET))

    closes = get_daily_closes_alpaca(api, tickers, days=800)
    if closes.empty:
        print("No data. Exiting.")
        return

    # Decide profile (auto) or apply fixed one
    def _fmt(x, nd=3):
        return "NA" if (x is None or not np.isfinite(x)) else str(round(float(x), nd))

    chosen = profile_name
    if profile_name == "auto":
        prev = state.get("last_profile")
        chosen, last_spy, sma200, r1, r3 = choose_profile_auto3(closes, prev_profile=prev)
        print(f"[AUTO] SPY last={_fmt(last_spy,2)} SMA200={_fmt(sma200,2)} r1={_fmt(r1,3)} r3={_fmt(r3,3)} → profile={chosen}")
        state["last_profile"] = chosen
        save_state(state)

    apply_profile(chosen)

    now = closes.index[-1]
    firsts = is_first_trading_of_month(closes.index) if freq=="monthly" else is_first_trading_of_week(closes.index)
    key_now = make_rebalance_key(now, freq)

    # Intra-period risk checks (if enabled by the chosen profile)
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
        print(f"[{datetime.now()}] Rebalance {freq} [{chosen}]: picks={targets}  scores={{ {', '.join(f'{k}: {('NA' if not np.isfinite(v) else round(v,4))}' for k,v in scores.items())} }}")
        send_sms(f"Rotation {freq} [{chosen}] rebalance -> {targets}")

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
def live_loop(freq: str, force: bool=False, after_hours: bool=False, profile_name: str="balanced"):
    while True:
        try:
            live_once(freq, force=force, after_hours=after_hours, profile_name=profile_name)
        except Exception as e:
            print("Error:", e)
        time.sleep(POLL_SECONDS)

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Momentum Rotation (monthly or weekly): backtest or live.")
    p.add_argument("--profile", choices=list(PROFILES.keys()) + ["auto"], default="balanced",
                   help="Parameter preset: conservative | balanced | aggressive | auto (auto-switches among the three)")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run local backtest with yfinance.")
    bt.add_argument("--start", default=BT_START)
    bt.add_argument("--end", default=BT_END)
    bt.add_argument("--topn", type=int, default=None, help="Override TOP_N")
    bt.add_argument("--months", default=None, help="Override lookbacks, e.g. '1,3,6'")
    bt.add_argument("--abs", type=float, default=None, help="Override ABS_THRESH")
    bt.add_argument("--freq", choices=["monthly","weekly"], default="weekly")

    live = sub.add_parser("live", help="Run Alpaca paper/live.")
    live.add_argument("--freq", choices=["monthly","weekly"], default="weekly")
    live.add_argument("--once", action="store_true", help="Run a single pass (ideal for Actions)")
    live.add_argument("--force", action="store_true", help="Run even if market is closed (for testing)")
    live.add_argument("--after-hours", action="store_true",
                      help="If market is closed, queue LIMIT orders eligible for extended hours")

    sweep = sub.add_parser("sweep", help="Sweep TREND_STRONG thresholds and compare performance.")
    sweep.add_argument("--start", default=BT_START)
    sweep.add_argument("--end", default=BT_END)
    sweep.add_argument("--freq", choices=["monthly","weekly"], default="monthly")
    sweep.add_argument("--vals", default="0,0.02,0.03,0.05",
                       help="Comma separated list of TREND_STRONG values to test (e.g., '0,0.02,0.03,0.05')")

    args = p.parse_args()

    # ---------- SWEEP ----------
    if args.cmd == "sweep":
        vals = [float(x.strip()) for x in args.vals.split(",") if x.strip()]
        run_threshold_sweep(vals, start=args.start, end=args.end, freq=args.freq)
        return

    # ---------- BACKTEST ----------
    if args.cmd == "backtest":
        months_override = None
        if args.months is not None:
            months_override = [int(x.strip()) for x in args.months.split(",") if x.strip()]
        topn_override = args.topn if args.topn is not None else None
        abs_override = float(args.abs) if args.abs is not None else None

        if args.profile == "auto":
            curve, stats = rotation_backtest(
                "auto", start=args.start, end=args.end, freq=args.freq,
                topn_override=topn_override, months_override=months_override, abs_override=abs_override
            )
        else:
            # Keep old behavior for non-auto: mutate globals, then call
            apply_profile(args.profile)
            if args.topn is not None:
                globals()["TOP_N"] = int(args.topn)
                globals()["MAX_PORTFOLIO_PCT_PER_TICKER"] = 1.0 / max(1, TOP_N)
            if args.months is not None:
                globals()["LOOKBACK_M"] = [int(x.strip()) for x in args.months.split(",") if x.strip()]
            if args.abs is not None:
                globals()["ABS_THRESH"] = float(args.abs)

            curve, stats = rotation_backtest(args.profile, start=args.start, end=args.end, freq=args.freq)

        print("Backtest Stats:")
        ordered_keys = [
            "start","end","freq","profile",
            "final_equity","return_pct","sharpe_naive","max_drawdown",
            "spy_final_equity","spy_return_pct","spy_sharpe_naive","spy_max_drawdown",
            "alpha_pct"
        ]
        for k in ordered_keys:
            if k in stats:
                print(f" - {k}: {stats[k]}")
        out = f"equity_{args.freq}.csv"
        curve.to_csv(out)
        print(f"Saved equity curve -> {out}")
        return

    # ---------- LIVE ----------
    if args.cmd == "live":
        if args.once:
            live_once(args.freq, force=args.force, after_hours=args.after_hours, profile_name=args.profile)
        else:
            live_loop(args.freq, force=args.force, after_hours=args.after_hours, profile_name=args.profile)

if __name__ == "__main__":
    main()
