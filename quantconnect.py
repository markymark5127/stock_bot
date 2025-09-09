# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
# endregion

class MomentumRotationQC(QCAlgorithm):

    # ==== Parameters (can be overridden in QC cloud "Parameters") ====
    FREQ = "Monthly"                 # "Monthly" or "Weekly"
    START_CASH = 100000
    START_DATE = "2014-01-01"
    END_DATE   = None                # or "2025-01-01" for backtest end
    DAYS_PER_MONTH = 21

    # Profiles
    PROFILES = {
        "conservative": {
            "UNIVERSE": ["SPY","QQQ","IWM","EFA","EEM"],
            "RISK_OFF": "SHY",
            "TOP_N": 2,
            "LOOKBACK_M": [3,6,12],
            "ABS_THRESH": 0.0,
        },
        "balanced": {
            "UNIVERSE": ["QQQ","XLK","SPY","ARKK","IWM"],
            "RISK_OFF": "GLD",
            "TOP_N": 2,
            "LOOKBACK_M": [1,3,6],
            "ABS_THRESH": 0.0,
        },
        "aggressive": {
            "UNIVERSE": ["QQQ","TQQQ","SOXL","ARKK","SPY"],
            "RISK_OFF": "GLD",
            "TOP_N": 1,
            "LOOKBACK_M": [1,3],
            "ABS_THRESH": 0.0,
        },
    }

    # Auto regime thresholds & hysteresis
    AUTO_HYSTERESIS = 0.002   # 0.2%
    TREND_STRONG    = 0.00    # SPY >= SMA200 -> strong
    TREND_OK        = -0.03   # within -3% of SMA200 -> ok

    # Enhancements
    RELATIVE_MOM_BENCH = "SPY"  # None or "SPY"
    SKIP_MONTH = True           # 12-1 momentum if True
    VOL_TARGET_ANNUAL = 0.12    # None to disable
    VOL_LOOKBACK_DAYS = 20
    VOL_MAX_LEVERAGE  = 1.5
    COST_BPS = 5                # per rebalance leg estimate; used only to log est costs

    # Risk-off basket (best-of each rebalance)
    RISK_OFF_BASKET = ["BIL","SHY","IEF"]

    def Initialize(self):
        # --- Parameters from QC UI (optional overrides) ---
        freq = self.GetParameter("freq")
        if freq:
            self.FREQ = freq
        vol_t = self.GetParameter("vol_target")
        if vol_t:
            try:
                self.VOL_TARGET_ANNUAL = float(vol_t)
            except:
                pass
        skip_m = self.GetParameter("skip_month")
        if skip_m:
            self.SKIP_MONTH = (skip_m.lower() in ("true","1","yes","y"))

        # --- Engine setup ---
        sdate = self.START_DATE.split("-")
        self.SetStartDate(int(sdate[0]), int(sdate[1]), int(sdate[2]))
        if self.END_DATE:
            ed = self.END_DATE.split("-")
            self.SetEndDate(int(ed[0]), int(ed[1]), int(ed[2]))
        self.SetCash(self.START_CASH)
        self.SetBenchmark("SPY")

        # Optional: simple fee & slippage models (tweak/disable as you like)
        self.SetSecurityInitializer(self._init_security_models)

        # --- Build union of all tickers we might trade in "auto" ---
        union = set(["SPY"]) | set(self.RISK_OFF_BASKET)
        for p in self.PROFILES.values():
            union |= set(p["UNIVERSE"])
            union.add(p["RISK_OFF"])

        # Subscribe all with daily resolution
        self.symbols = {}
        for t in sorted(union):
            try:
                self.symbols[t] = self.AddEquity(t, Resolution.Daily).Symbol
            except:
                self.Debug(f"Failed to add {t}")

        # For logging/benchmark rules
        self.spy = self.symbols.get("SPY")

        # Rebalance scheduler: first trading day each week/month @ +10 minutes after open
        if self.FREQ.lower().startswith("month"):
            self.Schedule.On(self.DateRules.MonthStart(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 10), self._rebalance)
        else:
            self.Schedule.On(self.DateRules.WeekStart(self.spy), self.TimeRules.AfterMarketOpen(self.spy, 10), self._rebalance)

        # State
        self.last_profile = None
        self.last_rebalance_key = None
        self.prev_targets = []  # set of tickers last chosen (for bands/cost est if you add later)

        self.Debug(f"Init done. FREQ={self.FREQ}, SKIP_MONTH={self.SKIP_MONTH}, VOL_TARGET={self.VOL_TARGET_ANNUAL}")

    # ------------------ Models (optional/simple) ------------------
    def _init_security_models(self, security: Security):
        # Simple constant bps model approx (both as fee & slippage, conservative)
        # Comment these two to revert to QC defaults.
        bps = 1.0  # 1 bp per fill as "fee" proxy
        security.SetFeeModel(ConstantFeeModel(0.0))  # set to 0 and bake bps as slippage
        security.SetSlippageModel(ConstantSlippageModel(bps / 10000.0))

    # ------------------ Utility: history matrix -------------------
    def _get_close_matrix(self, lookback_days:int) -> pd.DataFrame:
        syms = list(self.symbols.values())
        hist = self.History(syms, lookback_days+5, Resolution.Daily)
        if hist.empty:
            return pd.DataFrame()
        # QC returns MultiIndex DataFrame; pivot to dates x tickers
        df = hist.close.unstack(level=0)  # columns = Symbol
        # Map columns back to tickers
        mapper = {sym: str(sym) for sym in df.columns}
        df.columns = [self.Securities[s].Symbol.Value for s in df.columns]
        return df.sort_index()

    # ------------------ Math helpers -------------------
    def _sma(self, s: pd.Series, window:int) -> float:
        s = s.dropna()
        if len(s) < window: return np.nan
        return float(s.tail(window).mean())

    def _month_return(self, s: pd.Series, months:int) -> float:
        # Use trading-day approximation
        k = months * self.DAYS_PER_MONTH
        s = s.dropna()
        if len(s) <= k: return np.nan
        p0 = float(s.iloc[-k-1]); p1 = float(s.iloc[-1])
        return (p1/p0 - 1.0) if p0 > 0 else np.nan

    def _blended_momentum(self, s: pd.Series, months_list, skip_month:bool) -> float:
        if not months_list:
            return np.nan
        s = s.dropna()
        shift_end = self.DAYS_PER_MONTH if skip_month else 0
        # Need enough history:
        max_m = max(months_list) + (1 if skip_month else 0)
        min_needed = max(5, (max_m * self.DAYS_PER_MONTH) + 2)
        if len(s) < min_needed:
            return np.nan

        def ret_m(m):
            back = (m + (1 if skip_month else 0)) * self.DAYS_PER_MONTH
            if len(s) <= back + shift_end: 
                return np.nan
            p0 = float(s.iloc[-back-1])
            p1 = float(s.iloc[-1 - shift_end]) if skip_month else float(s.iloc[-1])
            return (p1/p0 - 1.0) if p0 > 0 else np.nan

        vals = [ret_m(m) for m in months_list]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    # ------------------ Regime switcher (auto) -------------------
    def _choose_profile_auto(self, prices: pd.DataFrame, prev_profile:str|None=None):
        if "SPY" not in prices.columns:
            return "balanced", np.nan, np.nan, np.nan, np.nan
        spy = prices["SPY"].dropna()
        if spy.empty:
            return "balanced", np.nan, np.nan, np.nan, np.nan

        sma200 = self._sma(spy, 200)
        sma100 = self._sma(spy, 100)
        last   = float(spy.iloc[-1])
        if not np.isfinite(sma200) or sma200 <= 0:
            return "balanced", last, sma200, np.nan, np.nan

        pct_above = last / sma200 - 1.0
        r1 = self._month_return(spy,1)
        r3 = self._month_return(spy,3)
        any_pos = (np.isfinite(r1) and r1 > 0) or (np.isfinite(r3) and r3 > 0)

        # SMA100 slope (approx 1 week)
        sma100_prev = np.nan
        if len(spy) >= 105:
            sma_series = spy.rolling(100).mean()
            sma100_prev = float(sma_series.iloc[-6])
        sma100_rising = (np.isfinite(sma100) and np.isfinite(sma100_prev) and sma100 > sma100_prev)

        # Tech overrides
        override_aggr = False
        if "QQQ" in prices.columns:
            qqq = prices["QQQ"].dropna()
            r3_qqq = self._month_return(qqq,3)
            if np.isfinite(r3_qqq) and r3_qqq >= 0.10:
                override_aggr = True
        for lev in ("TQQQ","SOXL"):
            if lev in prices.columns:
                s = prices[lev].dropna()
                r3_lev = self._month_return(s,3)
                if np.isfinite(r3_lev) and r3_lev >= 0.20:
                    override_aggr = True

        # QQQ SMA200 confirm
        qqq_confirm = False
        if "QQQ" in prices.columns:
            qqq = prices["QQQ"].dropna()
            qqq_sma200 = self._sma(qqq,200)
            if np.isfinite(qqq_sma200):
                if float(qqq.iloc[-1]) / qqq_sma200 - 1.0 > (0.0 + self.AUTO_HYSTERESIS):
                    qqq_confirm = True

        # Bear failsafe
        if (pct_above < (self.TREND_OK - 0.02)) and (np.isfinite(r3) and r3 < -0.04):
            desired = "conservative"
        else:
            if override_aggr:
                desired = "aggressive"
            elif qqq_confirm and any_pos:
                desired = "aggressive"
            elif (pct_above > (self.TREND_STRONG + self.AUTO_HYSTERESIS) and any_pos) or (sma100_rising and any_pos):
                desired = "aggressive"
            elif pct_above > (self.TREND_OK - self.AUTO_HYSTERESIS) and (np.isfinite(r3) and r3 >= -0.02):
                desired = "balanced"
            else:
                desired = "conservative"

            if prev_profile:
                if self.TREND_STRONG - self.AUTO_HYSTERESIS <= pct_above <= self.TREND_STRONG + self.AUTO_HYSTERESIS:
                    desired = prev_profile
                if self.TREND_OK - self.AUTO_HYSTERESIS <= pct_above <= self.TREND_OK + self.AUTO_HYSTERESIS:
                    desired = prev_profile

        return desired, last, sma200, r1, r3

    # ------------------ Core rebalance -------------------
    def _rebalance(self):
        # Grab enough history for all calcs
        max_m = 12 + (1 if self.SKIP_MONTH else 0)
        need_days = max(260, max_m * self.DAYS_PER_MONTH + 50)  # extra buffer
        prices = self._get_close_matrix(need_days)
        if prices.empty:
            self.Debug("No history yet")
            return

        # Decide profile (auto), or keep last selection if you prefer fixed
        chosen, last_spy, sma200, r1, r3 = self._choose_profile_auto(prices, prev_profile=self.last_profile)
        self.last_profile = chosen
        cfg = self.PROFILES[chosen]
        universe = [t for t in cfg["UNIVERSE"] if t in prices.columns]
        risk_off = cfg["RISK_OFF"] if cfg["RISK_OFF"] in prices.columns else None

        # Compute blended momentum for universe
        scores = {}
        for t in universe:
            s = prices[t].dropna()
            scores[t] = self._blended_momentum(s, cfg["LOOKBACK_M"], self.SKIP_MONTH)

        ranked = sorted(scores.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -9e9), reverse=True)

        # Relative momentum vs SPY
        rel_gate = -1e9
        if self.RELATIVE_MOM_BENCH and self.RELATIVE_MOM_BENCH in prices.columns:
            rel_gate = self._blended_momentum(prices[self.RELATIVE_MOM_BENCH].dropna(), cfg["LOOKBACK_M"], self.SKIP_MONTH)

        qualified = []
        for t, s in ranked:
            if not np.isfinite(s): 
                continue
            if s <= cfg["ABS_THRESH"]:
                continue
            if self.RELATIVE_MOM_BENCH and t != self.RELATIVE_MOM_BENCH and np.isfinite(rel_gate) and s <= rel_gate:
                continue
            qualified.append((t, s))

        picks = [t for t,_ in qualified[:cfg["TOP_N"]]]

        # SPY fallback (if nothing else qualifies but SPY trend positive)
        if (not picks) and ("SPY" in prices.columns):
            spy = prices["SPY"].dropna()
            spy_ok = False
            if len(spy) >= 200:
                sma200_spy = spy.rolling(200).mean().iloc[-1]
                if np.isfinite(sma200_spy) and sma200_spy > 0:
                    spy_ok = (spy.iloc[-1]/sma200_spy - 1.0) > 0 and (self._month_return(spy,3) > 0)
            if spy_ok:
                picks = ["SPY"]

        # Risk-off best-of (3m momentum)
        risk_cands = [r for r in (self.RISK_OFF_BASKET + ([risk_off] if risk_off else [])) if r in prices.columns]
        if risk_cands:
            ro_scores = {r: self._month_return(prices[r].dropna(),3) for r in risk_cands}
            best_ro = max(ro_scores.items(), key=lambda kv: (-1e9 if not np.isfinite(kv[1]) else kv[1]))[0]
        else:
            best_ro = risk_off

        # Vol targeting
        w_risky = 1.0
        if self.VOL_TARGET_ANNUAL is not None and picks:
            look = prices[[t for t in picks if t in prices.columns]].dropna()
            if not look.empty:
                ret = look.pct_change().dropna()
                if not ret.empty:
                    ew = ret.mean(axis=1)
                    ann_vol = float(ew.std() * np.sqrt(252)) if len(ew) else np.nan
                    if np.isfinite(ann_vol) and ann_vol > 1e-8:
                        w_risky = min(self.VOL_TARGET_ANNUAL / ann_vol, self.VOL_MAX_LEVERAGE)
                        w_risky = float(max(0.0, w_risky))
        w_safe = max(0.0, 1.0 - min(1.0, w_risky))

        # Build target weights
        targets = {}
        if picks:
            eq_w = w_risky / len(picks)
            for t in picks:
                if t in self.symbols:
                    targets[self.symbols[t]] = eq_w
        if best_ro and best_ro in self.symbols:
            targets[self.symbols[best_ro]] = targets.get(self.symbols[best_ro], 0.0) + w_safe

        # Log
        nice_scores = ", ".join(f"{k}:{'NA' if (v is None or not np.isfinite(v)) else round(float(v),4)}" 
                                for k,v in scores.items())
        self.Log(f"[{self.Time:%Y-%m-%d}] Rebalance [{self.FREQ}|{chosen}] picks={picks or [best_ro]} "
                 f"w_risky={round(w_risky,3)} w_safe={round(w_safe,3)}  scores={{ {nice_scores} }}")

        # Apply portfolio targets
        # First, zero out anything not in targets
        current_syms = [kv.Key for kv in self.Portfolio if self.Portfolio[kv.Key].Invested]
        for sym in current_syms:
            if sym not in targets:
                self.SetHoldings(sym, 0)

        # Then set desired weights
        for sym, w in targets.items():
            self.SetHoldings(sym, float(w))

        # Save a simple rebalance key to avoid accidental repeats if you add extra intraday triggers
        if self.FREQ.lower().startswith("month"):
            key = f"M-{self.Time.year}-{self.Time.month:02d}"
        else:
            iso = self.Time.isocalendar()
            key = f"W-{iso[0]}-{int(iso[1]):02d}"
        self.last_rebalance_key = key
