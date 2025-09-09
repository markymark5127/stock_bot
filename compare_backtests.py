#!/usr/bin/env python3
"""
compare_backtests.py

Run backtests for multiple profiles over the same period and
produce a single CSV summary plus an equity curve chart.
Usage:
  python compare_backtests.py --start 2016-01-01 --end 2025-09-08 --freq weekly
"""
import argparse
import subprocess
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

PROFILES = ["conservative", "balanced", "aggressive"]

def run_backtest(profile: str, start: str, end: str | None, freq: str) -> str:
    cmd = [
        sys.executable, "momentum_rotation.py",
        "--profile", profile,
        "backtest",
        "--start", start,
        "--freq", freq
    ]
    if end:
        cmd += ["--end", end]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    # momentum_rotation.py writes equity_{freq}.csv in the CWD
    return f"equity_{freq}.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--freq", choices=["weekly", "monthly"], default="weekly")
    args = ap.parse_args()

    rows = []
    equity_frames = []

    for prof in PROFILES:
        # remove pre-existing file to avoid ambiguity
        out_csv = f"equity_{args.freq}.csv"
        if os.path.exists(out_csv):
            os.remove(out_csv)

        try:
            csv_path = run_backtest(prof, args.start, args.end, args.freq)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] backtest failed for {prof}: {e}", file=sys.stderr)
            continue

        # Parse the backtest stats from stdout would be messy; instead read equity file and compute simple stats.
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        start_equity = float(df['equity'].iloc[0])
        end_equity = float(df['equity'].iloc[-1])
        ret_pct = (end_equity / start_equity - 1.0) * 100.0
        daily = df['equity'].pct_change().dropna()
        sharpe = (daily.mean() / (daily.std() + 1e-12)) * (252 ** 0.5) if len(daily) else 0.0
        max_dd = ((df['equity']/df['equity'].cummax()) - 1.0).min()

        rows.append({
            "profile": prof,
            "start": df.index[0].date().isoformat(),
            "end": df.index[-1].date().isoformat(),
            "final_equity": round(end_equity, 2),
            "return_pct": round(ret_pct, 2),
            "sharpe_naive": round(sharpe, 3),
            "max_drawdown": round(float(max_dd), 4)
        })

        # Keep labeled equity for chart
        s = df['equity'].rename(prof)
        equity_frames.append(s)

    # Save summary CSV
    summary_path = "comparison_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    # Save combined equity chart
    if equity_frames:
        eq = pd.concat(equity_frames, axis=1).dropna(how="all")
        plt.figure(figsize=(10,6))
        for col in eq.columns:
            plt.plot(eq.index, eq[col], label=col)
        plt.title(f"Momentum Rotation — Equity Curves ({args.freq})")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        chart_path = "comparison_equity.png"
        plt.savefig(chart_path, dpi=150)
        print(f"Saved chart -> {chart_path}")
    else:
        print("No equity frames to chart.")

if __name__ == "__main__":
    main()
