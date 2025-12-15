from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    total_pnl: float
    num_trades: int
    hit_ratio: float
    avg_pnl: float
    max_drawdown: float
    annualized_pnl: float


def mean_reversion_backtest(
    series: pd.DataFrame,
    *,
    z_entry: float = 2.0,
    z_exit: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, BacktestMetrics]:
    """Simple mean-reversion backtest on spread using z-score triggers.

    Strategy rules:
    - Enter SHORT spread when z < -z_entry
    - Enter LONG spread when z > +z_entry
    - Exit when z crosses 0 (<=0 for long, >=0 for short)

    P&L is computed on the spread itself:
    - LONG pnl = spread_exit - spread_entry
    - SHORT pnl = -(spread_exit - spread_entry)

    Inputs:
    - series: DataFrame indexed by dt (or has dt column) with at least columns: spread, zscore

    Returns:
    - trades_df: one row per trade
    - equity_df: time series with columns dt, equity
    - metrics
    """

    df = series.copy()
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], utc=True)
        df = df.set_index("dt")

    if df.empty:
        trades = pd.DataFrame(columns=["entry_dt", "exit_dt", "side", "entry_spread", "exit_spread", "pnl"])
        equity = pd.DataFrame(columns=["dt", "equity"])
        m = BacktestMetrics(total_pnl=0.0, num_trades=0, hit_ratio=float("nan"), avg_pnl=float("nan"), max_drawdown=0.0, annualized_pnl=0.0)
        return trades, equity, m

    df = df[["spread", "zscore"]].dropna().copy()
    if len(df) < 5:
        raise ValueError("not enough data to backtest")

    position = 0  # +1 long spread, -1 short spread, 0 flat
    entry_dt = None
    entry_spread = None

    trades: list[dict[str, object]] = []
    equity = 0.0
    equity_points: list[tuple[pd.Timestamp, float]] = []

    start_dt = df.index.min()
    end_dt = df.index.max()

    for dt, row in df.iterrows():
        z = float(row["zscore"])
        spr = float(row["spread"])

        # Exit logic
        if position == 1 and z <= z_exit:
            pnl = spr - float(entry_spread)
            trades.append(
                {
                    "entry_dt": entry_dt,
                    "exit_dt": dt,
                    "side": "LONG",
                    "entry_spread": float(entry_spread),
                    "exit_spread": spr,
                    "pnl": float(pnl),
                }
            )
            equity += float(pnl)
            position = 0
            entry_dt = None
            entry_spread = None

        elif position == -1 and z >= -z_exit:
            pnl = -(spr - float(entry_spread))
            trades.append(
                {
                    "entry_dt": entry_dt,
                    "exit_dt": dt,
                    "side": "SHORT",
                    "entry_spread": float(entry_spread),
                    "exit_spread": spr,
                    "pnl": float(pnl),
                }
            )
            equity += float(pnl)
            position = 0
            entry_dt = None
            entry_spread = None

        # Entry logic (only if flat)
        if position == 0:
            if z > z_entry:
                position = 1
                entry_dt = dt
                entry_spread = spr
            elif z < -z_entry:
                position = -1
                entry_dt = dt
                entry_spread = spr

        equity_points.append((dt, float(equity)))

    trades_df = pd.DataFrame(trades)

    equity_df = pd.DataFrame(equity_points, columns=["dt", "equity"])
    equity_df["dt"] = pd.to_datetime(equity_df["dt"], utc=True)

    total_pnl = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 0.0
    num_trades = int(len(trades_df))
    hit_ratio = float((trades_df["pnl"] > 0).mean()) if num_trades > 0 else float("nan")
    avg_pnl = float(trades_df["pnl"].mean()) if num_trades > 0 else float("nan")

    # Max drawdown on equity curve
    eq = equity_df["equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Annualized PnL rate (notional-free): total_pnl / elapsed_years
    elapsed_seconds = max(1.0, float((end_dt - start_dt).total_seconds()))
    elapsed_years = elapsed_seconds / (365.25 * 24 * 3600)
    annualized_pnl = float(total_pnl / elapsed_years) if elapsed_years > 0 else 0.0

    metrics = BacktestMetrics(
        total_pnl=total_pnl,
        num_trades=num_trades,
        hit_ratio=hit_ratio,
        avg_pnl=avg_pnl,
        max_drawdown=max_dd,
        annualized_pnl=annualized_pnl,
    )

    return trades_df, equity_df, metrics
