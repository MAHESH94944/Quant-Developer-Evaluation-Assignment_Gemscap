from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from db.connection import connect
from db.schema import SCHEMA_SQL
from analytics.kalman_filter import kalman_filter_regression
from analytics.regression import OLSResult, RobustResult, hedge_ratio_ols, hedge_ratio_robust_huber


@dataclass(frozen=True)
class PairAnalytics:
    hedge_ratio: float
    intercept: float
    r2: float
    last_spread: float
    last_zscore: float
    last_corr: float


RegressionModel = Literal["ols", "robust", "kalman"]


def spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: float, intercept: float = 0.0) -> pd.Series:
    """Spread = A - (intercept + hedge_ratio * B)."""
    df = pd.concat([price_a.rename("a"), price_b.rename("b")], axis=1)
    return df["a"] - (intercept + hedge_ratio * df["b"])


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    s = pd.Series(series)
    if window < 2:
        raise ValueError("window must be >= 2")
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if window < 2:
        raise ValueError("window must be >= 2")
    return df["a"].rolling(window=window, min_periods=window).corr(df["b"])


def ensure_schema(db_path: str) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def fetch_bars_close(
    db_path: str,
    symbol: str,
    interval: str,
    *,
    limit: int = 2000,
) -> pd.DataFrame:
    """Fetch bar closes for a symbol+interval from SQLite.

    Returns DataFrame with columns: ts, close, dt (UTC datetime).
    """
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT ts, close
            FROM bars
            WHERE symbol=? AND interval=?
            ORDER BY ts DESC
            LIMIT ?;
            """,
            (symbol, interval, int(limit)),
        ).fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=["ts", "close"])
    if df.empty:
        return pd.DataFrame(columns=["ts", "close", "dt"])

    df = df.sort_values("ts")
    df["ts"] = df["ts"].astype("int64")
    df["close"] = df["close"].astype("float64")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")
    return df


def compute_pair_analytics(
    close_a: pd.Series,
    close_b: pd.Series,
    *,
    window: int = 120,
    intercept: bool = True,
    model: RegressionModel = "ols",
) -> tuple[pd.DataFrame, dict[str, float], pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute time series analytics for a pair.

    Returns:
    - aligned DataFrame with columns a,b
    - regression dict: {hedge_ratio, intercept, r2}
    - spread series
    - zscore series
    - correlation series
    - alpha series (for plotting)
    - beta series (for plotting)
    """

    df = pd.concat([close_a.rename("a"), close_b.rename("b")], axis=1).dropna()
    if len(df) < max(30, window):
        raise ValueError("not enough data to compute analytics")

    model = str(model).lower().strip()
    if model not in ("ols", "robust", "kalman"):
        raise ValueError("unsupported model; choose one of: ols, robust, kalman")

    if model == "ols":
        ols = hedge_ratio_ols(df["a"], df["b"], intercept=intercept)
        alpha_s = pd.Series(float(ols.intercept), index=df.index, name="alpha")
        beta_s = pd.Series(float(ols.hedge_ratio), index=df.index, name="beta")
        spr = df["a"] - (alpha_s + beta_s * df["b"])
        reg = {"hedge_ratio": float(ols.hedge_ratio), "intercept": float(ols.intercept), "r2": float(ols.r2)}

    elif model == "robust":
        rr: RobustResult = hedge_ratio_robust_huber(df["a"], df["b"], intercept=intercept)
        alpha_s = pd.Series(float(rr.intercept), index=df.index, name="alpha")
        beta_s = pd.Series(float(rr.hedge_ratio), index=df.index, name="beta")
        spr = df["a"] - (alpha_s + beta_s * df["b"])
        reg = {"hedge_ratio": float(rr.hedge_ratio), "intercept": float(rr.intercept), "r2": float(rr.r2)}

    else:
        kf = kalman_filter_regression(df["a"], df["b"], intercept=intercept)
        alpha_s = kf.alpha
        beta_s = kf.beta
        spr = kf.spread

        y = df["a"].to_numpy(dtype=float)
        y_hat = (alpha_s + beta_s * df["b"]).to_numpy(dtype=float)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else float("nan")

        reg = {
            "hedge_ratio": float(beta_s.iloc[-1]),
            "intercept": float(alpha_s.iloc[-1]),
            "r2": float(r2),
        }

    z = rolling_zscore(spr, window=window)
    c = rolling_correlation(df["a"], df["b"], window=window)

    return df, reg, spr.rename("spread"), z.rename("zscore"), c.rename("corr"), alpha_s, beta_s


def compute_pair_snapshot_from_db(
    *,
    db_path: str,
    symbol_a: str,
    symbol_b: str,
    interval: str,
    window: int,
    regression_intercept: bool = True,
    model: RegressionModel = "ols",
    limit: int = 2000,
) -> PairAnalytics:
    a = fetch_bars_close(db_path, symbol_a, interval, limit=limit)
    b = fetch_bars_close(db_path, symbol_b, interval, limit=limit)

    df, reg, spr, z, c, _alpha, _beta = compute_pair_analytics(
        a["close"],
        b["close"],
        window=window,
        intercept=regression_intercept,
        model=model,
    )

    last_spread = float(spr.dropna().iloc[-1])
    last_z = float(z.dropna().iloc[-1])
    last_c = float(c.dropna().iloc[-1])

    return PairAnalytics(
        hedge_ratio=float(reg["hedge_ratio"]),
        intercept=float(reg["intercept"]),
        r2=float(reg["r2"]),
        last_spread=last_spread,
        last_zscore=last_z,
        last_corr=last_c,
    )
