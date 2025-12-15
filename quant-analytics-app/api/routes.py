from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from analytics.regression import adf_test
from analytics.stats import (
    compute_pair_analytics,
    compute_pair_snapshot_from_db,
    fetch_bars_close,
)
from analytics.stats import RegressionModel
from analytics.backtest import mean_reversion_backtest
from db.connection import connect
from db.schema import SCHEMA_SQL


def _default_db_path() -> str:
    return os.environ.get("QADB_PATH", str(Path("db") / "ticks.db"))


def _ensure_schema(db_path: str) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/symbols")
def symbols(db_path: str = Query(default_factory=_default_db_path)) -> dict[str, Any]:
    _ensure_schema(db_path)
    conn = connect(db_path)
    try:
        tick_syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol;").fetchall()]
        bar_syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol;").fetchall()]
    finally:
        conn.close()

    # union keeping stable order preference: bars first (likely used in UI), then remaining ticks
    seen: set[str] = set()
    out: list[str] = []
    for s in bar_syms + tick_syms:
        s = str(s)
        if s not in seen:
            out.append(s)
            seen.add(s)

    return {"symbols": out, "counts": {"ticks": len(tick_syms), "bars": len(bar_syms)}}


@router.get("/bars")
def get_bars(
    symbol: str,
    interval: Literal["1s", "1m", "5m"],
    db_path: str = Query(default_factory=_default_db_path),
    limit: int = Query(2000, ge=1, le=20000),
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> dict[str, Any]:
    _ensure_schema(db_path)

    where = ["symbol = ?", "interval = ?"]
    params: list[object] = [symbol, interval]

    if start_ts is not None:
        where.append("ts >= ?")
        params.append(int(start_ts))
    if end_ts is not None:
        where.append("ts < ?")
        params.append(int(end_ts))

    sql = (
        "SELECT ts, open, high, low, close, volume, trades, vwap "
        "FROM bars "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY ts DESC "
        "LIMIT ?;"
    )
    params.append(int(limit))

    conn = connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    rows = list(reversed(rows))  # return ascending
    return {
        "symbol": symbol,
        "interval": interval,
        "rows": [dict(r) for r in rows],
    }


@router.get("/analytics/snapshot")
def analytics_snapshot(
    symbol_a: str,
    symbol_b: str,
    interval: Literal["1s", "1m", "5m"],
    window: int = Query(120, ge=10, le=5000),
    intercept: bool = True,
    model: RegressionModel = Query("ols"),
    db_path: str = Query(default_factory=_default_db_path),
) -> dict[str, Any]:
    try:
        snap = compute_pair_snapshot_from_db(
            db_path=db_path,
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            interval=interval,
            window=int(window),
            regression_intercept=bool(intercept),
            model=str(model).lower(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "interval": interval,
        "window": int(window),
        "model": str(model).lower(),
        "regression": {"hedge_ratio": snap.hedge_ratio, "intercept": snap.intercept, "r2": snap.r2},
        "latest": {"spread": snap.last_spread, "zscore": snap.last_zscore, "corr": snap.last_corr},
    }


@router.get("/analytics/series")
def analytics_series(
    symbol_a: str,
    symbol_b: str,
    interval: Literal["1s", "1m", "5m"],
    window: int = Query(120, ge=10, le=5000),
    intercept: bool = True,
    model: RegressionModel = Query("ols"),
    db_path: str = Query(default_factory=_default_db_path),
    limit: int = Query(2000, ge=50, le=20000),
) -> dict[str, Any]:
    a = fetch_bars_close(db_path, symbol_a, interval, limit=int(limit))
    b = fetch_bars_close(db_path, symbol_b, interval, limit=int(limit))

    try:
        df, reg, spr, z, c, alpha_s, beta_s = compute_pair_analytics(
            a["close"],
            b["close"],
            window=int(window),
            intercept=bool(intercept),
            model=str(model).lower(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    out = pd.DataFrame(
        {
            "dt": df.index.astype("datetime64[ns, UTC]"),
            "a": df["a"].astype(float),
            "b": df["b"].astype(float),
            "spread": spr.astype(float),
            "zscore": z.astype(float),
            "corr": c.astype(float),
            "alpha": alpha_s.astype(float),
            "beta": beta_s.astype(float),
        }
    )

    # Serialize datetimes as ISO strings
    out["dt"] = out["dt"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    return {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "interval": interval,
        "window": int(window),
        "model": str(model).lower(),
        "regression": {"hedge_ratio": float(reg["hedge_ratio"]), "intercept": float(reg["intercept"]), "r2": float(reg["r2"])},
        "series": out.to_dict(orient="records"),
    }


@router.post("/analytics/adf")
def analytics_adf(
    symbol_a: str,
    symbol_b: str,
    interval: Literal["1s", "1m", "5m"],
    window: int = Query(120, ge=20, le=10000),
    intercept: bool = True,
    model: RegressionModel = Query("ols"),
    db_path: str = Query(default_factory=_default_db_path),
    limit: int = Query(5000, ge=100, le=20000),
) -> dict[str, Any]:
    a = fetch_bars_close(db_path, symbol_a, interval, limit=int(limit))
    b = fetch_bars_close(db_path, symbol_b, interval, limit=int(limit))

    try:
        df, reg, spr, z, c, _alpha, _beta = compute_pair_analytics(
            a["close"],
            b["close"],
            window=int(window),
            intercept=bool(intercept),
            model=str(model).lower(),
        )
        res = adf_test(spr.dropna())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "interval": interval,
        "window": int(window),
        "model": str(model).lower(),
        "adf": {
            "adf_stat": res.adf_stat,
            "pvalue": res.pvalue,
            "used_lag": res.used_lag,
            "nobs": res.nobs,
            "critical_values": res.critical_values,
        },
    }


@router.post("/backtest")
def backtest(
    symbol_a: str,
    symbol_b: str,
    interval: Literal["1s", "1m", "5m"],
    window: int = Query(120, ge=10, le=5000),
    intercept: bool = True,
    model: RegressionModel = Query("ols"),
    z_entry: float = Query(2.0, ge=0.1, le=10.0),
    db_path: str = Query(default_factory=_default_db_path),
    limit: int = Query(5000, ge=200, le=20000),
) -> dict[str, Any]:
    a = fetch_bars_close(db_path, symbol_a, interval, limit=int(limit))
    b = fetch_bars_close(db_path, symbol_b, interval, limit=int(limit))

    try:
        df, reg, spr, z, c, alpha_s, beta_s = compute_pair_analytics(
            a["close"],
            b["close"],
            window=int(window),
            intercept=bool(intercept),
            model=str(model).lower(),
        )
        series = pd.DataFrame(
            {
                "dt": df.index.astype("datetime64[ns, UTC]"),
                "spread": spr.astype(float),
                "zscore": z.astype(float),
            }
        )
        trades, equity, metrics = mean_reversion_backtest(series, z_entry=float(z_entry), z_exit=0.0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not trades.empty:
        trades_out = trades.copy()
        trades_out["entry_dt"] = pd.to_datetime(trades_out["entry_dt"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        trades_out["exit_dt"] = pd.to_datetime(trades_out["exit_dt"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        trades_payload = trades_out.to_dict(orient="records")
    else:
        trades_payload = []

    equity_out = equity.copy()
    if not equity_out.empty:
        equity_out["dt"] = pd.to_datetime(equity_out["dt"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    equity_payload = equity_out.to_dict(orient="records")

    return {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "interval": interval,
        "window": int(window),
        "model": str(model).lower(),
        "regression": reg,
        "metrics": {
            "total_pnl": metrics.total_pnl,
            "num_trades": metrics.num_trades,
            "hit_ratio": metrics.hit_ratio,
            "avg_pnl": metrics.avg_pnl,
            "max_drawdown": metrics.max_drawdown,
            "annualized_pnl": metrics.annualized_pnl,
        },
        "equity": equity_payload,
        "trades": trades_payload,
    }


@router.get("/export/bars.csv")
def export_bars_csv(
    symbol: str,
    interval: Literal["1s", "1m", "5m"],
    db_path: str = Query(default_factory=_default_db_path),
) -> StreamingResponse:
    _ensure_schema(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT ts, open, high, low, close, volume, trades, vwap
            FROM bars
            WHERE symbol=? AND interval=?
            ORDER BY ts ASC;
            """,
            (symbol, interval),
        ).fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "trades", "vwap"])
    if not df.empty:
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    buf = StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    filename = f"{symbol.lower()}_{interval}_bars.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Quant Analytics API", version="0.1.0")
    app.include_router(router)
    return app
