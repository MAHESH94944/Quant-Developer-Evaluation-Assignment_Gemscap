from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from db.connection import connect
from db.schema import SCHEMA_SQL

logger = logging.getLogger(__name__)


_SUPPORTED_INTERVALS = {"1s": "1s", "1m": "1min", "5m": "5min"}


def normalize_interval(interval: str) -> str:
    v = interval.strip().lower()
    if v not in _SUPPORTED_INTERVALS:
        raise ValueError(f"unsupported interval {interval!r}; choose one of {sorted(_SUPPORTED_INTERVALS)}")
    return v


def _interval_to_pandas_rule(interval: str) -> str:
    return _SUPPORTED_INTERVALS[normalize_interval(interval)]


def _ms_for_interval(interval: str) -> int:
    interval = normalize_interval(interval)
    if interval == "1s":
        return 1_000
    if interval == "1m":
        return 60_000
    if interval == "5m":
        return 300_000
    raise ValueError(interval)


def ensure_schema(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def list_symbols(db_path: str | Path) -> list[str]:
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol;").fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def _load_last_bar_ts(conn, symbol: str, interval: str) -> Optional[int]:
    row = conn.execute(
        "SELECT last_ts FROM resample_state WHERE symbol=? AND interval=?;",
        (symbol, interval),
    ).fetchone()
    return int(row[0]) if row else None


def _save_last_bar_ts(conn, symbol: str, interval: str, last_ts: int) -> None:
    conn.execute(
        """
        INSERT INTO resample_state(symbol, interval, last_ts)
        VALUES(?, ?, ?)
        ON CONFLICT(symbol, interval) DO UPDATE SET
            last_ts=excluded.last_ts,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');
        """,
        (symbol, interval, int(last_ts)),
    )


def fetch_ticks_df(
    db_path: str | Path,
    symbol: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch ticks for a symbol from SQLite into a DataFrame.

    Returns columns: ts(ms), price, size. Sorted by ts.
    """
    ensure_schema(db_path)

    where = ["symbol = ?"]
    params: list[object] = [symbol]

    if start_ts is not None:
        where.append("ts >= ?")
        params.append(int(start_ts))
    if end_ts is not None:
        where.append("ts < ?")
        params.append(int(end_ts))

    sql = (
        "SELECT ts, price, size FROM ticks "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY ts ASC;"
    )

    conn = connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=["ts", "price", "size"])  # empty

    df = pd.DataFrame(rows, columns=["ts", "price", "size"])
    df["ts"] = df["ts"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["size"] = df["size"].astype("float64")
    return df


def resample_ohlcv_from_ticks(ticks: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample tick DataFrame into OHLCV bars.

    Input ticks columns: ts(ms), price, size.
    Output bars columns: ts(ms start), open, high, low, close, volume, trades, vwap.
    """
    interval = normalize_interval(interval)
    if ticks.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "trades", "vwap"])

    # Guardrail: ignore invalid ticks (can happen if DB was built before validation).
    ticks = ticks[(ticks["price"] > 0) & (ticks["size"] > 0)].copy()
    if ticks.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "trades", "vwap"])

    rule = _interval_to_pandas_rule(interval)

    dt_index = pd.to_datetime(ticks["ts"], unit="ms", utc=True)
    df = ticks.copy()
    df.index = dt_index

    price = df["price"].astype("float64")
    size = df["size"].astype("float64")

    ohlc = price.resample(rule, label="left", closed="left").ohlc()
    volume = size.resample(rule, label="left", closed="left").sum().rename("volume")
    trades = size.resample(rule, label="left", closed="left").count().rename("trades")

    notional = (price * size).resample(rule, label="left", closed="left").sum().rename("notional")
    out = pd.concat([ohlc, volume, trades, notional], axis=1)

    # Drop empty bars (no trades)
    out = out[out["trades"] > 0]

    out["vwap"] = out["notional"] / out["volume"].where(out["volume"] != 0.0)
    out = out.drop(columns=["notional"])

    # Convert index (UTC) to epoch ms
    # `reset_index()` names the datetime column using the index name.
    # Our index often inherits name='ts' from the source column, so don't assume 'index'.
    out = out.reset_index()
    dt_col = "dt" if "dt" in out.columns else ("index" if "index" in out.columns else out.columns[0])
    out = out.rename(columns={dt_col: "dt"})

    out["ts"] = (out["dt"].astype("int64") // 1_000_000).astype("int64")
    out = out.drop(columns=["dt"])

    # Enforce dtypes
    out["open"] = out["open"].astype("float64")
    out["high"] = out["high"].astype("float64")
    out["low"] = out["low"].astype("float64")
    out["close"] = out["close"].astype("float64")
    out["volume"] = out["volume"].astype("float64")
    out["trades"] = out["trades"].astype("int64")
    out["vwap"] = out["vwap"].astype("float64")

    return out[["ts", "open", "high", "low", "close", "volume", "trades", "vwap"]]


def upsert_bars(
    db_path: str | Path,
    symbol: str,
    interval: str,
    bars: pd.DataFrame,
) -> int:
    """Upsert bars into SQLite. Returns number of rows attempted."""
    interval = normalize_interval(interval)
    if bars.empty:
        return 0

    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)

        rows = [
            (
                symbol,
                interval,
                int(r.ts),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume),
                int(r.trades),
                float(r.vwap) if pd.notna(r.vwap) else None,
            )
            for r in bars.itertuples(index=False)
        ]

        conn.executemany(
            """
            INSERT INTO bars(symbol, interval, ts, open, high, low, close, volume, trades, vwap)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, ts) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                trades=excluded.trades,
                vwap=excluded.vwap,
                updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');
            """,
            rows,
        )

        last_ts = int(bars["ts"].max())
        _save_last_bar_ts(conn, symbol, interval, last_ts)

        conn.commit()
        return len(rows)
    finally:
        conn.close()


def export_bars_to_csv(db_path: str | Path, symbol: str, interval: str, out_dir: str | Path) -> Path:
    interval = normalize_interval(interval)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_schema(db_path)
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

    out_path = out_dir / f"{symbol.lower()}_{interval}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def resample_symbol(
    db_path: str | Path,
    symbol: str,
    interval: str,
    *,
    lookback_bars: int = 2,
) -> dict[str, int]:
    """Resample a single symbol/interval and persist to SQLite.

    Uses resample_state to only recompute from the last bar minus a small lookback.
    """
    interval = normalize_interval(interval)
    ensure_schema(db_path)

    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        last = _load_last_bar_ts(conn, symbol, interval)
    finally:
        conn.close()

    start_ts = None
    if last is not None:
        start_ts = last - lookback_bars * _ms_for_interval(interval)

    ticks = fetch_ticks_df(db_path, symbol, start_ts=start_ts)
    bars = resample_ohlcv_from_ticks(ticks, interval)
    written = upsert_bars(db_path, symbol, interval, bars)

    return {"ticks": int(len(ticks)), "bars": int(len(bars)), "written": int(written)}


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Resample ticks into OHLCV bars (1s/1m/5m)")
    parser.add_argument("--db", default=str(Path("db") / "ticks.db"), help="SQLite db path")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to resample (default: all symbols present in ticks)",
    )
    parser.add_argument(
        "--intervals",
        nargs="*",
        default=["1s", "1m", "5m"],
        help="Intervals to resample (default: 1s 1m 5m)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export bars to data/processed as CSV after resampling",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    symbols = args.symbols or list_symbols(db_path)
    intervals = [normalize_interval(i) for i in args.intervals]

    if not symbols:
        logger.warning("No symbols found in ticks table.")
        return

    for symbol in symbols:
        for interval in intervals:
            stats = resample_symbol(db_path, symbol, interval)
            logger.info("Resampled %s %s: %s", symbol, interval, stats)

            if args.export_csv:
                out_path = export_bars_to_csv(db_path, symbol, interval, Path("data") / "processed")
                logger.info("Exported %s", out_path)


if __name__ == "__main__":
    main()
