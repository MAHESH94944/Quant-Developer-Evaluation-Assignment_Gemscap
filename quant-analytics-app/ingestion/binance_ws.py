from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from db.connection import connect
from db.schema import SCHEMA_SQL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WsTick:
    symbol: str
    ts: int  # epoch ms
    price: float
    size: float


def _ensure_schema(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s:
        raise ValueError("symbol empty")
    return s


def _combined_stream_url(symbols: list[str]) -> str:
    # Binance combined streams: wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade
    streams = "/".join([f"{_normalize_symbol(s).lower()}@trade" for s in symbols])
    return f"wss://stream.binance.com:9443/stream?streams={streams}"


def _parse_trade_message(msg: dict) -> WsTick:
    # Combined stream: {"stream": "btcusdt@trade", "data": {...}}
    data = msg.get("data") if isinstance(msg, dict) else None
    if not isinstance(data, dict):
        raise ValueError("invalid message")

    symbol = _normalize_symbol(data.get("s"))
    ts = int(data.get("E"))  # event time (ms)
    price = float(data.get("p"))
    size = float(data.get("q"))

    if ts <= 0:
        raise ValueError("invalid ts")
    if not (price > 0.0 and size > 0.0):
        raise ValueError("invalid price/size")

    return WsTick(symbol=symbol, ts=ts, price=price, size=size)


def _insert_ticks(db_path: str | Path, rows: list[tuple[str, int, float, float, str, int]]) -> int:
    _ensure_schema(db_path)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)

        # Update per-symbol last_ts state for monitoring/resumability.
        max_ts: dict[str, int] = {}
        for sym, ts, *_rest in rows:
            prev = max_ts.get(sym)
            if prev is None or int(ts) > prev:
                max_ts[sym] = int(ts)

        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO ticks(symbol, ts, price, size, src_file, src_line)
            VALUES(?, ?, ?, ?, ?, ?);
            """,
            rows,
        )

        for sym, ts in max_ts.items():
            conn.execute(
                """
                INSERT INTO tick_state(symbol, last_ts)
                VALUES(?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_ts=excluded.last_ts,
                    updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');
                """,
                (str(sym).upper(), int(ts)),
            )

        conn.commit()
        return int(cur.rowcount if cur.rowcount != -1 else 0)
    finally:
        conn.close()


async def ingest_binance_trades(
    *,
    symbols: list[str],
    db_path: str | Path,
    stop_event: Optional[asyncio.Event] = None,
    batch_size: int = 500,
    flush_seconds: float = 0.5,
    max_backoff_seconds: float = 15.0,
) -> None:
    """Continuously ingest Binance `@trade` messages into SQLite.

    This is intended for local demo/dev use (not a production-grade market data stack).

    Writes ticks to table `ticks` as: {symbol, ts(ms), price, size}.
    """

    try:
        import websockets
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: websockets. Install it via requirements.txt") from e

    url = _combined_stream_url(symbols)
    logger.info("Connecting to %s", url)

    backoff = 0.5
    src_line = 0

    while stop_event is None or not stop_event.is_set():
        rows: list[tuple[str, int, float, float, str, int]] = []
        last_flush = time.time()

        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=5) as ws:
                backoff = 0.5
                while stop_event is None or not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        raw = None

                    if raw is not None:
                        src_line += 1
                        try:
                            msg = json.loads(raw)
                            tick = _parse_trade_message(msg)
                            rows.append((tick.symbol, tick.ts, tick.price, tick.size, "binance_ws", src_line))
                        except Exception:
                            # ignore malformed messages
                            continue

                    now = time.time()
                    if rows and (len(rows) >= batch_size or (now - last_flush) >= flush_seconds):
                        inserted = _insert_ticks(db_path, rows)
                        logger.info("WS ingest flush: inserted~%s rows=%s", inserted, len(rows))
                        rows.clear()
                        last_flush = now

        except Exception as e:  # noqa: BLE001
            logger.warning("WS ingest error: %s", e)

        # Backoff and retry
        if stop_event is not None and stop_event.is_set():
            break

        jitter = random.uniform(0.0, 0.25)
        sleep_for = min(max_backoff_seconds, backoff) + jitter
        logger.info("Reconnecting in %.2fs", sleep_for)
        await asyncio.sleep(sleep_for)
        backoff *= 1.8


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest Binance trade ticks via WebSocket into SQLite")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to subscribe to (default: BTCUSDT ETHUSDT)",
    )
    parser.add_argument("--db", default=str(Path("db") / "ticks.db"), help="SQLite db path")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--flush-seconds", type=float, default=0.5)

    args = parser.parse_args()

    stop = asyncio.Event()

    async def _run() -> None:
        await ingest_binance_trades(
            symbols=list(args.symbols),
            db_path=args.db,
            stop_event=stop,
            batch_size=int(args.batch_size),
            flush_seconds=float(args.flush_seconds),
        )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
