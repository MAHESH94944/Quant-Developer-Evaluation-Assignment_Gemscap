from __future__ import annotations

import argparse
import json
import logging
import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from db.connection import connect
from db.schema import SCHEMA_SQL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tick:
    symbol: str
    ts: int  # epoch milliseconds
    price: float
    size: float


def _coerce_ts(value: Any) -> int:
    # Expected: epoch ms int. We also accept seconds (heuristic) and ISO strings.
    if value is None:
        raise ValueError("ts missing")

    if isinstance(value, (int, float)):
        ts = int(value)
        # Heuristic: if looks like seconds since epoch, convert to ms
        if ts < 10_000_000_000:
            ts *= 1000
        return ts

    if isinstance(value, str):
        v = value.strip()
        if v.isdigit():
            return _coerce_ts(int(v))
        # Minimal ISO parsing without extra deps
        # Accepts: 2025-01-01T12:00:00.123Z (or without Z)
        try:
            from datetime import datetime, timezone

            if v.endswith("Z"):
                v = v[:-1]
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"unsupported ts format: {value!r}") from e

    raise ValueError(f"unsupported ts type: {type(value).__name__}")


def _parse_tick(obj: dict[str, Any]) -> Tick:
    missing = [k for k in ("symbol", "ts", "price", "size") if k not in obj]
    if missing:
        raise ValueError(f"missing keys: {missing}")

    symbol = str(obj["symbol"]).strip().upper()
    if not symbol:
        raise ValueError("symbol empty")

    ts = _coerce_ts(obj["ts"])

    try:
        price = float(obj["price"])
        size = float(obj["size"])
    except Exception as e:  # noqa: BLE001
        raise ValueError("price/size not numeric") from e

    # Basic sanity checks: real trades should have positive, finite price/size.
    if not math.isfinite(price) or not math.isfinite(size):
        raise ValueError("price/size not finite")
    if price <= 0.0 or size <= 0.0:
        raise ValueError("price/size must be > 0")

    return Tick(symbol=symbol, ts=ts, price=price, size=size)


def ensure_schema(db_path: str | Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _load_state(conn, file_path: str) -> int:
    row = conn.execute(
        "SELECT byte_offset FROM ingestion_state WHERE file_path=?;",
        (file_path,),
    ).fetchone()
    return int(row[0]) if row else 0


def _save_state(conn, file_path: str, byte_offset: int, mtime_ns: Optional[int]) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_state(file_path, byte_offset, mtime_ns)
        VALUES(?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            byte_offset=excluded.byte_offset,
            mtime_ns=excluded.mtime_ns,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');
        """,
        (file_path, int(byte_offset), mtime_ns),
    )


def _save_tick_state(conn, symbol: str, last_ts: int) -> None:
    conn.execute(
        """
        INSERT INTO tick_state(symbol, last_ts)
        VALUES(?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            last_ts=excluded.last_ts,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');
        """,
        (str(symbol).upper(), int(last_ts)),
    )


def ingest_ndjson(
    ndjson_path: str | Path,
    db_path: str | Path = Path("db") / "ticks.db",
    *,
    resume: bool = True,
    batch_size: int = 5000,
) -> dict[str, int]:
    """Ingest an NDJSON file into SQLite.

    - Reads line-by-line
    - Validates {symbol, ts, price, size}
    - Stores into ticks(symbol, ts, price, size)
    - Maintains byte offset in ingestion_state for incremental ingestion
    """

    ndjson_path = Path(ndjson_path)
    if not ndjson_path.exists():
        raise FileNotFoundError(str(ndjson_path))

    ensure_schema(db_path)

    abs_file = str(ndjson_path.resolve())
    mtime_ns = ndjson_path.stat().st_mtime_ns

    conn = connect(db_path)
    inserted = 0
    skipped = 0
    bad = 0

    try:
        conn.executescript(SCHEMA_SQL)

        start_offset = _load_state(conn, abs_file) if resume else 0

        # If file was truncated, reset offset
        size_now = ndjson_path.stat().st_size
        if start_offset > size_now:
            start_offset = 0

        with ndjson_path.open("r", encoding="utf-8") as f:
            f.seek(start_offset)
            byte_offset = start_offset
            batch: list[tuple[str, int, float, float, str, int]] = []
            batch_max_ts: dict[str, int] = {}
            src_line = 0

            while True:
                pos_before = f.tell()
                line = f.readline()
                if not line:
                    break

                pos_after = f.tell()
                byte_offset = pos_after
                src_line += 1

                # If last line is partial and not valid JSON, don't advance state
                if not line.endswith("\n"):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        byte_offset = pos_before
                        break
                try:
                    obj = json.loads(line)
                    tick = _parse_tick(obj)
                    batch.append((tick.symbol, tick.ts, tick.price, tick.size, abs_file, src_line))
                    prev = batch_max_ts.get(tick.symbol)
                    if prev is None or tick.ts > prev:
                        batch_max_ts[tick.symbol] = tick.ts
                except json.JSONDecodeError:
                    bad += 1
                    continue
                except ValueError:
                    bad += 1
                    continue

                if len(batch) >= batch_size:
                    cur = conn.executemany(
                        """
                        INSERT OR IGNORE INTO ticks(symbol, ts, price, size, src_file, src_line)
                        VALUES(?, ?, ?, ?, ?, ?);
                        """,
                        batch,
                    )
                    # sqlite3 doesn't give per-row ignore count reliably; approximate
                    inserted += cur.rowcount if cur.rowcount != -1 else 0
                    skipped += max(0, len(batch) - (cur.rowcount if cur.rowcount != -1 else 0))
                    batch.clear()
                    for sym, ts in batch_max_ts.items():
                        _save_tick_state(conn, sym, ts)
                    batch_max_ts.clear()
                    _save_state(conn, abs_file, byte_offset, mtime_ns)
                    conn.commit()

            if batch:
                cur = conn.executemany(
                    """
                    INSERT OR IGNORE INTO ticks(symbol, ts, price, size, src_file, src_line)
                    VALUES(?, ?, ?, ?, ?, ?);
                    """,
                    batch,
                )
                inserted += cur.rowcount if cur.rowcount != -1 else 0
                skipped += max(0, len(batch) - (cur.rowcount if cur.rowcount != -1 else 0))
                batch.clear()

            for sym, ts in batch_max_ts.items():
                _save_tick_state(conn, sym, ts)

            _save_state(conn, abs_file, byte_offset, mtime_ns)
            conn.commit()

    finally:
        conn.close()

    return {"inserted": inserted, "skipped": skipped, "bad": bad}


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="NDJSON -> SQLite tick ingestion")
    parser.add_argument("ndjson", help="Path to NDJSON file (Binance ticks)")
    parser.add_argument("--db", default=str(Path("db") / "ticks.db"), help="SQLite db path")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from prior offset")
    parser.add_argument("--batch-size", type=int, default=5000)

    args = parser.parse_args()
    stats = ingest_ndjson(args.ndjson, args.db, resume=not args.no_resume, batch_size=args.batch_size)
    logger.info("Ingestion complete: %s", stats)


if __name__ == "__main__":
    main()
