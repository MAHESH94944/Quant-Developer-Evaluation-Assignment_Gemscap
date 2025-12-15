from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,         -- epoch milliseconds
    price REAL NOT NULL,
    size REAL NOT NULL,
    src_file TEXT,
    src_line INTEGER,
    inserted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(symbol, ts, price, size)
);

CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks(symbol, ts);

CREATE TABLE IF NOT EXISTS ingestion_state (
    file_path TEXT PRIMARY KEY,
    byte_offset INTEGER NOT NULL,
    mtime_ns INTEGER,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Last processed tick timestamp per symbol (useful for resumability/monitoring).
CREATE TABLE IF NOT EXISTS tick_state (
    symbol TEXT PRIMARY KEY,
    last_ts INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,     -- e.g. '1s', '1m', '5m'
    ts INTEGER NOT NULL,         -- bar start epoch milliseconds (UTC)
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    trades INTEGER NOT NULL,
    vwap REAL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY(symbol, interval, ts)
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_interval_ts ON bars(symbol, interval, ts);

CREATE TABLE IF NOT EXISTS resample_state (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    last_ts INTEGER NOT NULL,    -- last bar start (ms) written to bars
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY(symbol, interval)
);
"""
