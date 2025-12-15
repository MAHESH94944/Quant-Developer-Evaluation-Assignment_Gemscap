from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from analytics.resample import list_symbols as db_list_symbols
from analytics.resample import resample_symbol
from ingestion.loader import ingest_ndjson


def _run_resample_loop(
    *,
    db_path: str,
    intervals: list[str],
    poll_seconds: float,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            symbols = db_list_symbols(db_path)
            for s in symbols:
                for interval in intervals:
                    resample_symbol(db_path, s, interval)
        except Exception:
            pass
        stop_event.wait(poll_seconds)


def _run_background_pipeline(
    *,
    ndjson_path: str,
    db_path: str,
    intervals: list[str],
    poll_seconds: float,
    stop_event: threading.Event,
) -> None:
    # Best-effort loop: ingest new ticks then resample.
    while not stop_event.is_set():
        try:
            ingest_ndjson(ndjson_path, db_path, resume=True)
            symbols = db_list_symbols(db_path)
            for s in symbols:
                for interval in intervals:
                    resample_symbol(db_path, s, interval)
        except Exception:
            # Keep loop alive; UI will show "no data" until pipeline succeeds.
            pass

        stop_event.wait(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-command runner for local quant analytics")
    parser.add_argument("--db", default=str(Path("db") / "ticks.db"), help="SQLite DB path")
    parser.add_argument("--ndjson", default=None, help="Optional NDJSON file path to ingest continuously")
    parser.add_argument(
        "--ws",
        action="store_true",
        help="Ingest live ticks from Binance WebSocket (@trade) for selected symbols",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["BTCUSDT", "ETHUSDT"],
        help="Symbols for WS ingestion (default: BTCUSDT ETHUSDT)",
    )
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--poll", type=float, default=1.0, help="Polling seconds for ingest+resample loop")
    args = parser.parse_args()

    os.environ.setdefault("QADB_PATH", args.db)
    os.environ.setdefault("QA_API_URL", f"http://127.0.0.1:{args.api_port}")

    stop_event = threading.Event()

    bg_thread = None
    if args.ndjson:
        bg_thread = threading.Thread(
            target=_run_background_pipeline,
            kwargs={
                "ndjson_path": args.ndjson,
                "db_path": args.db,
                "intervals": ["1s", "1m", "5m"],
                "poll_seconds": float(args.poll),
                "stop_event": stop_event,
            },
            daemon=True,
        )
        bg_thread.start()

    ws_thread = None
    if args.ws:
        # Run async WS ingest in a dedicated thread.
        from ingestion.binance_ws import ingest_binance_trades
        import asyncio

        def _ws_runner() -> None:
            stop = asyncio.Event()

            async def _run() -> None:
                await ingest_binance_trades(
                    symbols=list(args.symbols),
                    db_path=args.db,
                    stop_event=stop,
                    batch_size=500,
                    flush_seconds=max(0.2, float(args.poll)),
                )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            def _watch_stop() -> None:
                # Bridge threading.Event -> asyncio.Event
                while not stop_event.is_set():
                    time.sleep(0.1)
                loop.call_soon_threadsafe(stop.set)

            threading.Thread(target=_watch_stop, daemon=True).start()
            try:
                loop.run_until_complete(_run())
            finally:
                try:
                    loop.stop()
                finally:
                    loop.close()

        ws_thread = threading.Thread(target=_ws_runner, daemon=True)
        ws_thread.start()

    # If we are ingesting via WS (or user is ingesting externally), keep resampling running.
    resample_thread = None
    if args.ws and not args.ndjson:
        resample_thread = threading.Thread(
            target=_run_resample_loop,
            kwargs={
                "db_path": args.db,
                "intervals": ["1s", "1m", "5m"],
                "poll_seconds": float(args.poll),
                "stop_event": stop_event,
            },
            daemon=True,
        )
        resample_thread.start()

    py = sys.executable

    api_cmd = [
        py,
        "-m",
        "uvicorn",
        "api.routes:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.api_port),
        "--log-level",
        os.environ.get("LOG_LEVEL", "info").lower(),
    ]

    ui_cmd = [
        py,
        "-m",
        "streamlit",
        "run",
        str(Path("frontend") / "dashboard.py"),
        "--server.port",
        str(args.ui_port),
        "--server.headless",
        "true",
    ]

    api_proc = subprocess.Popen(api_cmd, env=os.environ.copy())
    ui_proc = subprocess.Popen(ui_cmd, env=os.environ.copy())

    def _shutdown(*_sig: object) -> None:
        stop_event.set()
        for p in (ui_proc, api_proc):
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass

    try:
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    except Exception:
        # On some Windows setups SIGTERM handling differs; ignore.
        pass

    # Wait for either process to exit
    while True:
        if api_proc.poll() is not None:
            _shutdown()
            return int(api_proc.returncode or 0)
        if ui_proc.poll() is not None:
            _shutdown()
            return int(ui_proc.returncode or 0)
        time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
