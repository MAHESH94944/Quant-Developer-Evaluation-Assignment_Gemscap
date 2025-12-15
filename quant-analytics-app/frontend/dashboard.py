from __future__ import annotations

import os
import time
from dataclasses import asdict
from io import BytesIO
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from alerts.rules import ZScoreRule
from analytics.resample import ensure_schema
from db.connection import connect


def _coerce_dt_to_ts_ms(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, utc=True, errors="coerce")
    if dt.isna().any():
        raise ValueError("Could not parse some datetimes in dt column")
    # ns -> ms
    return (dt.astype("int64") // 1_000_000).astype("int64")


def _upsert_bars_from_df(db_path: str, df: pd.DataFrame, *, symbol: str, interval: str) -> int:
    ensure_schema(db_path)

    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for r in df.itertuples(index=False):
        rows.append(
            (
                str(symbol).upper(),
                str(interval).lower(),
                int(r.ts),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(getattr(r, "volume", 0.0) or 0.0),
                int(getattr(r, "trades", 0) or 0),
                float(getattr(r, "vwap", 0.0)) if getattr(r, "vwap", None) is not None else None,
            )
        )

    conn = connect(db_path)
    try:
        cur = conn.executemany(
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
        conn.commit()
        return int(cur.rowcount if cur.rowcount != -1 else len(rows))
    finally:
        conn.close()


def _api_base() -> str:
    return os.environ.get("QA_API_URL", "http://127.0.0.1:8000")


def _get_json(path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    url = _api_base().rstrip("/") + path
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _post_json(path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    url = _api_base().rstrip("/") + path
    r = requests.post(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _get_bytes(path: str, params: Optional[dict[str, Any]] = None) -> bytes:
    url = _api_base().rstrip("/") + path
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.content


def _line_chart(df: pd.DataFrame, x: str, ys: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for y in ys:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=320)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    return fig


def main() -> None:
    st.set_page_config(page_title="Quant Analytics Dashboard", layout="wide")

    st.title("Gemscap Quant Analytics (Local)")

    with st.sidebar:
        st.header("Controls")
        db_path = st.text_input("SQLite DB path", value=os.environ.get("QADB_PATH", "db/ticks.db"))
        # Default to 1s so most sample datasets have enough bars for analytics.
        interval = st.selectbox("Timeframe", options=["1s", "1m", "5m"], index=0)
        window = st.number_input("Rolling window", min_value=10, max_value=5000, value=60, step=10)
        model_label = st.selectbox(
            "Regression model",
            options=["OLS", "Robust (Huber)", "Kalman Filter"],
            index=0,
        )
        model = {"OLS": "ols", "Robust (Huber)": "robust", "Kalman Filter": "kalman"}[model_label]

        regression = st.selectbox("Intercept", options=["On", "Off"], index=0)
        intercept = regression == "On"

        live = st.checkbox("Live refresh", value=True)
        refresh_ms = st.number_input("Refresh (ms)", min_value=250, max_value=5000, value=500, step=250)

        st.divider()
        st.header("Alerts")
        z_thr = st.number_input("Z-score threshold", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        z_mode = st.selectbox("Direction", options=["abs", ">", "<"], index=0)
        rule = ZScoreRule(threshold=float(z_thr), direction=str(z_mode))

        st.divider()
        st.header("Upload OHLC")
        uploaded = st.file_uploader("Upload OHLC bars CSV", type=["csv"], accept_multiple_files=False)
        upload_symbol = st.text_input("Upload symbol", value="BTCUSDT")
        upload_interval = st.selectbox("Upload interval", options=["1s", "1m", "5m"], index=1)
        if uploaded is not None and st.button("Import OHLC CSV into DB"):
            try:
                raw = uploaded.getvalue()
                dfu = pd.read_csv(BytesIO(raw))
                dfu.columns = [c.strip().lower() for c in dfu.columns]

                if "ts" not in dfu.columns:
                    if "dt" in dfu.columns:
                        dfu["ts"] = _coerce_dt_to_ts_ms(dfu["dt"])
                    else:
                        raise ValueError("CSV must contain either ts (epoch ms) or dt (ISO datetime)")

                # Optional columns
                for col in ["volume", "trades", "vwap"]:
                    if col not in dfu.columns:
                        # keep defaults (volume=0, trades=0, vwap=NULL)
                        continue

                inserted = _upsert_bars_from_df(
                    db_path,
                    dfu[[c for c in dfu.columns if c in {"ts","open","high","low","close","volume","trades","vwap"}]].copy(),
                    symbol=upload_symbol,
                    interval=upload_interval,
                )
                st.success(f"Imported ~{inserted} bars into {db_path}")
            except Exception as e:
                st.error(f"Import failed: {e}")

    # periodic refresh (best-effort)
    if live:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=int(refresh_ms), key="qa_autorefresh")
        except Exception:
            st.caption("Install `streamlit-autorefresh` for smoother live refresh.")

    # Load symbols (and surface API health)
    try:
        sym_resp = _get_json("/symbols", params={"db_path": db_path})
        symbols = sym_resp.get("symbols", [])
    except Exception as e:
        st.error(f"API not reachable at {_api_base()}: {e}")
        st.stop()

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Prices")
        selected_symbols = st.multiselect("Symbols", options=symbols, default=symbols[:2])

        if selected_symbols:
            price_df = None
            for s in selected_symbols:
                bars = _get_json("/bars", params={"symbol": s, "interval": interval, "db_path": db_path, "limit": 2000})
                rows = bars.get("rows", [])
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df = df.sort_values("dt")
                df = df[["dt", "close"]].rename(columns={"close": s})
                price_df = df if price_df is None else price_df.merge(df, on="dt", how="outer")

            if price_df is not None and not price_df.empty:
                price_df = price_df.sort_values("dt")
                st.plotly_chart(_line_chart(price_df, "dt", [c for c in price_df.columns if c != "dt"], "Close"), use_container_width=True)

                # CSV download: one symbol at a time (simple)
                dl_symbol = st.selectbox("Download bars CSV", options=selected_symbols)
                csv_bytes = _get_bytes("/export/bars.csv", params={"symbol": dl_symbol, "interval": interval, "db_path": db_path})
                st.download_button(
                    label="Download bars.csv",
                    data=csv_bytes,
                    file_name=f"{dl_symbol.lower()}_{interval}_bars.csv",
                    mime="text/csv",
                )
            else:
                st.info("No bars found yet. Run ingestion + resampling first.")

    with col_b:
        st.subheader("Pair Analytics")
        if len(symbols) < 2:
            st.info("Need at least 2 symbols in DB to compute pair analytics.")
            st.stop()

        default_a = symbols[0]
        default_b = symbols[1] if len(symbols) > 1 else symbols[0]
        symbol_a = st.selectbox("Symbol A", options=symbols, index=0)
        symbol_b = st.selectbox("Symbol B", options=[s for s in symbols if s != symbol_a] or symbols, index=0)

        # Snapshot
        try:
            snap = _get_json(
                "/analytics/snapshot",
                params={
                    "symbol_a": symbol_a,
                    "symbol_b": symbol_b,
                    "interval": interval,
                    "window": int(window),
                    "intercept": bool(intercept),
                    "model": model,
                    "db_path": db_path,
                },
            )
        except requests.HTTPError as e:
            st.warning(
                "Analytics not ready yet (usually not enough bars for the selected window). "
                "Try a smaller window or switch timeframe to 1s.\n\n"
                f"Details: {e.response.text if e.response is not None else e}"
            )
            snap = None

        if snap is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hedge Ratio", f"{snap['regression']['hedge_ratio']:.6f}")
            m2.metric("R²", f"{snap['regression']['r2']:.4f}")
            m3.metric("Spread", f"{snap['latest']['spread']:.6f}")
            m4.metric("Z-score", f"{snap['latest']['zscore']:.3f}")

            if rule.triggered(float(snap["latest"]["zscore"])):
                st.error(f"Alert: z-score {snap['latest']['zscore']:.3f} triggered rule {asdict(rule)}")

        # Series
        try:
            series = _get_json(
                "/analytics/series",
                params={
                    "symbol_a": symbol_a,
                    "symbol_b": symbol_b,
                    "interval": interval,
                    "window": int(window),
                    "intercept": bool(intercept),
                    "model": model,
                    "db_path": db_path,
                    "limit": 2000,
                },
            )
        except requests.HTTPError as e:
            st.warning(f"Series not ready: {e.response.text if e.response is not None else e}")
            return

        sdf = pd.DataFrame(series.get("series", []))
        if sdf.empty:
            st.info("No analytics series available yet.")
            st.stop()

        sdf["dt"] = pd.to_datetime(sdf["dt"], utc=True)

        # Dynamic hedge ratio chart (beta over time). For OLS/Robust this will be a flat line.
        if "beta" in sdf.columns:
            st.plotly_chart(_line_chart(sdf, "dt", ["beta"], "Hedge Ratio (β)"), use_container_width=True)

        st.plotly_chart(_line_chart(sdf, "dt", ["spread"], "Spread"), use_container_width=True)
        st.plotly_chart(_line_chart(sdf, "dt", ["zscore"], "Z-score"), use_container_width=True)
        st.plotly_chart(_line_chart(sdf, "dt", ["corr"], "Rolling Correlation"), use_container_width=True)

        # Manual ADF
        if st.button("Run ADF test (manual)"):
            try:
                adf = _post_json(
                    "/analytics/adf",
                    params={
                        "symbol_a": symbol_a,
                        "symbol_b": symbol_b,
                        "interval": interval,
                        "window": int(window),
                        "intercept": bool(intercept),
                        "model": model,
                        "db_path": db_path,
                    },
                )
                st.json(adf["adf"], expanded=True)
            except requests.HTTPError as e:
                st.error(f"ADF failed: {e.response.text if e.response is not None else e}")

        st.divider()
        st.subheader("Mini Mean-Reversion Backtest")
        z_entry = st.number_input("Entry threshold |z|", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        if st.button("Run backtest"):
            try:
                bt = _post_json(
                    "/backtest",
                    params={
                        "symbol_a": symbol_a,
                        "symbol_b": symbol_b,
                        "interval": interval,
                        "window": int(window),
                        "intercept": bool(intercept),
                        "model": model,
                        "z_entry": float(z_entry),
                        "db_path": db_path,
                        "limit": 5000,
                    },
                )

                m = bt.get("metrics", {})
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total P&L", f"{m.get('total_pnl', 0.0):.4f}")
                c2.metric("# Trades", f"{int(m.get('num_trades', 0))}")
                c3.metric("Hit Ratio", f"{float(m.get('hit_ratio', 0.0)):.2%}" if m.get("num_trades", 0) else "n/a")
                c4.metric("Max Drawdown", f"{float(m.get('max_drawdown', 0.0)):.4f}")
                c5.metric("Annualized P&L", f"{float(m.get('annualized_pnl', 0.0)):.2f}")

                eq = pd.DataFrame(bt.get("equity", []))
                if not eq.empty:
                    eq["dt"] = pd.to_datetime(eq["dt"], utc=True)
                    st.plotly_chart(_line_chart(eq, "dt", ["equity"], "Backtest Equity (Cumulative P&L)"), use_container_width=True)

                trades = pd.DataFrame(bt.get("trades", []))
                if not trades.empty:
                    trades["entry_dt"] = pd.to_datetime(trades["entry_dt"], utc=True)
                    trades["exit_dt"] = pd.to_datetime(trades["exit_dt"], utc=True)
                    st.dataframe(trades, use_container_width=True)

                    # Download trades
                    buf = BytesIO()
                    trades.to_csv(buf, index=False)
                    st.download_button(
                        label="Download backtest_trades.csv",
                        data=buf.getvalue(),
                        file_name=f"backtest_trades_{symbol_a.lower()}_{symbol_b.lower()}_{interval}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No trades generated for the chosen parameters.")
            except requests.HTTPError as e:
                st.error(f"Backtest failed: {e.response.text if e.response is not None else e}")

        # Analytics CSV
        csv_buf = BytesIO()
        out_csv = sdf[["dt", "a", "b", "spread", "zscore", "corr"]].copy()
        out_csv.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download analytics.csv",
            data=csv_buf.getvalue(),
            file_name=f"analytics_{symbol_a.lower()}_{symbol_b.lower()}_{interval}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
