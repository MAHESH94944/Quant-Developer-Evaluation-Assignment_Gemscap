# Gemscap Quant Analytics App (Local)

Local real-time(ish) quantitative analytics dashboard for Binance tick data (live WebSocket) and NDJSON tick captures.

## Setup

```bash
pip install -r requirements.txt
```

Windows note:

- If you are using the repo-level venv at `GemScap/.venv`, run commands from `quant-analytics-app/` using `../.venv/Scripts/python.exe` (as seen in your terminal history).

## Collect Data

Use your provided Binance WebSocket collector (HTML) to produce an NDJSON file with lines like:

```json
{ "symbol": "BTCUSDT", "ts": 1734260000123, "price": 104123.12, "size": 0.0021 }
```

Place it under `data/raw/` (recommended).

## Run (single command)

From this folder:

```bash
python app.py --ndjson data/raw/your_file.ndjson
```

If `python` is not pointing at your venv, use the repo-level venv explicitly:

```bash
../.venv/Scripts/python.exe app.py --ndjson data/raw/your_file.ndjson
```

### Run with live Binance WebSocket ingestion

This subscribes to Binance `@trade` ticks and continuously ingests + resamples:

```bash
python app.py --ws --symbols BTCUSDT ETHUSDT --poll 1.0
```

Repo-level venv version:

```bash
../.venv/Scripts/python.exe app.py --ws --symbols BTCUSDT ETHUSDT --poll 1.0
```

- FastAPI: `http://127.0.0.1:8000`
- Streamlit UI: `http://127.0.0.1:8501`

If you omit `--ndjson`, the UI/API will still run, but you must ingest/resample using your own process.

## Optional CLI utilities

These are useful for one-off runs:

```bash
python -m ingestion.loader data/raw/your_file.ndjson
python -m analytics.resample --db db/ticks.db --export-csv
python -m ingestion.binance_ws --db db/ticks.db --symbols BTCUSDT ETHUSDT
```

Tip: the API also supports selecting the regression model via `model=ols|robust|kalman` on `/analytics/*` and `/backtest`.

## API (FastAPI)

Base URL (default): `http://127.0.0.1:8000`

Endpoints:

- `GET /health`
- `GET /symbols`
- `GET /bars?symbol=...&interval=1s|1m|5m&limit=...`
- `GET /analytics/snapshot?symbol_a=...&symbol_b=...&interval=...&window=...&intercept=true|false&model=ols|robust|kalman`
- `GET /analytics/series?symbol_a=...&symbol_b=...&interval=...&window=...&intercept=true|false&model=ols|robust|kalman&limit=...`
- `POST /analytics/adf?symbol_a=...&symbol_b=...&interval=...&window=...&intercept=true|false&model=ols|robust|kalman`
- `POST /backtest?symbol_a=...&symbol_b=...&interval=...&window=...&intercept=true|false&model=ols|robust|kalman&z_entry=...&limit=...`
- `GET /export/bars.csv?symbol=...&interval=...`

Examples:

```bash
curl "http://127.0.0.1:8000/symbols"

curl "http://127.0.0.1:8000/analytics/series?symbol_a=BTCUSDT&symbol_b=ETHUSDT&interval=1s&window=120&intercept=true&model=kalman&limit=1000"

curl -X POST "http://127.0.0.1:8000/backtest?symbol_a=BTCUSDT&symbol_b=ETHUSDT&interval=1s&window=120&intercept=true&model=robust&z_entry=2.0&limit=5000"
```

## Troubleshooting

- `./.venv/Scripts/python.exe: not found` or exit code `127` (Git Bash): you likely created the venv at the repo root (`GemScap/.venv`), not inside `quant-analytics-app/`. Use `../.venv/Scripts/python.exe ...`.
- `python app.py` fails because it’s using system Python: run via `../.venv/Scripts/python.exe app.py ...`.
- Ports already in use: change ports with `--api-port 8001 --ui-port 8502`.

## Run in background (Git Bash)

If you want to keep the app running while continuing to use the terminal:

```bash
cd "/d/Resume/On Campus Internship/GemScap/quant-analytics-app"
"/d/Resume/On Campus Internship/GemScap/.venv/Scripts/python.exe" app.py --ndjson data/raw/ticks_2025-12-15.ndjson > run.log 2>&1 &
echo "started pid=$!"
```

- Logs will be written to `run.log`.
- Stop it with: `kill <pid>`

## Quick Verification

After starting the app, these should return JSON:

```bash
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/symbols"
```

## Notes

- Resampling intervals supported: `1s`, `1m`, `5m`
- Pair analytics: OLS/Robust/Kalman hedge estimation, spread, rolling z-score, rolling correlation; ADF is manual.

## What’s New / Improvements

This repo was incrementally hardened and extended to be both more **correct** (data validation + resumability) and more **advanced** (robust/dynamic models + backtest).

Correctness + reliability:

- **Invalid tick filtering**: ingestion/resampling drops ticks with non-positive `price`/`size` to prevent pathological bars (e.g., `close=0`) and misleading analytics.
- **DB-backed resumability**: ingestion and resampling maintain progress in SQLite state tables (`ingestion_state`, `resample_state`, and per-symbol `tick_state`).
- **Unified pipeline**: NDJSON ingestion and live WS ingestion both feed the same `ticks` table; resampling and analytics operate consistently on top.

Advanced analytics:

- **Robust regression (Huber)** option (`model=robust`) to reduce the influence of outliers.
- **Kalman filter regression** option (`model=kalman`) to estimate time-varying hedge ratio $\beta_t$.
- **Mini mean-reversion backtest** endpoint + dashboard panel (equity curve + metrics + trades).

UI upgrades:

- **Regression model selector** (OLS / Robust / Kalman).
- **Dynamic hedge ratio chart** (plots $\beta$ over time; dynamic in Kalman mode).
- **Backtest panel** with `z_entry` input and downloadable trade list.

## Advanced Extensions

These are optional “advanced” features included to improve robustness and demonstrate extensibility.

### 1) Dynamic Hedge Estimation (Kalman Filter)

Instead of a single fixed hedge ratio $\beta$ (static OLS), the Kalman model estimates a time-varying $\beta_t$ (and optional intercept $\alpha_t$) for:

$$
y_t \approx \alpha_t + \beta_t x_t
$$

The state $[\alpha_t, \beta_t]$ follows a random walk and is updated online each bar.

How to use:

- In the Streamlit sidebar, set **Regression model** → **Kalman Filter**.
- The dashboard plots **Hedge Ratio (β)** over time and uses the Kalman-implied spread.

### 2) Robust Regression (Huber)

Robust regression reduces the influence of outliers compared to OLS. This app uses Huber’s loss via `statsmodels.RLM`.

How to use:

- In the Streamlit sidebar, set **Regression model** → **Robust (Huber)**.

### 3) Mini Mean-Reversion Backtest

A small demonstration backtest driven by the rolling z-score of the spread:

- Enter when $|z| > z_{entry}$.
- Exit when z crosses back through 0.

How to use:

- In the Streamlit sidebar, open **Mini Mean-Reversion Backtest**.
- Choose `Entry threshold |z|` and click **Run backtest**.
- The UI shows metrics, an equity curve, and a trades table (downloadable as CSV).

## Methodology / Caveats

- **Spread definition**: spread is computed as $y - (\alpha + \beta x)$ where $(\alpha, \beta)$ come from the selected model (OLS / robust Huber / Kalman).
- **Z-score**: rolling z-score is computed on the spread using the selected `window`.
- **Backtest P&L meaning**: the demo backtest treats the spread as the traded series and computes P&L from spread changes while a position is open. It does **not** model execution, fees, slippage, funding, or exchange constraints.
- **Lookahead**: signals are generated from rolling statistics; the intent is “use information available up to the bar”, but this is still a simplified demo.
- **Not investment advice**: this is a technical evaluation project, not a production trading system.

## Design Notes

- **Single source of truth**: SQLite (`ticks` and `bars`) is the canonical store; both API and UI read from it.
- **Resumability / state**: ingestion and resampling keep progress in DB state tables (see `ingestion_state`, `resample_state`, and per-symbol `tick_state`), so long runs can be restarted without losing place.
- **Data quality**: ingestion/resampling filters invalid ticks (non-positive price/size) to avoid pathological bars (e.g., `close=0`) and misleading analytics.
- **Extensibility**: analytics endpoints accept `model=ols|robust|kalman`; the UI surfaces the same switch and plots dynamic $\beta_t$ when applicable.

## OHLC Upload

The Streamlit UI includes an **Upload OHLC** section that can import a CSV into the `bars` table.

Supported columns:

- Required: `ts` (epoch ms) **or** `dt` (ISO datetime), plus `open`, `high`, `low`, `close`
- Optional: `volume`, `trades`, `vwap`

This is optional: the app works without any upload (it will build bars from ticks).

## Deliverables

- Runnable single-command app: `python app.py`
- Architecture diagram source + export:
  - diagrams/architecture.drawio
  - diagrams/architecture.svg

## Project Structure

```text
quant-analytics-app/
  app.py                         # Single-command runner (starts FastAPI + Streamlit; optional ingest/resample loop)
  requirements.txt               # Python dependencies
  README.md                      # Setup + usage + design notes
  run.log                        # Runtime logs (only present if you run with output redirection)

  api/
    routes.py                    # FastAPI app + routes (/bars, /analytics, /export, /backtest)
    __init__.py

  ingestion/
    loader.py                    # NDJSON -> SQLite ticks ingestion (resume offsets, validation)
    binance_ws.py                # Live Binance WebSocket (@trade) -> SQLite ticks ingestion
    __init__.py

  db/
    connection.py                # SQLite connection helpers + pragmas
    schema.py                    # SQL schema for ticks/bars/state tables
    ticks.db                     # Local SQLite database (created/updated at runtime)
    __init__.py

  analytics/
    resample.py                  # Tick -> OHLCV/VWAP resampling (1s/1m/5m) + CSV export
    regression.py                # Hedge ratio (OLS + robust Huber) + ADF test wrapper
    kalman_filter.py             # Dynamic hedge estimation (time-varying alpha/beta)
    backtest.py                  # Mini mean-reversion backtest (z-score entry/exit)
    stats.py                     # Spread/z-score/corr + model selection (ols/robust/kalman)
    __init__.py

  alerts/
    rules.py                     # Simple alert rules (e.g., z-score thresholds)
    __init__.py

  frontend/
    dashboard.py                 # Streamlit UI (controls, Plotly charts, exports, alerts, OHLC upload)
    __init__.py

  data/
    raw/                         # Input data (NDJSON tick captures)
      ticks_2025-12-15.ndjson
    processed/                   # Outputs (resampled OHLCV CSVs)
      btcusdt_1s.csv
      btcusdt_1m.csv
      btcusdt_5m.csv
      ethusdt_1s.csv
      ethusdt_1m.csv
      ethusdt_5m.csv

  diagrams/
    architecture.drawio          # Editable architecture diagram source
    architecture.svg             # Exported diagram for viewing/sharing

  __pycache__/                   # Python bytecode cache (safe to ignore)
```

## AI / ChatGPT usage transparency

AI tools were used to speed up scaffolding, debugging, and documentation.

Typical prompts used:

- “Implement NDJSON ingestion with resume offsets into SQLite.”
- “Implement resampling to OHLCV+VWAP and persist incrementally.”
- “Build a Streamlit + Plotly dashboard that calls a FastAPI backend.”
- “Debug unexpected spikes in spread (check for invalid ticks / bars).”

Model referenced in the assistant: GPT-5.2 (Preview).
