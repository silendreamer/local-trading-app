# Local Paper-Trading Research App

A local Python 3.11 research dashboard for paper-trading experiments. It uses Streamlit for the dashboard and Polygon for market data.

This project intentionally does not include live trading. Broker code is paper-trading only and uses Alpaca's paper endpoint when credentials are configured.

This software is for research and education only. It is not financial advice, investment advice, or a recommendation to buy or sell any security.

## Features

- Loads 20 research tickers from `config/tickers.yaml`
- Fetches daily and intraday OHLC data from Polygon
- Runs a long-only moving-average crossover strategy
- Includes a conservative daily swing-trading strategy module
- Backtests the swing strategy across the configured ticker set with slippage and transaction costs
- Compares swing strategy equity against SPY
- Saves backtest reports to `reports/`
- Applies basic position and gross-exposure risk limits
- Enforces hard order caps of 10% per position and 5 open positions
- Backtests with next-day execution assumptions to avoid same-close lookahead
- Previews paper orders without sending anything to a live broker
- Fetches Alpaca paper account balance and positions when paper credentials are configured
- Submits Alpaca paper orders only after an explicit dashboard approval click
- Runs a controlled intraday trade cycle from the `TRADE` tab
- Scans recent 15-minute candles and returns BUY/SELL/HOLD/SKIP decisions
- Validates pre-trade risk before any order submission
- Logs proposed and submitted orders to `logs/orders.log` and `logs/orders.csv`
- Keeps secrets and local settings in `.env`
- Uses small, testable modules under `src/trading_app`

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add Alpaca paper-trading credentials to `.env` when you are ready to connect a paper account:

```dotenv
DRY_RUN=true
AUTO_TRADE=false
POLYGON_API_KEY=your-polygon-key
SNAPSHOT_STORAGE=github
GITHUB_SNAPSHOT_REPO=your-github-user-or-org/local-trading-app
GITHUB_SNAPSHOT_BRANCH=snapshot-data
GITHUB_SNAPSHOT_DIR=snapshots
GITHUB_SNAPSHOT_TOKEN=your-fine-grained-github-token
ALPACA_PAPER=true
ALPACA_API_KEY=your-paper-key
ALPACA_SECRET_KEY=your-paper-secret
```

`DRY_RUN=true` is the default. With dry run enabled, approved dashboard orders are logged but not sent. To send approved orders to Alpaca paper trading, set `DRY_RUN=false` and click the approval button in the dashboard. `AUTO_TRADE=false` is the default; do not set it to `true` unless you understand that eligible paper orders may be submitted without the same manual approval step. Live trading endpoints are not supported.

Safety defaults:

- `DRY_RUN=true`
- `ALPACA_PAPER=true`
- `AUTO_TRADE=false`
- Alpaca client is hard-coded to `https://paper-api.alpaca.markets`
- Manual approval is required before any paper order submission
- Proposed and submitted orders are logged to `logs/orders.log` and `logs/orders.csv`
- `.env` and `.env.*` are ignored by git so API keys are not committed by default
- Risk caps are enforced in code before order generation and again before submission
- The scheduler runs only during US market hours and stops at 4:00 PM America/New_York
- The app rejects stale market data, duplicate same-ticker orders, and BUY orders for already-held tickers
- Current positions use exit logic only; unheld tickers use entry logic only
- Pre-trade validation enforces max 1% account risk per trade, max 5 open positions, and max 20% allocation per ticker

## Intraday Trade Cycle

Use the `TRADE` tab for controlled intraday operation:

1. Click `Start Trading` to start the background scheduler.
2. The scheduler runs a trade cycle every 15 minutes during US market hours.
3. The cycle stops automatically at 4:00 PM America/New_York.
4. Click `Stop Trading` to stop the scheduler manually.
5. Review the status panel, latest signals, current Alpaca positions, proposed trades, risk validation results, and submitted orders.

Each trade cycle:

- Loads configured tickers from `config/tickers.yaml`
- Fetches Alpaca paper account, positions, and open orders when credentials are configured
- Fetches recent 15-minute market data
- Generates BUY/SELL/HOLD/SKIP decisions
- Runs pre-trade risk validation
- Builds limit orders with risk-based BUY sizing
- Logs orders only when `DRY_RUN=true`
- Submits to Alpaca paper only when dry run is disabled and approval or `AUTO_TRADE=true` allows it
- Saves latest cycle CSV outputs under `logs/`

This tool is intentionally paper-first. Keep `DRY_RUN=true` while developing or reviewing strategy behavior.

## Momentum Scanner

Use the `SCANNER` tab to find momentum candidates. The scanner uses Polygon's full-market stock snapshot endpoint to retrieve current market data for all US stock tickers in one request, then filters locally by gap, volume, and price.

Polygon Snapshot scan:

1. Call Polygon `/v2/snapshot/locale/us/markets/stocks/tickers`.
2. Filter by configured gap percentage, volume, and price range.
3. Rank by gap percentage and volume.
4. Return the configured top N movers.

Set `POLYGON_API_KEY` in `.env` before using the Polygon scanner. Polygon snapshot recency depends on your Polygon plan.

### Scanner2 Snapshot Service

`SCANNER2` uses persisted Polygon snapshots for consistent historical analysis and lower API usage. Run the snapshot capture service during the market day:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m trading_app.scanner2.snapshot_service
```

The service captures Polygon full-market snapshots every 15 minutes and stores them under:

```text
data/snapshots/snapshot_YYYYMMDD_HHMM.json
```

The `SCANNER2` tab analyzes those saved snapshots. It does not call the snapshot endpoint per ticker and does not place trades.

On Streamlit Community Cloud, local files can disappear when the app reboots, redeploys, or hibernates. To persist snapshots, set the GitHub snapshot secrets above. The app writes snapshot JSON files to the configured `GITHUB_SNAPSHOT_BRANCH`, which should be different from your deployed app branch so snapshot commits do not redeploy the app.

## Run The Dashboard

```powershell
streamlit run app.py
```

The app runs locally in your browser. Market data comes from Polygon, so `POLYGON_API_KEY` and an internet connection are required when fetching prices or snapshots.

## Run Tests

```powershell
pytest
```

## Project Layout

```text
config/tickers.yaml          20 research tickers
app.py                       Streamlit dashboard
src/trading_app/data.py      Polygon data fetching and normalization
src/trading_app/strategies/strategy.py
src/trading_app/strategies/swing_strategy.py
src/trading_app/backtesting.py
src/trading_app/risk.py
src/trading_app/broker.py    local and Alpaca paper-only broker interfaces
src/trading_app/logging_config.py
reports/                     generated backtest CSV reports
logs/                        proposed and submitted order logs
tests/                       unit-test friendly examples
```

## Notes

- Do not place real secrets in source files. Use `.env`, which is ignored by git.
- The paper broker is deliberately in-memory and resets when the process restarts.
- Alpaca integration is paper-trading only. Never use live Alpaca credentials with this app.
- This app is not financial advice. Review every strategy, order, and risk assumption yourself before using even paper-trading features.
