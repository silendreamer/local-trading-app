# Local Paper-Trading Research App

A local Python 3.11 research dashboard for paper-trading experiments. It uses Streamlit for the dashboard and yfinance for initial market data.

This project intentionally does not include live trading. Broker code is paper-trading only and uses Alpaca's paper endpoint when credentials are configured.

This software is for research and education only. It is not financial advice, investment advice, or a recommendation to buy or sell any security.

## Features

- Loads 20 research tickers from `config/tickers.yaml`
- Fetches adjusted daily close data from yfinance
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
- Logs proposed and submitted orders to `logs/orders.log`
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
ALPACA_API_KEY=your-paper-key
ALPACA_SECRET_KEY=your-paper-secret
```

`DRY_RUN=true` is the default. With dry run enabled, approved dashboard orders are logged but not sent. To send approved orders to Alpaca paper trading, set `DRY_RUN=false` and click the approval button in the dashboard. Live trading endpoints are not supported.

Safety defaults:

- `DRY_RUN=true`
- `ALPACA_PAPER=true`
- Alpaca client is hard-coded to `https://paper-api.alpaca.markets`
- Manual approval is required before any paper order submission
- Proposed and submitted orders are logged to `logs/orders.log`
- `.env` and `.env.*` are ignored by git so API keys are not committed by default
- Risk caps are enforced in code before order generation and again before submission

## Run The Dashboard

```powershell
streamlit run app.py
```

The app runs locally in your browser. Market data comes from yfinance, so an internet connection is required when fetching prices.

## Run Tests

```powershell
pytest
```

## Project Layout

```text
config/tickers.yaml          20 research tickers
app.py                       Streamlit dashboard
src/trading_app/data.py      yfinance data fetching and normalization
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
