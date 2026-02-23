"""Standalone multi-timeframe RSI strategy backtest for selected NSE symbols.

This example uses yfinance data directly and is intentionally independent from
Backtesting.py's Strategy API.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ========= SETTINGS =========
capital = 30000
risk_per_trade = 0.05
max_shares = 25
rsi_len = 14
ema_len = 20

# NSE symbols (Yahoo format .NS)
symbols = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "LT.NS",
    "SBIN.NS",
    "ITC.NS",
]

start = "2010-01-01"
end = "2025-01-01"


# ========= INDICATORS =========
def RSI(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ========= BACKTEST =========

def main() -> None:
    equity = capital
    equity_curve: list[float] = []
    trades: list[float] = []

    for symbol in symbols:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True)

        if len(df) < 100:
            continue

        # Daily indicators
        df["RSI_D"] = RSI(df["Close"], rsi_len)
        df["EMA20"] = df["Close"].ewm(span=ema_len).mean()

        # Weekly
        weekly = df["Close"].resample("W-FRI").last()
        weekly_rsi = RSI(weekly, rsi_len)
        df["RSI_W"] = weekly_rsi.reindex(df.index, method="ffill")

        # Monthly
        monthly = df["Close"].resample("M").last()
        monthly_rsi = RSI(monthly, rsi_len)
        df["RSI_M"] = monthly_rsi.reindex(df.index, method="ffill")

        position = 0
        entry_price = 0.0
        shares = 0

        for i in range(50, len(df)):
            row = df.iloc[i]
            date = df.index[i]

            friday = date.weekday() == 4

            # ENTRY
            if position == 0:
                if (
                    friday
                    and row["RSI_D"] > 60
                    and row["RSI_W"] > 58
                    and row["RSI_M"] > 60
                ):
                    risk_amount = equity * risk_per_trade
                    price = row["Close"]

                    shares_risk = int(risk_amount / price)
                    shares = min(shares_risk, max_shares)

                    if shares > 0:
                        position = 1
                        entry_price = price

            # EXIT
            elif position == 1 and row["Close"] < row["EMA20"]:
                exit_price = row["Close"]
                pnl = (exit_price - entry_price) * shares
                equity += pnl
                trades.append(pnl)
                position = 0
                shares = 0

            equity_curve.append(equity)

    if not equity_curve:
        print("No data/trades available for the selected universe and period.")
        return

    # ========= RESULTS =========
    equity_series = pd.Series(equity_curve)

    total_return = (equity - capital) / capital * 100
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else np.inf

    cum_max = equity_series.cummax()
    drawdown = (equity_series - cum_max) / cum_max
    max_dd = drawdown.min() * 100

    print("Total Return %:", round(total_return, 2))
    print("Trades:", len(trades))
    print("Win Rate %:", round(win_rate, 2))
    print("Profit Factor:", round(profit_factor, 2))
    print("Max Drawdown %:", round(max_dd, 2))

    plt.figure()
    plt.plot(equity_series)
    plt.title("Equity Curve")
    plt.show()


if __name__ == "__main__":
    main()
