"""Download one year of daily closing prices for a small sample universe.
Run once:  python fetch_data.py
Writes data/tickers.csv and data/prices.csv (public market data from Yahoo Finance)."""
import pandas as pd
import yfinance as yf  # pip install yfinance

UNIVERSE = {
    "Oil & Gas": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL", "DVN", "BKR"],
    "Automobiles": ["TSLA", "GM", "F", "TM", "HMC", "STLA", "RACE", "RIVN", "LCID", "NIO", "LI", "XPEV"],
    "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "STT", "BK", "SCHW"],
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO"],
    "Healthcare": ["JNJ", "LLY", "UNH", "MRK", "ABBV", "PFE", "TMO", "ABT", "AMGN", "BMY", "GILD", "CVS"],
}

def main():
    rows = [(t, ind) for ind, ts in UNIVERSE.items() for t in ts]
    tickers = pd.DataFrame(rows, columns=["ticker", "industry"])
    px = yf.download(list(tickers.ticker), period="1y", progress=False, auto_adjust=True)["Close"]
    long = px.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close").dropna()
    long = long.rename(columns={"Date": "date"})
    long["date"] = pd.to_datetime(long["date"]).dt.date
    long["close"] = long["close"].round(4)
    tickers.to_csv("data/tickers.csv", index=False)
    long.sort_values(["ticker", "date"]).to_csv("data/prices.csv", index=False)
    print(f"{long.ticker.nunique()} tickers, {len(long)} rows, {long.date.min()} to {long.date.max()}")

if __name__ == "__main__":
    main()
