"""Data acquisition and quality assessment utilities."""

import pandas as pd
import numpy as np
import yfinance as yf

from .config import CONFIG


def download_stock_data(ticker, start_date, end_date, verbose=True):
    """Download historical stock data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date, end_date : str
        Date range in YYYY-MM-DD format.
    verbose : bool
        Print download status.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with OHLCV data indexed by Date, or None on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        df = df.reset_index()
        df = df.dropna(subset=["Close", "Volume"])
        df = df.set_index("Date")

        if verbose:
            print(f"  Downloaded {ticker}: {len(df)} days")

        return df

    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None


def download_multiple_stocks(tickers, start_date=None, end_date=None, config=None):
    """Download data for multiple stocks.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start_date, end_date : str, optional
        Override config dates.
    config : dict, optional
        Configuration dict (defaults to CONFIG).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to DataFrame.
    """
    if config is None:
        config = CONFIG
    start_date = start_date or config["start_date"]
    end_date = end_date or config["end_date"]

    data = {}
    failed = []

    print(f"Downloading data for {len(tickers)} stocks...")
    for ticker in tickers:
        df = download_stock_data(ticker, start_date, end_date)
        if df is not None:
            data[ticker] = df
        else:
            failed.append(ticker)

    print(f"Successfully downloaded: {len(data)}/{len(tickers)} stocks")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    return data


def assess_data_quality(data_dict):
    """Compute data-quality summary across all stocks.

    Parameters
    ----------
    data_dict : dict[str, pd.DataFrame]

    Returns
    -------
    pd.DataFrame
        One row per ticker with quality metrics.
    """
    rows = []
    for ticker, df in data_dict.items():
        daily_returns = df["Close"].pct_change()
        rows.append(
            {
                "Ticker": ticker,
                "Days": len(df),
                "Date Range (days)": (df.index.max() - df.index.min()).days,
                "Missing (%)": (df.isnull().sum() / len(df) * 100).max(),
                "Avg Price": df["Close"].mean(),
                "Price Std": df["Close"].std(),
                "Avg Return (%)": daily_returns.mean() * 100,
                "Return Std (%)": daily_returns.std() * 100,
                "Extreme Moves (>10%)": (abs(daily_returns) > 0.10).sum(),
            }
        )
    return pd.DataFrame(rows)
