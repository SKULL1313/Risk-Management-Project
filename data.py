"""
data.py -> download prices, convert to USD, produce daily returns. 

Every other script in this project imports from here, so the portfolio is defined in exactly one place.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Portfolio definition
# ---------------------------------------------------------------------------
# FLY is deliberately excluded: it IPO'd 2025-08-07 and has only ~267 days of
# history. I can only use dates where EVERY asset has a price, so keeping it
# would truncate the whole sample to 267 rows and leave ~17 days to backtest
# on, far too few for the Kupiec test to say anything meaningful.

TICKERS = [ "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "QQQ", "VOO", "SCHD", "VUG", "VXUS", "AIQ", "APLD", 
           "RGTI", "FLNC", "ONDS", "AAOI", "RKLB", "BE", "SOFI", "LYFT", "ETH-USD", 
           "SIVE.ST", #Sivers Semiconductors AB (Frankfurt, priced in EUR)
           "LPK.DE", # LPKF Laser & Electronics (XETRA, priced in EUR)
]

# Foreign-listed holdings, and the FX series that converts each to USD.
# SIVE.ST (Stockholm) is used instead of the Frankfurt listing 2DG.F: Frankfurt
# reported the Nov 2024 crash a full day late and overshot the Mar 2026 move
# (+107% vs +74%). A one-day lag would destroy this asset's correlation with
# the rest of the book, which matters more than the wrong magnitude.
FOREIGN = {
    "SIVE.ST": "SEKUSD=X",   # Sivers Semiconductors    (Stockholm, SEK)
    "LPK.DE":  "EURUSD=X",   # LPKF Laser & Electronics (XETRA, EUR)
}
FX_TICKERS = sorted(set(FOREIGN.values()))

FX = "EURUSD=X"                     #quoted as US dollars per 1 euro
START = "2021-08-30"
END = "2026-08-29"                  # an exclusive upper bound, so the sample ends 2026-08-28.
                                    # this is deliberatlely pinned so today's bar is still moving while
                                    # the market is open, and a partial day would give a return
                                    # that changes on every-rerun.

DATA_DIR = Path(__file__).parent / "data"
PRICE_CACHE = DATA_DIR / "prices.csv"


def download_prices(refresh: bool = False) -> pd.DataFrame:
    """Closing prices for every ticker, plus the EUR/USD rate."""
    if PRICE_CACHE.exists() and not refresh:
        #Reuse the saved copy so results stay reproducible between runs.
        return pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)

    raw = yf.download(
        TICKERS + FX_TICKERS,
        start = START, 
        end = END, 
        auto_adjust=True,       # corrects for splits and dividends
        progress=False,
    )
    # Passing a LIST gives two levels of columns (field on top, ticker
    # underneath), so select the Close block to flatten it.
    prices = raw["Close"]

    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(PRICE_CACHE)
    return prices


def to_usd(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert every foreign-priced holding into dollars, then drop FX columns."""
    prices = prices.copy()

    for ticker, fx in FOREIGN.items():
        # Forward-fill only. The most recent known rate is the best estimate for a
        # missing day. Back-filling would use tomorrow's rate to price today: which
        # is lookahead bias, and it would quietly flatter every later result.
        prices[ticker] = prices[ticker] * prices[fx].ffill()

    return prices.drop(columns=FX_TICKERS)


def fx_rate(fx_ticker: str) -> float:
    """FX rate on the last day of the pinned sample."""
    prices = download_prices()

    return float(prices[fx_ticker].ffill().iloc[-1])


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns, keeping only fully populated days."""
    # Drop incomplete days FIRST, so every remaining row is a real trading day
    # where all assets priced. If we computed returns before this, Monday would
    # be measured against a missing Sunday and come out NaN, silently deleting
    # every Monday in the sample.
    prices = prices.dropna(how="any")

    returns = prices.pct_change(fill_method=None)

    # how="any": a portfolio return needs every asset present. One missing
    # price makes the row unusable, and a stray NaN would corrupt the
    # covariance matrix later without ever raising an error.
    #Only the very first row should be Nan: nothing precedes it.
    return returns.dropna(how="any")


def load_returns(refresh:bool = False) -> pd.DataFrame:
    """The one function the rest of the project calls."""
    return to_returns(to_usd(download_prices(refresh)))


def latest_fx() -> float:
    """EUR/USD on the last day of the pinned sample."""
    prices = download_prices()
    return float(prices[FX].ffill().iloc[-1])


if __name__ == "__main__":
    returns = load_returns()
    print(f"shape      : {returns.shape[0]} days x {returns.shape[1]} assets")
    print(f"date range : {returns.index[0].date()} -> {returns.index[-1].date()}")
    print(f"nulls left : {int(returns.isna().sum().sum())}")
    print()
    print(returns.tail(3).iloc[:, :5].round(4))