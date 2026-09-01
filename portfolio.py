"""
portfolio.py -> collapse 25 asset returns into one portfolio return per day.
"""

import pandas as pd
from data import load_returns, fx_rate

# ---------------------------------------------------------------------------
# Position sizes, in dollars.
# ---------------------------------------------------------------------------
# Replaced these with what I actually hold. They aren't exact to the cent,
# but the RELATIVE sizes are what matter: a position that is 30% of your
# money drives roughly 30% of your risk. Leaving them all equal analyses an
# equally weighted portfolio, which is a valid starting point but is not mine.
# Positions held in US dollars (current market value).
USD_HOLDINGS = {
    "AMZN": 1060, "GOOGL": 826, "META": 1097, "MSFT": 1790, "NVDA": 2519,
    "TSLA": 1830, "QQQ": 2743,  "VOO": 3138,  "SCHD": 1323, "VUG": 1179,
    "VXUS": 1364, "AIQ": 700,  "APLD": 124, "RGTI": 172, "FLNC": 183,
    "ONDS": 1922, "AAOI": 1889, "RKLB": 693, "BE": 418,   "SOFI": 1917,
    "LYFT": 1685,  "ETH-USD": 2458,
}

# Positions held in euros (current market value, in EUR).
# Kept in EUR and converted below, so this file records what I actually own
# instead of a hand-converted number that goes stale the moment FX moves.
EUR_HOLDINGS = {
    "SIVE.ST": 6474, 
    "LPK.DE": 894,
}

TRADING_DAYS = 252


def build_holdings() -> dict:
    """Every positino expressed in USD, at the rate on the sample end date."""
    rate = fx_rate("EURUSD=X")
    holdings = dict(USD_HOLDINGS)
    for ticker, eur in EUR_HOLDINGS.items():
        holdings[ticker] = eur * rate
    return holdings


def get_weights(returns: pd.DataFrame) -> pd.Series:
    """Position sizes converted to fractions of the portfolio, summing to 1."""
    w = pd.Series(build_holdings(), dtype=float)

    # A typo in HOLDINGS would silently drop an asset or invent one. Fail loudly
    # instead: label mismatches are the most common silent bug in this step.
    missing = set(returns.columns) - set(w.index)
    extra = set(w.index) - set(returns.columns)
    if missing or extra:
        raise ValueError(f"HOLDINGS mismatch. missing={sorted(missing)} extra={sorted(extra)}")

    w = w / w.sum()                     # normalizing so the weights summ to 1
    return w.reindex(returns.columns)   # align order with the returns columns


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """One number per day: what the whole portfolio did"""
    # Weighted sum across assets. This assumes weights stay constant, i.e. the
    # portfolio is rebalanced back to target every day. The assumption is stated
    # in the README.
    return (returns * weights).sum(axis=1)


if __name__ == "__main__":
    r = load_returns()
    w = get_weights(r)
    pr = portfolio_returns(r, w)

    ann = pr.std() * (TRADING_DAYS ** 0.5) * 100
    avg_single = (r.std() * (TRADING_DAYS ** 0.5)).mean() * 100

    print(f"days                     : {len(pr)}")
    print(f"weights sum              : {w.sum():.6f}")
    print(f"portfolio annual vol     : {ann:.1f}%")
    print(f"average single-asset vol : {avg_single:.1f}%")
    print(f"worst day                : {pr.min()*100:.2f}%  on {pr.idxmin().date()}")
    print(f"best day                 : {pr.max()*100:.2f}%  on {pr.idxmax().date()}")