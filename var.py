"""
var.py -> Value at Risk estimates for the portfolio.
"""

import numpy as np
import pandas as pd

from data import load_returns
from portfolio import get_weights, portfolio_returns, build_holdings
from scipy import stats

LAMBDA = 0.94


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the loss level exceeded only (1 - confidence) of the time
    in the observed sample.

    Returned as a POSITIVE number meaning a loss, which is the market
    convention: "95% VaR of 3.4%" describes a 3.4% loss, not a gain.
    """
    # At 95% confidence we want the 5th percentile of the return distribution:
    # the point where 5% of observed days were worse.
    cutoff = np.percentile(returns, (1.0 - confidence) * 100.0)

    return -cutoff


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (variance-covariance) VaR.

    Assumes daily returns are normally distributed, so mean and standard
    deviation alone describe the whole distribution, tail included.
    """
    mu = returns.mean()
    sigma = returns.std()
    # z is the point on a standard bell curve with (1 - confidence) of the area
    # to its left: -1.645 at 95%, -2.326 at 99%
    z = stats.norm.ppf(1.0 - confidence)

    return -(mu + z * sigma)


def normality_report(returns: pd.Series) -> None:
    """Measure whether the bell-curve assumption is defensible, not assume it."""
    skew = stats.skew(returns)
    ex_kurt = stats.kurtosis(returns)           # 0 for a true normal
    jb_stat, jb_p = stats.jarque_bera(returns)

    print(f"skew            : {skew:+.2f}   (0 = symmetric)")
    print(f"excess kurtosis : {ex_kurt:+.2f}   (0 = normal tails)")
    print(f"Jarque-Bera p   : {jb_p:.2e}  "
          f"({'reject' if jb_p < 0.05 else 'cannot reject'} normality)")


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR, aka expected shortfall: the average loss on days
    that breached VaR. VaR marks where the bad zone starts; CVaR measures how
    deep it goes.
    """
    var = historical_var(returns, confidence)
    tail = returns[returns < -var]        # only the days worse than the cutoff
    if len(tail) == 0:
        return var
    
    return -tail.mean()


def parametric_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall under the normal assumption (closed form)."""
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)

    # Mean of a normal distribution below its alpha-quantile.
    return -(mu - sigma * stats.norm.pdf(z) / alpha)


def ewma_var(returns: pd.Series, confidence: float = 0.95,
             lam: float = LAMBDA) -> float:
    """
    Parametric VaR using an exponentially weighted volatility estimate.

    Identical to parametric_var except for how sigma is measured: recent days
    carry more weight, so the estimate widens within days of a volatility
    spike instead of taking months to notice.
    """
    r = returns.values
    n = len(r)

    # Weights decay into the past. Index 0 is the oldest observation and gets
    # lam**(n-1); the most recent gets lam**0 = 1. Then normalise to sum to 1.
    weights = lam ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()

    mu = returns.mean()                        # drift estimated as before
    sigma = np.sqrt(np.sum(weights * (r - mu) ** 2))

    z = stats.norm.ppf(1.0 - confidence)
    return -(mu + z * sigma)


if __name__ == "__main__":
    r = load_returns()
    w = get_weights(r)
    pr = portfolio_returns(r, w)
    value = sum(build_holdings().values())

    print(f"portfolio value : ${value:,.0f}")
    print(f"sample          : {len(pr)} days, "
          f"{pr.index[0].date()} to {pr.index[-1].date()}")
    print()

    for c in (0.95, 0.99, 0.995):
        v = historical_var(pr, c)
        # How many real observations sit beyond the cutoff? This is the honest
        # measure of how much evidence the number actually rests on.
        breaches = int((pr < -v).sum())
        print(f"{c:.1%} historical VaR : {v*100:5.2f}%  = ${v*value:,.0f}   "
              f"({breaches} days worse, out of {len(pr)})")

    # comparison table
    print()
    print(f"{'conf':>5}  {'hist VaR':>9}  {'par VaR':>9}  {'hist CVaR':>10}  {'par CVaR':>9}")
    for c in (0.95, 0.99, 0.995):
        print(f"{c:>5.1%}  {historical_var(pr, c)*100:>8.2f}%  "
              f"{parametric_var(pr, c)*100:>8.2f}%  "
              f"{historical_cvar(pr, c)*100:>9.2f}%  "
              f"{parametric_cvar(pr, c)*100:>8.2f}%")

    print()
    normality_report(pr)