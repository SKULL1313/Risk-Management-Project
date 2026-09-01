"""
backtest.py -> out-of-sample validation of the VaR estomates (Kupiec POF test)
"""

import numpy as np
import pandas as pd
from scipy import stats

from data import load_returns
from portfolio import get_weights, portfolio_returns
from var import historical_var, parametric_var, ewma_var

WINDOW = 250


def rolling_breaches(pr: pd.Series, method, confidence: float, window: int = WINDOW):
    """
    Walk forward through history. At each step, estimate VaR using ONLY the
    prior `window` days, then test it against the next day. The forecast never
    sees the day it is judged on, which is the whole point.
    """
    breaches, forecasts = [], []

    for i in range(window, len(pr)):
        past = pr.iloc[i - window:i]        # strictly before the test day
        v = method(past, confidence)
        forecasts.append(v)
        breaches.append(bool(pr.iloc[i] < -v))
    idx = pr.index[window:]

    return pd.Series(breaches, index=idx), pd.Series(forecasts, index=idx)


def kupiec_pof(n_obs: int, n_breaches: int, confidence: float):
    """
    Kupiec proportion-of-failures test.

    H0: the true breach rate equals (1 - confidence). A small p-value means the
    observed number of breaches is too unlikely for the model to be calibrated.
    """
    p = 1.0 - confidence
    n, x = n_obs, n_breaches

    if x == 0:
        lr = -2.0 * n * np.log(1 - p)
    else:
        pi = x / n
        lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi))

    return lr, stats.chi2.sf(lr, df=1)


def christoffersen_independence(breaches: pd.Series):
    """
    Christoffersen independence test.

    H0: a breach is no more likely after a breach than after a calm day.
    Clustered breaches mean the model is not adapting to volatility.
    """
    b = breaches.astype(int).values
    prev, curr = b[:-1], b[1:]

    # Count each type of day-to-day transition.
    n00 = int(((prev == 0) & (curr == 0)).sum())   # calm  -> calm
    n01 = int(((prev == 0) & (curr == 1)).sum())   # calm  -> breach
    n10 = int(((prev == 1) & (curr == 0)).sum())   # breach -> calm
    n11 = int(((prev == 1) & (curr == 1)).sum())   # breach -> breach

    # Breach probability conditional on what happened yesterday.
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def xlog(count, p):
        # Convention: 0 * log(0) = 0, which occurs whenever a transition
        # type never happened (commonly n11 at high confidence).
        return count * np.log(p) if count > 0 else 0.0

    lr = -2.0 * (
        xlog(n00 + n10, 1 - pi) + xlog(n01 + n11, pi)
        - xlog(n00, 1 - pi01) - xlog(n01, pi01)
        - xlog(n10, 1 - pi11) - xlog(n11, pi11)
    )
    return lr, stats.chi2.sf(lr, df=1), (n00, n01, n10, n11), pi01, pi11


if __name__ == "__main__":
    r = load_returns()
    pr = portfolio_returns(r, get_weights(r))

    print(f"window {WINDOW} days, {len(pr) - WINDOW} out-of-sample tests\n")
    print(f"{'method':<12}{'conf':>6}{'exp':>7}{'obs':>6}"
          f"{'POF p':>8}{'IND p':>8}{'CC p':>8}  verdict")

    for name, method in (("historical", historical_var),
                         ("parametric", parametric_var), 
                         ("ewma", ewma_var)):
        for c in (0.95, 0.99):
            br, _ = rolling_breaches(pr, method, c)
            n, x = len(br), int(br.sum())

            lr_pof, p_pof = kupiec_pof(n, x, c)
            lr_ind, p_ind, counts, pi01, pi11 = christoffersen_independence(br)

            # Conditional coverage: both hypotheses at once, so 2 dof.
            lr_cc = lr_pof + lr_ind
            p_cc = stats.chi2.sf(lr_cc, df=2)

            verdict = "PASS" if p_cc >= 0.05 else "REJECT"
            print(f"{name:<12}{c:>6.0%}{(1-c)*n:>7.1f}{x:>6d}"
                  f"{p_pof:>8.3f}{p_ind:>8.3f}{p_cc:>8.3f}  {verdict}")
            print(f"{'':<12}  P(breach after calm) {pi01:.2%}   "
                  f"P(breach after breach) {pi11:.2%}   "
                  f"consecutive breaches: {counts[3]}")