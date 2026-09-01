"""
montecarlo.py -> simulate correlated portfolio outcomes.
"""

import numpy as np
import pandas as pd

from data import load_returns
from portfolio import get_weights


def covariance_report(returns: pd.DataFrame) -> None:
    """Check whether the covariance matrix is safe to decompose."""
    cov = returns.cov()
    # eigvalsh, not eigvals: covariance is symmetric, so this is faster and
    # cannot return tiny imaginary parts from floating-point noise.
    eig = np.linalg.eigvalsh(cov)

    print(f"assets             : {cov.shape[0]}")
    print(f"observations       : {len(returns)}")
    print(f"smallest eigenvalue: {eig.min():.3e}")
    print(f"largest eigenvalue : {eig.max():.3e}")
    print(f"condition number   : {eig.max() / eig.min():,.0f}")
    print(f"positive definite  : {bool(eig.min() > 0)}")

    # Which holdings are nearly the same bet?
    corr = returns.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(ascending=False)
    print("\nmost correlated pairs:")
    for (a, b), v in pairs.head(6).items():
        print(f"   {a:9} {b:9} {v:.3f}")


def simulate(returns: pd.DataFrame, weights: pd.Series,
             n_sims: int = 100_000, seed: int = 42) -> np.ndarray:
    """
    Simulate one day of portfolio returns, preserving how assets move together.
    """
    mu = returns.mean().values
    cov = returns.cov().values

    # Cholesky gives L such that L @ L.T == cov. It is the matrix equivalent
    # of a square root, and it is the tool that turns independent random
    # numbers into ones with your portfolio's real correlation structure.
    L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(seed)     # seeded: same numbers every run
    z = rng.standard_normal((n_sims, len(mu)))   # independent noise
    correlated = z @ L.T                          # now they move together
    sims = mu + correlated                        # add each asset's drift back

    return sims @ weights.values          # collapse to one number per scenario


if __name__ == "__main__":
    r = load_returns()
    w = get_weights(r)
    covariance_report(r)

    sims = simulate(r, w)
    actual = (r * w).sum(axis=1)

    print(f"\nsimulations        : {len(sims):,}")
    print(f"simulated daily vol: {sims.std()*100:.3f}%")
    print(f"actual   daily vol : {actual.std()*100:.3f}%")
    print()

    for c in (0.95, 0.99, 0.995):
        v = -np.percentile(sims, (1 - c) * 100)
        print(f"{c:>6.1%} Monte Carlo VaR : {v*100:5.2f}%  = ${v*39618:,.0f}")