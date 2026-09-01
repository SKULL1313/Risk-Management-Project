"""
risk.py -> where my portfolio's risk actually sits.
"""

import numpy as np
import pandas as pd

from data import load_returns
from portfolio import get_weights

TRADING_DAYS = 252


def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Split total portfolio volatility across positions.

    A position's contribution reflects its size, its own volatility, AND how it
    moves with the rest of the book. The components sum exactly to portfolio
    volatility, so they can be read as percentage shares.
    """
    cov = returns.cov().values
    w = weights.values

    port_vol = np.sqrt(w @ cov @ w)

    marginal = cov @ w / port_vol      # d(portfolio vol) / d(weight in asset i)
    component = w * marginal           # sums to port_vol

    out = pd.DataFrame({
        "weight_pct": w * 100,
        "risk_pct": component / port_vol * 100,
        "own_vol_annual": returns.std().values * np.sqrt(TRADING_DAYS) * 100,
    }, index=returns.columns)
    out["risk_per_weight"] = out["risk_pct"] / out["weight_pct"]
    return out.sort_values("risk_pct", ascending=False)


def diversification_ratio(returns: pd.DataFrame, weights: pd.Series) -> float:
    """Weighted average of individual vols over portfolio vol. 1.0 = no benefit."""
    w, vols, cov = weights.values, returns.std().values, returns.cov().values
    return float((w @ vols) / np.sqrt(w @ cov @ w))


def effective_bets(returns: pd.DataFrame, weights: pd.Series) -> float:
    """
    Meucci's effective number of bets.

    Rotate onto the uncorrelated principal components, measure how variance is
    spread across them, and exponentiate the entropy of that spread.
    """
    cov = returns.cov().values
    eigvals, eigvecs = np.linalg.eigh(cov)

    w_pc = eigvecs.T @ weights.values     # the portfolio, seen in factor space
    var_pc = (w_pc ** 2) * eigvals        # variance carried by each factor
    p = var_pc / var_pc.sum()
    p = p[p > 1e-12]                      # guard log(0)
    return float(np.exp(-np.sum(p * np.log(p))))


def effective_positions(weights: pd.Series) -> float:
    """Concentration of weights alone, ignoring correlation (inverse Herfindahl)."""
    return float(1.0 / np.sum(weights.values ** 2))


if __name__ == "__main__":
    r = load_returns()
    w = get_weights(r)
    rc = risk_contributions(r, w)

    print(f"{'asset':<10}{'weight':>9}{'risk':>9}{'own vol':>10}{'risk/wt':>9}")
    for name, row in rc.iterrows():
        print(f"{name:<10}{row.weight_pct:>8.2f}%{row.risk_pct:>8.2f}%"
              f"{row.own_vol_annual:>9.1f}%{row.risk_per_weight:>9.2f}")

    print()
    print(f"positions held               : {len(w)}")
    print(f"effective positions (weights): {effective_positions(w):.1f}")
    print(f"effective BETS (risk)        : {effective_bets(r, w):.1f}")
    print(f"diversification ratio        : {diversification_ratio(r, w):.2f}")