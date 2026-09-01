"""
charts.py -> figures for my risk report.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")            # write files; no interactive window needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from data import load_returns
from portfolio import get_weights, portfolio_returns
from var import historical_var, parametric_var, ewma_var, historical_cvar
from backtest import rolling_breaches, WINDOW
from risk import risk_contributions


FIG = Path(__file__).parent / "figures"

# Validated categorical palette. The ORDER is the colourblind-safety mechanism,
# not decoration: these three clear every separation gate as a set. Do not
# reshuffle or add a fourth without re-validating.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT,
    "text.color": INK, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "figure.dpi": 130,
})


def _style(ax, title, ylabel=None):
    """Recessive frame: the data should be the only loud thing on the page."""
    ax.set_title(title, color=INK, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)                       # grid behind the marks
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)       # drop non-informative ink
    if ylabel:
        ax.set_ylabel(ylabel)


def chart_rolling_var(pr, confidence=0.99):
    """The headline figure: how fast each method reacts."""
    fig, ax = plt.subplots(figsize=(11, 5))
    tested = pr.iloc[WINDOW:]

    # Realised returns sit underneath in a recessive grey: context, not a series.
    ax.plot(tested.index, tested * 100, lw=0.7, color="#c9c8c4",
            label="Daily return", zorder=1)

    for name, fn, colour in (("Historical", historical_var, BLUE),
                             ("Parametric", parametric_var, ORANGE),
                             ("EWMA", ewma_var, AQUA)):
        _, fc = rolling_breaches(pr, fn, confidence)
        ax.plot(fc.index, -fc * 100, lw=2, color=colour,
                label=f"{name} VaR", zorder=3)

    _style(ax, f"{confidence:.0%} VaR forecast vs realised return",
           "daily return / VaR threshold (%)")
    ax.legend(frameon=False, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "rolling_var.png")
    plt.close(fig)


def chart_distribution(pr):
    """Where the normal assumption visibly fails."""
    fig, ax = plt.subplots(figsize=(9, 5))
    pct = pr * 100

    ax.hist(pct, bins=80, density=True, color="#cde2fb",
            edgecolor=SURFACE, linewidth=0.5, label="Observed returns")

    x = np.linspace(pct.min(), pct.max(), 400)
    ax.plot(x, stats.norm.pdf(x, pct.mean(), pct.std()), lw=2,
            color=BLUE, label="Fitted normal")

    v, cv = historical_var(pr, 0.99) * 100, historical_cvar(pr, 0.99) * 100
    top = ax.get_ylim()[1]
    # Stagger heights and right-align into the empty space left of each line.
    for value, colour, label, y in ((-v, ORANGE, f"99% VaR {v:.2f}%", 0.62),
                                    (-cv, AQUA, f"99% CVaR {cv:.2f}%", 0.50)):
        ax.axvline(value, color=colour, lw=2, ls="--")
        ax.text(value - 0.3, top * y, label, color=INK_SOFT, fontsize=9,
                ha="right", va="center")


    _style(ax, "Daily return distribution vs a fitted normal", "density")
    ax.set_xlabel("daily return (%)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "distribution.png")
    plt.close(fig)


def chart_correlation(r):
    """Why the covariance matrix is ill-conditioned."""
    corr = r.corr()
    order = corr.mean().sort_values(ascending=False).index   # cluster the twins
    corr = corr.loc[order, order]

    # Every pair here is positively correlated (min 0.014), so this is
    # magnitude, not polarity. A sequential ramp uses the full scale; a
    # diverging one wastes half of it on negatives that never happen.
    cmap = LinearSegmentedColormap.from_list(
        "corr", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr.values, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    ax.set_title("Correlation of daily returns", color=INK, loc="left",
                 pad=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlation")
    fig.tight_layout()
    fig.savefig(FIG / "correlation.png")
    plt.close(fig)


def chart_backtest(pr):
    """Observed breach rate against the rate each model promised."""
    methods = (("Historical", historical_var, BLUE),
               ("Parametric", parametric_var, ORANGE),
               ("EWMA", ewma_var, AQUA))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, c in zip(axes, (0.95, 0.99)):
        rates, colours, names = [], [], []
        for name, fn, colour in methods:
            br, _ = rolling_breaches(pr, fn, c)
            rates.append(br.mean() * 100)
            colours.append(colour)
            names.append(name)

        bars = ax.bar(names, rates, color=colours, width=0.6)
        ax.axhline((1 - c) * 100, color=INK_SOFT, lw=1.5, ls="--")
        ax.text(2.45, (1 - c) * 100, f" target {(1-c)*100:.0f}%",
                color=INK_SOFT, fontsize=9, va="center")

        # Value labels: identity never rests on colour alone.
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.04,
                    f"{rate:.2f}%", ha="center", color=INK_SOFT, fontsize=9)

        _style(ax, f"{c:.0%} confidence", "observed breach rate (%)")

    fig.tight_layout()
    fig.savefig(FIG / "backtest.png")
    plt.close(fig)


def chart_risk_contribution(r, w, top_n=12):
    """Money share against risk share: where size and risk disagree."""
    rc = risk_contributions(r, w).head(top_n)
    y = np.arange(len(rc))
    h = 0.38                       # leaves a small gap between paired bars

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - h / 2, rc["weight_pct"], height=h, color=BLUE,
            label="Share of money")
    ax.barh(y + h / 2, rc["risk_pct"], height=h, color=ORANGE,
            label="Share of risk")

    ax.set_yticks(y, rc.index)
    ax.invert_yaxis()              # largest contributor on top
    _style(ax, "Share of money vs share of risk", None)
    ax.set_xlabel("percent of portfolio")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "risk_contribution.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    r = load_returns()
    pr = portfolio_returns(r, get_weights(r))

    chart_rolling_var(pr)
    chart_distribution(pr)
    chart_correlation(r)
    chart_backtest(pr)
    chart_risk_contribution(r, get_weights(r))
    print(f"wrote 5 figures to {FIG}")