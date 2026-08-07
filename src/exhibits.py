"""Station 3 - fund fact-sheet exhibits: the performance table + the 4 required
figures (growth of $1, drawdown, weights over time, Sharpe barplot). Uses
plotstyle's FT design system throughout, for one consistent visual language
across every exhibit in the report.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src import plotstyle, portfolios

RESULTS_FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {
    "equal_weight": "Equal-Weight",
    "min_variance": "Min-Variance",
    "max_sharpe": "Max-Sharpe",
    "risk_parity": "Risk-Parity",
    "mean_cvar": "Mean-CVaR",
}
# NEUTRAL_COLOR is reused deliberately (not a new arbitrary colour): equal-weight
# IS the naive, no-optimisation baseline in this project's own methodology
# (DeMiguel, Garlappi & Uppal, 2009), so grey-as-baseline is a genuine re-use of
# plotstyle's existing colour semantics, not a collision with it. The other three
# are new categorical colours chosen to stay visually distinct from
# EQUITY_COLOR/CRYPTO_COLOR/ACCENT_COLOR, which keep their own reserved meanings.
METHOD_COLORS = {
    "equal_weight": plotstyle.NEUTRAL_COLOR,
    "min_variance": "#0D7680",
    "max_sharpe": "#593380",
    "risk_parity": "#1E874B",
    "mean_cvar": "#B8860B",
}
UNIVERSES = ["Equity", "Crypto", "Combined"]


# ---------------------------------------------------------------------------
# Performance table (results/tables/performance_metrics.csv - exact filename)
# ---------------------------------------------------------------------------

def build_performance_table(fund_backtests: dict, fund_universe: dict,
                             performance_metrics_fn) -> pd.DataFrame:
    """One row per fund: annualised return/volatility, Sharpe, max drawdown.

    `fund_universe` maps fund_name -> (universe, method), so each fund is
    annualised with ITS OWN periods_per_year (already stored on the
    backtest result: 252 for Equity/Combined, 365 for Crypto).
    """
    rows = []
    for fund_name, result in fund_backtests.items():
        universe, method = fund_universe[fund_name]
        metrics = performance_metrics_fn(result["daily_returns"],
                                          periods_per_year=result["periods_per_year"])
        rows.append({
            "fund": fund_name,
            "universe": universe,
            "method": METHOD_LABELS[method],
            "annualised_return": metrics["annualised_return"],
            "annualised_volatility": metrics["annualised_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "n_days": metrics["n_days"],
            "first_oos_date": result["first_oos_date"].date().isoformat(),
        })
    order = {u: i for i, u in enumerate(UNIVERSES)}
    df = pd.DataFrame(rows)
    df["_u"] = df["universe"].map(order)
    return df.sort_values(["_u", "method"]).drop(columns="_u").reset_index(drop=True)


def save_performance_table(df: pd.DataFrame) -> Path:
    path = RESULTS_TABLES_DIR / "performance_metrics.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Exhibit 1 - growth of $1, one subplot per universe, 4 method lines each
# ---------------------------------------------------------------------------

def plot_growth_of_dollar(fund_backtests: dict, save_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, universe in zip(axes, UNIVERSES):
        for method, label in METHOD_LABELS.items():
            g = fund_backtests[f"{universe} {label}"]["growth_of_dollar"]
            ax.plot(g.index, g.values, label=label, color=METHOD_COLORS[method], linewidth=1.3)
        ax.axhline(1.0, color=plotstyle.FT_INK, linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_title(universe, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("Growth of $1")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(loc="upper left", fontsize=8)
    plotstyle.ft_title(fig, "Out-of-sample growth of $1",
                       subtitle="Four optimisation methods, monthly rebalance")
    plotstyle.ft_source(fig, "Source: own calculation from equity/crypto adjusted-close prices "
                              "(2020-2023). Out-of-sample only - see performance_metrics.csv.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Exhibit 2 - drawdown, all 4 methods of ONE universe (default: Combined)
# ---------------------------------------------------------------------------

def plot_drawdown(fund_backtests: dict, save_path: Path, universe: str = "Combined") -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for method, label in METHOD_LABELS.items():
        g = fund_backtests[f"{universe} {label}"]["growth_of_dollar"]
        drawdown = (g / g.cummax() - 1.0) * 100
        ax.plot(drawdown.index, drawdown.values, label=label,
                 color=METHOD_COLORS[method], linewidth=1.2)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=9)
    plotstyle.ft_title(fig, f"{universe} funds - drawdown",
                       subtitle="Peak-to-trough decline of $1 invested")
    plotstyle.ft_source(fig, f"Source: own calculation, out-of-sample {universe.lower()} funds "
                              "(see performance_metrics.csv for the sample period).")
    fig.tight_layout(rect=[0, 0.05, 1, 0.85])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Exhibit 3 - portfolio weights over time, ONE fund, grouped by sector
# (crypto collapsed into a single band, matching the lecture's convention)
# ---------------------------------------------------------------------------

def plot_weights_over_time(fund_backtests: dict, fund_name: str,
                            ticker_sector_map: dict, save_path: Path) -> Path:
    weights = fund_backtests[fund_name]["weights"]
    sector_of = {t: ticker_sector_map.get(t, "Crypto") for t in weights.columns}
    grouped = weights.T.groupby(sector_of).sum().T * 100  # % weight per sector, over time

    sectors_present = sorted(s for s in grouped.columns if s != "Crypto")
    colors = dict(zip(sectors_present, plotstyle.SECTOR_PALETTE))
    if "Crypto" in grouped.columns:
        colors["Crypto"] = plotstyle.FT_INK  # single near-black band, per the lecture convention
    order = sectors_present + (["Crypto"] if "Crypto" in grouped.columns else [])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.stackplot(grouped.index, [grouped[s] for s in order], labels=order,
                 colors=[colors[s] for s in order])
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=7, ncol=2, bbox_to_anchor=(1.0, 1.0))
    plotstyle.ft_title(fig, f"{fund_name} - allocation over time",
                       subtitle="Weights grouped by sector")
    plotstyle.ft_source(fig, "Source: own calculation. Rebalanced monthly; "
                              "weights held constant between rebalances.")
    fig.tight_layout(rect=[0, 0.05, 0.82, 0.85])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Exhibit 4 - Sharpe ratio, grouped bar chart (universe x method)
# ---------------------------------------------------------------------------

def plot_sharpe_barplot(performance_table: pd.DataFrame, save_path: Path) -> Path:
    x = np.arange(len(UNIVERSES))
    n_methods = len(METHOD_LABELS)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (method_key, method_label) in enumerate(METHOD_LABELS.items()):
        vals = [performance_table.loc[(performance_table.universe == u) &
                                       (performance_table.method == method_label),
                                       "sharpe_ratio"].iloc[0] for u in UNIVERSES]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=method_label,
               color=METHOD_COLORS[method_key])
    ax.set_xticks(x)
    ax.set_xticklabels(UNIVERSES)
    ax.set_ylabel("Out-of-sample Sharpe ratio")
    ax.axhline(0, color=plotstyle.FT_INK, linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    plotstyle.ft_title(fig, "Funds ranked by Sharpe",
                       subtitle="Higher is better - out-of-sample risk-adjusted return")
    plotstyle.ft_source(fig, "Source: own calculation (see performance_metrics.csv "
                              "for exact values and the out-of-sample period per universe).")
    fig.tight_layout(rect=[0, 0.05, 1, 0.85])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

# ---------------------------------------------------------------------------
# Exhibit 6 - sector sentiment index over time, 2x5 grid, one panel/sector
# ---------------------------------------------------------------------------

def plot_sector_sentiment_index(sector_index: pd.DataFrame, save_path: Path,
                                 smoothing_window: int = 21) -> Path:
    sectors = sorted(c for c in sector_index.columns if c != "date")
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for ax, sector in zip(axes.flat, sectors):
        smoothed = sector_index[sector].rolling(smoothing_window, min_periods=1).mean()
        ax.plot(sector_index["date"], smoothed, color=plotstyle.EQUITY_COLOR, linewidth=1.1)
        ax.axhline(0, color=plotstyle.FT_INK, linewidth=0.5, alpha=0.3)
        ax.set_title(sector, fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    for ax in axes.flat[len(sectors):]:
        ax.axis("off")
    plotstyle.ft_title(fig, "Sector news-sentiment over time",
                       subtitle=f"finVADER, {smoothing_window}-day rolling average, 2020-2023")
    plotstyle.ft_source(fig, "Source: own calculation from equity news headlines (2020-2023). "
                              "Sentiment applies to the equity side only - crypto is price-only.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.88])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Exhibit (bonus) - market fear-and-greed gauge, level + standardised
# ---------------------------------------------------------------------------

def plot_fear_greed_index(fear_greed: pd.DataFrame, save_path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(fear_greed["date"], fear_greed["level_0_100"], color=plotstyle.EQUITY_COLOR, linewidth=1.0)
    ax1.axhline(50, color=plotstyle.FT_INK, linewidth=0.6, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Fear <-> Greed (0-100)")
    ax1.set_title("Level - stays near or above neutral (50) most days", fontsize=10, loc="left")

    pos = fear_greed["standardised"].clip(lower=0)
    neg = fear_greed["standardised"].clip(upper=0)
    ax2.fill_between(fear_greed["date"], pos, color=plotstyle.EQUITY_COLOR, alpha=0.75, linewidth=0)
    ax2.fill_between(fear_greed["date"], neg, color=plotstyle.ACCENT_COLOR, alpha=0.75, linewidth=0)
    ax2.axhline(0, color=plotstyle.FT_INK, linewidth=0.6)
    ax2.set_ylabel("Standardised (z)")
    ax2.set_title("Standardised - the fear spikes the level hides", fontsize=10, loc="left")

    plotstyle.ft_title(fig, "A market fear-and-greed index from the news",
                       subtitle="All equity tickers, finVADER headline sentiment")
    plotstyle.ft_source(fig, "Source: own calculation from equity news headlines (2020-2023).")
    fig.tight_layout(rect=[0, 0.03, 1, 0.86])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

# ---------------------------------------------------------------------------
# Exhibit 7 - fusion before-vs-after: discovery-vs-holdout Sharpe per config
# (shows overfitting risk directly - matches the course's own reference)
# ---------------------------------------------------------------------------

def plot_fusion_discovery_holdout(comparison_df: pd.DataFrame, save_path: Path) -> Path:
    configs = comparison_df["config"].tolist()
    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, comparison_df["discovery_sharpe"], width,
           label="Discovery (2021-2022, tuned here)", color=plotstyle.NEUTRAL_COLOR)
    ax.bar(x + width / 2, comparison_df["holdout_sharpe"], width,
           label="Holdout (2023, never seen)", color=plotstyle.ACCENT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color=plotstyle.FT_INK, linewidth=0.8)
    ax.set_ylabel("Sharpe ratio")
    ax.legend(loc="upper right", fontsize=9)
    plotstyle.ft_title(fig, "Tuning a sentiment tilt on the past can overfit the future",
                       subtitle="Equity Min-Variance base, tilt tuned on 2021-2022 then tested once on 2023")
    plotstyle.ft_source(fig, "Source: own calculation (see fusion_comparison.csv).")
    fig.tight_layout(rect=[0, 0.05, 1, 0.85])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Exhibit 7 (companion) - growth of $1, base vs chosen tilt, HOLDOUT only
# ---------------------------------------------------------------------------

def plot_fusion_growth_holdout(results_by_name: dict, start: str, end: str,
                                save_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [plotstyle.NEUTRAL_COLOR, plotstyle.EQUITY_COLOR]
    for color, (name, result) in zip(colors, results_by_name.items()):
        r = result["daily_returns"]
        r_window = r[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
        g = (1.0 + r_window).cumprod()
        sharpe = portfolios.performance_metrics(
            r_window, periods_per_year=result["periods_per_year"])["sharpe_ratio"]
        ax.plot(g.index, g.values, label=f"{name}  (holdout Sharpe {sharpe:.2f})",
                 color=color, linewidth=1.3)
    ax.axhline(1.0, color=plotstyle.FT_INK, linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="best", fontsize=8)
    plotstyle.ft_title(fig, "Does news sentiment improve the equity fund?",
                       subtitle="Equity Min-Variance base vs sentiment-tilted, 2023 holdout only, before costs")
    plotstyle.ft_source(fig, "Source: own calculation, out-of-sample 2023 holdout.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.85])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
