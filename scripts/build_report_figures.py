"""Generate a SECOND, salmon/FT-styled set of exhibit figures for the written
report, into results/figures_report/ - completely separate from
results/figures/ (which the app reads, and which stays navy+green).

Rationale: the app intentionally uses a navy+green fintech theme (matches
its own dark UI), while the report intentionally uses a salmon/cream
editorial theme (better for a printed/PDF document) - see Bagian 4 for the
justification. Keeping them in two different output folders means neither
workflow can accidentally overwrite the other.

This script recomputes the SAME funds/sentiment/fusion results as
scripts/run_part_b.py (it does not read run_part_b.py's saved CSVs, since
the plotting functions need the full in-memory backtest objects, not just
the summary tables) - so it takes a similar few minutes to run. The
underlying NUMBERS are identical either way; only the colours differ.

Run from the project root, any time after run_part_b.py has been run at
least once (so results/data/ exists):

    python scripts/build_report_figures.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import data_access, features, fusion, etl, sentiment, exhibits, portfolios, plotstyle  # noqa: E402

REPORT_FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures_report"
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Salmon/FT palette - report-only. This reassigns the SAME attributes the
# app's plotstyle.py defines, but only inside this script's process, so it
# never touches results/figures/ or the live app.
# ---------------------------------------------------------------------------
plotstyle.FT_PAPER = "#FFF1E5"
plotstyle.FT_INK = "#33302E"
plotstyle.FT_GRID = "#E6D9CC"
plotstyle.EQUITY_COLOR = "#0F5499"
plotstyle.CRYPTO_COLOR = "#E85D04"
plotstyle.NEUTRAL_COLOR = "#8D8D8D"
plotstyle.ACCENT_COLOR = "#990F3D"
plotstyle.SECTOR_PALETTE = [
    "#0F5499", "#E85D04", "#0D7680", "#593380", "#990F3D",
    "#1E874B", "#B8860B", "#66605C", "#4C9C2E", "#A8326E",
]
plotstyle.apply_style()  # re-apply rcParams so the new colours actually take effect

# exhibits.py computed its METHOD_COLORS dict once at import time from the
# OLD (navy) plotstyle values, so those need to be explicitly reassigned too
# - simply changing plotstyle's attributes above does not retroactively
# change a dict that already copied the old values.
exhibits.METHOD_COLORS = {
    "equal_weight": plotstyle.NEUTRAL_COLOR,
    "min_variance": "#0D7680",
    "max_sharpe": "#593380",
    "risk_parity": "#1E874B",
    "mean_cvar": "#B8860B",
}
exhibits.RESULTS_FIGURES_DIR = REPORT_FIGURES_DIR

METHOD_LABELS = {
    "equal_weight": "Equal-Weight",
    "min_variance": "Min-Variance",
    "max_sharpe": "Max-Sharpe",
    "risk_parity": "Risk-Parity",
    "mean_cvar": "Mean-CVaR",
}
UNIVERSE_CONFIG = {
    "Equity":   dict(periods_per_year=252, first_live_date="2021-01-01"),
    "Crypto":   dict(periods_per_year=365, first_live_date="2021-01-01"),
    "Combined": dict(periods_per_year=252, first_live_date="2021-01-01"),
}


def main():
    print(f"[report figures] output folder: {REPORT_FIGURES_DIR}")

    # Station 1-2 (same as run_part_b.py)
    equity = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()
    news = etl.load_clean_news()
    equity_returns = features.daily_returns(equity, price_col="adjClose")
    crypto_returns = features.daily_returns(crypto, price_col="adjClose")
    combined_returns = features.build_combined_returns_panel(equity_returns, crypto_returns)
    trading_calendar = features.get_equity_trading_calendar(equity)
    headline_panel = features.assemble_headline_panel(news, trading_calendar)
    print("[report figures] Station 1-2 done")

    # Station 3a - fund grid
    UNIVERSE_CONFIG["Equity"]["returns"] = equity_returns
    UNIVERSE_CONFIG["Crypto"]["returns"] = crypto_returns
    UNIVERSE_CONFIG["Combined"]["returns"] = combined_returns

    fund_backtests, fund_universe = {}, {}
    for universe_name, cfg in UNIVERSE_CONFIG.items():
        for method in portfolios.METHODS:
            fund_name = f"{universe_name} {METHOD_LABELS[method]}"
            fund_universe[fund_name] = (universe_name, method)
            fund_backtests[fund_name] = portfolios.oos_backtest(
                cfg["returns"], method=method,
                first_live_date=cfg["first_live_date"],
                periods_per_year=cfg["periods_per_year"],
            )
    print("[report figures] fund grid rebuilt")

    perf_table = exhibits.build_performance_table(fund_backtests, fund_universe,
                                                  portfolios.performance_metrics)
    sector_df = data_access.load_sector_universe()
    sector_map = dict(zip(sector_df["ticker"], sector_df["sector"]))

    exhibits.plot_growth_of_dollar(fund_backtests, REPORT_FIGURES_DIR / "growth_of_dollar.png")
    exhibits.plot_drawdown(fund_backtests, REPORT_FIGURES_DIR / "drawdown_combined.png", universe="Combined")
    exhibits.plot_weights_over_time(
        fund_backtests, "Combined Min-Variance", sector_map,
        REPORT_FIGURES_DIR / "weights_over_time_combined_minvariance.png")
    exhibits.plot_sharpe_barplot(perf_table, REPORT_FIGURES_DIR / "sharpe_barplot.png")
    print("[report figures] saved 4 fund figures")

    # Station 3c - sentiment
    ticker_day_scores = sentiment.score_headlines(headline_panel)
    sector_index = sentiment.sector_sentiment_index(ticker_day_scores, trading_calendar)
    fear_greed = sentiment.market_fear_greed_index(ticker_day_scores, trading_calendar)

    exhibits.plot_sector_sentiment_index(sector_index, REPORT_FIGURES_DIR / "sector_sentiment_index.png")
    exhibits.plot_fear_greed_index(fear_greed, REPORT_FIGURES_DIR / "fear_greed_index.png")
    exhibits.plot_lexicon_comparison(ticker_day_scores, trading_calendar,
                                      REPORT_FIGURES_DIR / "lexicon_comparison.png")
    print("[report figures] saved 3 sentiment figures")

    # Station 3d - fusion
    DISCOVERY_START, DISCOVERY_END = "2021-01-01", "2022-12-31"
    HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2023-12-31"
    base_equity_min_var = fund_backtests["Equity Min-Variance"]
    tilt_configs = {
        "Base (no tilt)": None,
        "Naive Momentum (lam=+1)": dict(lam=1.0, use_intensity_confidence=False),
        "Naive Contrarian (lam=-1)": dict(lam=-1.0, use_intensity_confidence=False),
        "Intensity Momentum (lam=+1)": dict(lam=1.0, use_intensity_confidence=True),
        "Intensity Contrarian (lam=-1)": dict(lam=-1.0, use_intensity_confidence=True),
        "Intensity Momentum + Custom Lexicon (lam=+1)": dict(
            lam=1.0, use_intensity_confidence=True, score_col="sentiment_finvader_custom"),
    }
    fusion_results, fusion_rows = {}, []
    for name, cfg in tilt_configs.items():
        result = (base_equity_min_var if cfg is None else
                  fusion.apply_sentiment(base_equity_min_var, equity_returns,
                                         ticker_day_scores, trading_calendar, **cfg))
        fusion_results[name] = result
        disc = fusion.evaluate_on_window(result, DISCOVERY_START, DISCOVERY_END)
        hold = fusion.evaluate_on_window(result, HOLDOUT_START, HOLDOUT_END)
        fusion_rows.append({"config": name, "discovery_sharpe": disc["sharpe_ratio"],
                             "discovery_return": disc["annualised_return"],
                             "holdout_sharpe": hold["sharpe_ratio"],
                             "holdout_return": hold["annualised_return"]})
    fusion_comparison = pd.DataFrame(fusion_rows)
    tilted_only = fusion_comparison[fusion_comparison.config != "Base (no tilt)"]
    chosen_name = tilted_only.loc[tilted_only["discovery_sharpe"].idxmax(), "config"]

    exhibits.plot_fusion_discovery_holdout(fusion_comparison, REPORT_FIGURES_DIR / "fusion_discovery_holdout.png")
    exhibits.plot_fusion_growth_holdout(
        {"Base (no tilt)": fusion_results["Base (no tilt)"], chosen_name: fusion_results[chosen_name]},
        HOLDOUT_START, HOLDOUT_END, REPORT_FIGURES_DIR / "fusion_growth_holdout.png")
    print("[report figures] saved 2 fusion figures")

    print(f"\n[report figures] all 9 figures saved to {REPORT_FIGURES_DIR} - "
          "insert these into the Word report, NOT the ones in results/figures/.")


if __name__ == "__main__":
    main()
