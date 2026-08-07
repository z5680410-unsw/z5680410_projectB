"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py

Station 1-2 (this file) rebuilds your Part A foundation from the raw hosted data,
so the whole pipeline is reproducible from a clean checkout - not from a saved
CSV. Station 3-4 (funds, sentiment, fusion, app artifacts) are added on top in
later phases; see the TODO block at the bottom of main().
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import data_access, features, fusion, etl, sentiment, exhibits, portfolios  # noqa: E402

# One fund grid entry per asset universe. estimation_window and
# rebalance_every are expressed in ROWS of that universe's own returns
# panel, so they are set separately for crypto (365-day calendar) vs
# equity/combined (252-day calendar) to represent the SAME ~1 year lookback
# and ~1 month rebalance in calendar time, not the same row count.
UNIVERSE_CONFIG = {
    # filled in with the actual returns panels inside main()
    "Equity":   dict(periods_per_year=252, estimation_window=252, rebalance_every=21),
    "Crypto":   dict(periods_per_year=365, estimation_window=365, rebalance_every=30),
    "Combined": dict(periods_per_year=252, estimation_window=252, rebalance_every=21),
}
METHOD_LABELS = {
    "equal_weight": "Equal-Weight",
    "min_variance": "Min-Variance",
    "max_sharpe": "Max-Sharpe",
    "risk_parity": "Risk-Parity",
    "mean_cvar": "Mean-CVaR",
}


def main():
    # ------------------------------------------------------------------
    # Station 1 - load + clean (Part A foundation, reproduced here)
    # ------------------------------------------------------------------
    equity = etl.load_clean_equities()
    crypto = etl.load_clean_crypto()
    news = etl.load_clean_news()
    print(f"\n[Station 1] equity={equity.shape}  crypto={crypto.shape}  news={news.shape}")

    # ------------------------------------------------------------------
    # Station 2 - return features + text panel (Part A foundation, reproduced here)
    # ------------------------------------------------------------------
    equity_returns = features.daily_returns(equity, price_col="adjClose")
    crypto_returns = features.daily_returns(crypto, price_col="adjClose")
    combined_returns = features.build_combined_returns_panel(equity_returns, crypto_returns)
    features.save_combined_returns_panel(combined_returns)

    stats = features.descriptive_stats_by_asset_class(equity_returns, crypto_returns)
    features.save_descriptive_stats(stats)

    trading_calendar = features.get_equity_trading_calendar(equity)
    headline_panel = features.assemble_headline_panel(news, trading_calendar)
    features.save_headline_panel(headline_panel)

    print(f"[Station 2] equity_returns={equity_returns.shape}  "
          f"crypto_returns={crypto_returns.shape}  "
          f"combined_returns={combined_returns.shape}  "
          f"headline_panel={headline_panel.shape}")
    print(f"[Station 2] trading calendar: {trading_calendar.min().date()} "
          f"to {trading_calendar.max().date()} ({len(trading_calendar)} trading days)")
    print("\n" + stats.to_string(index=False))

    # ------------------------------------------------------------------
    # Station 3a - funds: walk-forward OOS backtest, 3 universes x 4 methods
    # (Fase 1-2)
    # ------------------------------------------------------------------
    UNIVERSE_CONFIG["Equity"]["returns"] = equity_returns
    UNIVERSE_CONFIG["Crypto"]["returns"] = crypto_returns
    UNIVERSE_CONFIG["Combined"]["returns"] = combined_returns

    print("\n[Station 3] building the fund grid (3 universes x "
          f"{len(portfolios.METHODS)} methods - this can take a few minutes)...")
    fund_backtests = {}
    fund_universe = {}
    fund_returns_rows, fund_weights_frames = [], []
    for universe_name, cfg in UNIVERSE_CONFIG.items():
        for method in portfolios.METHODS:
            fund_name = f"{universe_name} {METHOD_LABELS[method]}"
            fund_universe[fund_name] = (universe_name, method)
            result = portfolios.oos_backtest(
                cfg["returns"], method=method,
                estimation_window=cfg["estimation_window"],
                rebalance_every=cfg["rebalance_every"],
                periods_per_year=cfg["periods_per_year"],
            )
            fund_backtests[fund_name] = result
            print(f"  {fund_name:<28s} first_oos={result['first_oos_date'].date()}  "
                  f"n_days={len(result['daily_returns'])}  "
                  f"n_rebalances={len(result['weights'])}")

            for date, ret in result["daily_returns"].items():
                fund_returns_rows.append({"fund": fund_name, "date": date, "daily_return": ret})
            w_long = (result["weights"].reset_index()
                      .melt(id_vars="date", var_name="asset", value_name="weight"))
            w_long.insert(0, "fund", fund_name)
            fund_weights_frames.append(w_long)

    fund_returns_df = pd.DataFrame(fund_returns_rows).sort_values(["fund", "date"])
    fund_weights_df = pd.concat(fund_weights_frames, ignore_index=True).sort_values(
        ["fund", "date", "asset"])

    fund_returns_path = features.RESULTS_DATA_DIR / "fund_returns.csv"
    fund_weights_path = features.RESULTS_DATA_DIR / "fund_weights.csv"
    fund_returns_df.to_csv(fund_returns_path, index=False)
    fund_weights_df.to_csv(fund_weights_path, index=False)
    print(f"\n[Station 3] saved {fund_returns_path.name} {fund_returns_df.shape} and "
          f"{fund_weights_path.name} {fund_weights_df.shape}")

    # ------------------------------------------------------------------
    # Station 3b - fact sheet: performance table + the 4 required exhibits
    # (Fase 3)
    # ------------------------------------------------------------------
    perf_table = exhibits.build_performance_table(fund_backtests, fund_universe,
                                                  portfolios.performance_metrics)
    exhibits.save_performance_table(perf_table)
    print("\n[Station 3] performance_metrics.csv:\n" + perf_table.to_string(index=False))

    sector_df = data_access.load_sector_universe()
    sector_map = dict(zip(sector_df["ticker"], sector_df["sector"]))

    exhibits.plot_growth_of_dollar(
        fund_backtests, exhibits.RESULTS_FIGURES_DIR / "growth_of_dollar.png")
    exhibits.plot_drawdown(
        fund_backtests, exhibits.RESULTS_FIGURES_DIR / "drawdown_combined.png", universe="Combined")
    exhibits.plot_weights_over_time(
        fund_backtests, "Combined Min-Variance", sector_map,
        exhibits.RESULTS_FIGURES_DIR / "weights_over_time_combined_minvariance.png")
    exhibits.plot_sharpe_barplot(
        perf_table, exhibits.RESULTS_FIGURES_DIR / "sharpe_barplot.png")
    print(f"[Station 3] saved 4 figures to {exhibits.RESULTS_FIGURES_DIR}")

    # ------------------------------------------------------------------
    # Station 3c - sentiment: score headlines, sector index, fear & greed
    # (Fase 4-5)
    # ------------------------------------------------------------------
    print("\n[Station 3] scoring headlines with VADER + finVADER "
          f"({len(headline_panel):,} headlines, ~20-30s)...")
    ticker_day_scores = sentiment.score_headlines(headline_panel)
    print(f"[Station 3] ticker-day sentiment scores: {ticker_day_scores.shape}")

    sector_index = sentiment.sector_sentiment_index(ticker_day_scores, trading_calendar)
    sector_index_path = features.RESULTS_DATA_DIR / "sector_sentiment_index.csv"
    sector_index.to_csv(sector_index_path, index=False)
    coverage = sector_index.drop(columns="date").notna().mean().mean()
    print(f"[Station 3] saved {sector_index_path.name} {sector_index.shape} "
          f"(avg. {coverage:.0%} of ticker-sector-days have at least one headline)")

    fear_greed = sentiment.market_fear_greed_index(ticker_day_scores, trading_calendar)
    fear_greed.to_csv(features.RESULTS_DATA_DIR / "fear_greed_index.csv", index=False)

    exhibits.plot_sector_sentiment_index(
        sector_index, exhibits.RESULTS_FIGURES_DIR / "sector_sentiment_index.png")
    exhibits.plot_fear_greed_index(
        fear_greed, exhibits.RESULTS_FIGURES_DIR / "fear_greed_index.png")
    print("[Station 3] saved sector_sentiment_index.png and fear_greed_index.png")

    # ------------------------------------------------------------------
    # Station 3d - fusion + innovation (Fase 6-7)
    # ------------------------------------------------------------------
    # Discipline (slide 36): tune on DISCOVERY only, reveal HOLDOUT once.
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

    print("\n[Station 3] fusion: tilting Equity Min-Variance with lagged sentiment...")
    fusion_results, fusion_rows = {}, []
    for name, cfg in tilt_configs.items():
        result = (base_equity_min_var if cfg is None else
                  fusion.apply_sentiment(base_equity_min_var, equity_returns,
                                         ticker_day_scores, trading_calendar, **cfg))
        fusion_results[name] = result
        disc = fusion.evaluate_on_window(result, DISCOVERY_START, DISCOVERY_END)
        hold = fusion.evaluate_on_window(result, HOLDOUT_START, HOLDOUT_END)
        fusion_rows.append({
            "config": name,
            "discovery_sharpe": disc["sharpe_ratio"],
            "discovery_return": disc["annualised_return"],
            "holdout_sharpe": hold["sharpe_ratio"],
            "holdout_return": hold["annualised_return"],
        })
        print(f"  {name:<28s} discovery Sharpe={disc['sharpe_ratio']:+.3f}  "
              f"holdout Sharpe={hold['sharpe_ratio']:+.3f}")

    fusion_comparison = pd.DataFrame(fusion_rows)
    fusion_comparison.to_csv(features.RESULTS_TABLES_DIR / "fusion_comparison.csv", index=False)

    # Disciplined choice: best DISCOVERY Sharpe among the TILTED configs only
    # (never picked using the holdout number).
    tilted_only = fusion_comparison[fusion_comparison.config != "Base (no tilt)"]
    chosen_name = tilted_only.loc[tilted_only["discovery_sharpe"].idxmax(), "config"]
    print(f"[Station 3] disciplined choice (best on discovery, revealed on holdout once): "
          f"'{chosen_name}'")

    exhibits.plot_fusion_discovery_holdout(
        fusion_comparison, exhibits.RESULTS_FIGURES_DIR / "fusion_discovery_holdout.png")
    exhibits.plot_fusion_growth_holdout(
        {"Base (no tilt)": fusion_results["Base (no tilt)"],
         chosen_name: fusion_results[chosen_name]},
        HOLDOUT_START, HOLDOUT_END,
        exhibits.RESULTS_FIGURES_DIR / "fusion_growth_holdout.png")
    print("[Station 3] saved fusion_comparison.csv, fusion_discovery_holdout.png, "
          "fusion_growth_holdout.png")


if __name__ == "__main__":
    main()
