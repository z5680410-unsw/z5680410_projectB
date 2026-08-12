"""Quantvestment - systematic multi-asset funds, priced out-of-sample.

Loads ONLY precomputed results/ artifacts - no backtest, no sentiment scoring,
no nltk import, so cold starts stay fast on the free tier (see
PROJECT_BRIEF.md, Station 4).

Run locally:   streamlit run streamlit_app.py
Deploy:        push this folder to a public GitHub repo, then connect it on
               share.streamlit.io with entrypoint streamlit_app.py (see brief App. D).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402

from src import app_data  # noqa: E402
from src import plotstyle  # noqa: E402  (applies the FT rcParams style on import)

FIGURES_DIR = pathlib.Path(__file__).resolve().parent / "results" / "figures"

st.set_page_config(page_title="Quantvestment", layout="wide")
st.title("Quantvestment")
st.caption("Systematically managed multi-asset funds, priced out-of-sample. "
           "Compare funds, read a fact sheet, follow the news-sentiment analytics, "
           "then build your own allocation.")

tab_compare, tab_factsheet, tab_sentiment, tab_allocation = st.tabs(
    ["Compare Funds", "Fund Fact Sheet", "Sentiment Analytics", "Build Allocation"])

# ---------------------------------------------------------------------------
# Tab 1 - Compare Funds
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Compare the funds")
    st.caption("All figures are out-of-sample - the weights that produced them "
               "were set using only earlier data.")

    perf = app_data.load_performance_metrics()
    display_cols = ["fund", "universe", "method", "annualised_return",
                     "annualised_volatility", "sharpe_ratio", "max_drawdown",
                     "first_oos_date"]
    st.dataframe(
        perf[display_cols].style.format({
            "annualised_return": "{:.1%}",
            "annualised_volatility": "{:.1%}",
            "sharpe_ratio": "{:.2f}",
            "max_drawdown": "{:.1%}",
        }),
        width="stretch", hide_index=True,
    )

    sharpe_chart = FIGURES_DIR / "sharpe_barplot.png"
    if sharpe_chart.exists():
        st.image(str(sharpe_chart), width="stretch")
    else:
        st.warning(f"{sharpe_chart.name} not found - run scripts/run_part_b.py first.")

# ---------------------------------------------------------------------------
# Tab 2 - Fund Fact Sheet
# ---------------------------------------------------------------------------
with tab_factsheet:
    st.subheader("Fund fact sheet")
    perf = app_data.load_performance_metrics()
    fund_returns = app_data.load_fund_returns()
    fund_weights = app_data.load_fund_weights()

    selected_fund = st.selectbox("Choose a fund", perf["fund"].tolist())
    row = perf[perf["fund"] == selected_fund].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised return", f"{row['annualised_return']:.1%}")
    c2.metric("Annualised volatility", f"{row['annualised_volatility']:.1%}")
    c3.metric("Sharpe ratio", f"{row['sharpe_ratio']:.2f}")
    c4.metric("Max drawdown", f"{row['max_drawdown']:.1%}")

    growth = app_data.growth_of_dollar(fund_returns, selected_fund)
    drawdown = (growth / growth.cummax() - 1.0) * 100

    col_left, col_right = st.columns(2)
    with col_left:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(growth.index, growth.values, color=plotstyle.EQUITY_COLOR, linewidth=1.3)
        ax.axhline(1.0, color=plotstyle.FT_INK, linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_ylabel("Growth of $1")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plotstyle.ft_title(fig, f"{selected_fund} - growth of $1")
        fig.tight_layout(rect=[0, 0.02, 1, 0.82])
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(drawdown.index, drawdown.values, color=plotstyle.ACCENT_COLOR, linewidth=1.2)
        ax.set_ylabel("Drawdown (%)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plotstyle.ft_title(fig, f"{selected_fund} - drawdown")
        fig.tight_layout(rect=[0, 0.02, 1, 0.82])
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Current holdings** (target weights at the most recent rebalance)")
    holdings = app_data.latest_holdings(fund_weights, selected_fund, top_n=10)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(holdings["asset"], holdings["weight_pct"], color=plotstyle.EQUITY_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Weight (%)")
    plotstyle.ft_title(fig, f"{selected_fund} - top holdings", title_fontsize=10, subtitle_fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.85])
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Tab 3 - Sentiment Analytics
# ---------------------------------------------------------------------------
with tab_sentiment:
    st.subheader("News sentiment analytics")
    st.caption("Built from equity news headlines only - crypto is price-only, "
               "so it has no sentiment signal.")

    sector_index = app_data.load_sector_sentiment_index()
    all_sectors = sorted(c for c in sector_index.columns if c != "date")
    selected_sectors = st.multiselect("Choose sectors to display", all_sectors, default=all_sectors)

    if selected_sectors:
        n = len(selected_sectors)
        ncols = min(5, n)
        nrows = -(-n // ncols)  # ceiling division
        fig_h = max(2.6 * nrows, 3.3)  # floor so a 1-row grid never overlaps the title
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, fig_h), sharex=True)
        axes = np.atleast_1d(axes).flatten()
        for ax, sector in zip(axes, selected_sectors):
            smoothed = sector_index[sector].rolling(21, min_periods=1).mean()
            ax.plot(sector_index["date"], smoothed, color=plotstyle.EQUITY_COLOR, linewidth=1.1)
            ax.axhline(0, color=plotstyle.FT_INK, linewidth=0.5, alpha=0.3)
            ax.set_title(sector, fontsize=9, fontweight="bold", loc="left")
            ax.tick_params(axis="x", rotation=30, labelsize=7)
        for ax in axes[len(selected_sectors):]:
            ax.axis("off")
        top_frac = 1 - 0.55 / fig_h
        plotstyle.ft_title(fig, "Sector news-sentiment over time",
                            subtitle="finVADER, 21-day rolling average",
                            title_fontsize=11, subtitle_fontsize=8)
        fig.tight_layout(rect=[0, 0.03, 1, top_frac])
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Choose at least one sector.")

    fear_greed_path = FIGURES_DIR / "fear_greed_index.png"
    if fear_greed_path.exists():
        st.image(str(fear_greed_path), width="stretch")
    else:
        st.warning("fear_greed_index.png not found - run scripts/run_part_b.py first.")

    st.markdown("---")
    st.subheader("Does sentiment improve the equity fund? (fusion tilt)")
    st.caption("Base fund is Equity Min-Variance. Tilt configurations were chosen "
               "using only the 2021-2022 discovery window, then tested once on the "
               "2023 holdout - see report for the full discussion.")

    fusion_comparison = app_data.load_fusion_comparison()
    st.dataframe(
        fusion_comparison.style.format({
            "discovery_sharpe": "{:.2f}", "discovery_return": "{:.1%}",
            "holdout_sharpe": "{:.2f}", "holdout_return": "{:.1%}",
        }),
        width="stretch", hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        p1 = FIGURES_DIR / "fusion_discovery_holdout.png"
        if p1.exists():
            st.image(str(p1), width="stretch")
    with col_b:
        p2 = FIGURES_DIR / "fusion_growth_holdout.png"
        if p2.exists():
            st.image(str(p2), width="stretch")

# ---------------------------------------------------------------------------
# Tab 4 - Build Allocation
# ---------------------------------------------------------------------------
with tab_allocation:
    st.subheader("Build your own allocation")
    st.caption("Blend across funds and see the net growth of $1, after an annual "
               "management fee - this is how Quantvestment earns its revenue.")

    perf = app_data.load_performance_metrics()
    fund_returns = app_data.load_fund_returns()
    all_funds = perf["fund"].tolist()

    chosen_funds = st.multiselect("Choose funds to blend", all_funds,
                                   default=all_funds[:2])

    weights = {}
    if chosen_funds:
        cols = st.columns(len(chosen_funds))
        even_split = round(100 / len(chosen_funds))
        for col, fund in zip(cols, chosen_funds):
            with col:
                weights[fund] = st.slider(fund, 0, 100, even_split, key=f"alloc_{fund}")

    fee = st.slider("Annual management fee (%)", 0.0, 3.0, 1.0, 0.1)

    total_pct = sum(weights.values())
    if total_pct == 0:
        st.warning("Choose at least one fund with a positive allocation.")
    else:
        if total_pct != 100:
            st.caption(f"Sliders sum to {total_pct}% - weights are renormalised to 100% "
                       "before computing the blend.")

        blend = app_data.blended_growth(fund_returns, weights, fee_annual_pct=fee)

        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(blend.index, blend.values, color=plotstyle.EQUITY_COLOR, linewidth=1.4)
        ax.axhline(1.0, color=plotstyle.FT_INK, linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_ylabel("Growth of $1")
        ax.tick_params(axis="x", rotation=30)
        plotstyle.ft_title(fig, "Your blended allocation",
                            subtitle=f"Net of a {fee:.1f}% annual management fee",
                            title_fontsize=11, subtitle_fontsize=9)
        fig.tight_layout(rect=[0, 0.03, 1, 0.82])
        st.pyplot(fig)
        plt.close(fig)

        # Annualise using the blend's OWN calendar span, not a fixed 252/365 -
        # a blend can mix equity (252 days/yr) and crypto (365 days/yr) funds,
        # so no single fixed constant is correct for every possible blend.
        span_years = (blend.index[-1] - blend.index[0]).days / 365.25
        periods_per_year = len(blend) / span_years if span_years > 0 else 252
        blend_returns = blend.pct_change().dropna()
        ann_return = blend_returns.mean() * periods_per_year
        ann_vol = blend_returns.std() * (periods_per_year ** 0.5)
        sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric("Net annualised return", f"{ann_return:.1%}")
        c2.metric("Annualised volatility", f"{ann_vol:.1%}")
        c3.metric("Net Sharpe ratio", f"{sharpe:.2f}")
