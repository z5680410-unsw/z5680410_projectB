"""Streamlit app - cached data loaders. Reads ONLY precomputed results/ files -
no backtest, no sentiment scoring, no nltk import (keeps cold starts fast on the
free tier, per the brief).
"""
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@st.cache_data(ttl=86_400, show_spinner=False)
def load_fund_returns() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "data" / "fund_returns.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def load_fund_weights() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "data" / "fund_weights.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def load_performance_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "tables" / "performance_metrics.csv")


@st.cache_data(ttl=86_400, show_spinner=False)
def load_sector_sentiment_index() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "data" / "sector_sentiment_index.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def load_fear_greed_index() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "data" / "fear_greed_index.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def load_fusion_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "tables" / "fusion_comparison.csv")


def latest_holdings(fund_weights: pd.DataFrame, fund_name: str, top_n: int = 10) -> pd.DataFrame:
    """Most recent rebalance's weights for one fund, top N by weight (%)."""
    sub = fund_weights[fund_weights["fund"] == fund_name]
    last_date = sub["date"].max()
    latest = sub[sub["date"] == last_date].copy()
    latest["weight_pct"] = latest["weight"] * 100
    return latest.sort_values("weight_pct", ascending=False).head(top_n)[["asset", "weight_pct"]]


def growth_of_dollar(fund_returns: pd.DataFrame, fund_name: str) -> pd.Series:
    """Growth of $1 for ONE fund, from fund_returns.csv's long format."""
    sub = fund_returns[fund_returns["fund"] == fund_name].sort_values("date")
    return (1.0 + sub.set_index("date")["daily_return"]).cumprod()


def blended_growth(fund_returns: pd.DataFrame, weights: dict,
                    fee_annual_pct: float = 0.0) -> pd.Series:
    """Growth of $1 for a user-chosen blend of funds (weights need not sum to
    1 - renormalised here), net of an annual management fee applied daily
    (fee/365 per day - a simple approximation, stated as such to the user).
    Funds with no return on a given date (different OOS start dates, e.g.
    equity vs crypto) are treated as contributing 0 that day, not dropped -
    so a blend can mix funds with different first_oos_date.
    """
    pivot = fund_returns.pivot(index="date", columns="fund", values="daily_return")
    names = [n for n in weights if weights.get(n, 0) > 0]
    if not names:
        return pd.Series(dtype=float)
    w = pd.Series({n: weights[n] for n in names})
    w = w / w.sum()
    blended_return = pivot[names].fillna(0.0).mul(w, axis=1).sum(axis=1)
    daily_fee = fee_annual_pct / 100.0 / 365.0
    net_return = blended_return - daily_fee
    return (1.0 + net_return).cumprod()


