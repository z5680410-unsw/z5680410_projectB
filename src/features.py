"""Station 2 - your features: return features and headline text assembly.

Build your return (and optional risk) features here, and assemble the headlines
into a daily text panel. Part A is assembly only - scoring the text into sentiment
(and lagging the signal) is the Station 3 model, built in Part B, not here.
"""
from pathlib import Path

import pandas as pd
import numpy as np

RESULTS_TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DATA_DIR = Path(__file__).resolve().parent.parent / "results" / "data"
RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker. Use adjClose.

    Returns a wide-format panel: one row per date, one column per ticker,
    values are simple daily returns (pct_change) computed within each
    ticker's own, independently sorted price series. Wide format matches
    what the Phase 4 calendar merge expects
    (pd.merge(equity_returns, crypto_returns, on='date', how='left')), and
    is also the shape the Part B portfolio optimiser will need later.

    The first observed date for every ticker is NaN by construction
    (pct_change has no prior value to compare against) - this is expected,
    not a data quality issue.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = df.groupby("ticker")[price_col].pct_change(fill_method=None)

    wide = df.pivot(index="date", columns="ticker", values="daily_return")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def normalize_news_timezone(headlines: pd.DataFrame) -> pd.DataFrame:
    """Strip UTC timezone-awareness from the news `date` column, and
    normalise to midnight so date comparisons in align_to_trading_day()
    compare calendar dates, not timestamps.

    Design choice: strip tz from the news dates rather than localise the
    (naive) price dates to UTC. Price dates represent trading days - a
    calendar concept with no meaningful time-of-day - so adding timezone
    information to them would be artificial. The news `date` carries a real
    UTC timestamp, but once headlines are aligned to a trading day, only the
    calendar date matters, so stripping tz here is the simpler, lower-risk
    direction of normalisation.

    .dt.normalize() is required, not just defensive: verified empirically
    (see scripts/check_timestamps.py) that 575 of 149,683 raw headlines
    carry a non-midnight timestamp. Without normalising to midnight,
    align_to_trading_day()'s searchsorted call mis-shifts any such headline
    published ON a trading day forward to the NEXT trading day (see
    scripts/quantify_timestamp_bug.py for the exact affected count).
    """
    df = headlines.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def build_combined_returns_panel(equity_returns: pd.DataFrame,
                                  crypto_returns: pd.DataFrame) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading calendar.

    Equity returns define the trading calendar (~252 days/year). Crypto
    returns are computed independently, on crypto's own 7-day calendar
    (see daily_returns()), and are only combined with equities here, at the
    return level - never at the price level. Left-merging onto the equity
    dates intentionally drops weekend-only crypto moves, since a fund that
    only rebalances on equity trading days could never act on them.
    """
    combined = pd.merge(equity_returns, crypto_returns, on="date", how="left",
                        validate="one_to_one")
    assert len(combined) == len(equity_returns), (
        f"Left-merge changed the row count: "
        f"{len(equity_returns)} -> {len(combined)}"
    )
    return combined


def descriptive_stats_by_asset_class(equity_returns: pd.DataFrame,
                                      crypto_returns: pd.DataFrame,
                                      equity_periods_per_year: int = 252,
                                      crypto_periods_per_year: int = 365) -> pd.DataFrame:
    """Pooled descriptive statistics of daily returns, by asset class.

    All ticker columns within an asset class are pooled into a single
    return distribution (the date/ticker structure is not preserved) and
    summarised with mean, std (volatility), min, max, skew, and kurtosis
    (excess kurtosis - 0 for a normal distribution). NaNs from the first
    date of each ticker's pct_change are dropped before pooling.

    Annualised mean and volatility are added using each asset class's own
    trading-day count: equities trade ~252 days/year, crypto trades every
    calendar day (~365 days/year). Annualised mean uses simple scaling
    (mean * periods_per_year); annualised volatility uses the square-root-
    of-time rule (std * sqrt(periods_per_year)), which assumes returns are
    approximately independent across days - a simplification, stated here
    rather than left implicit.
    """
    def _stats(wide_df: pd.DataFrame, label: str, periods_per_year: int) -> dict:
        values = wide_df.drop(columns=["date"]).to_numpy().flatten()
        values = pd.Series(values).dropna()
        mean = values.mean()
        std = values.std(ddof=0)
        return {
            "asset_class": label,
            "n_obs": int(values.shape[0]),
            "mean": mean,
            "std": std,
            "min": values.min(),
            "max": values.max(),
            "skew": values.skew(),
            "kurtosis": values.kurtosis(),
            "periods_per_year": periods_per_year,
            "annualised_mean": mean * periods_per_year,
            "annualised_std": std * np.sqrt(periods_per_year),
        }

    rows = [
        _stats(equity_returns, "Equity", equity_periods_per_year),
        _stats(crypto_returns, "Crypto", crypto_periods_per_year),
    ]
    return pd.DataFrame(rows)

def save_combined_returns_panel(combined_df: pd.DataFrame) -> Path:
    """Save the combined equity+crypto returns panel under results/data/.

    This is fully derived data (daily returns, not raw prices) and small
    enough (~1,006 rows) to save in full rather than sampled - matches the
    brief's own named example: combined_returns_panel.csv.
    """
    path = RESULTS_DATA_DIR / "combined_returns_panel.csv"
    combined_df.to_csv(path, index=False)
    return path

def save_descriptive_stats(stats_df: pd.DataFrame) -> Path:
    """Save the descriptive-statistics table under its required exact filename."""
    path = RESULTS_TABLES_DIR / "descriptive_stats_returns.csv"
    stats_df.to_csv(path, index=False)
    return path


def get_equity_trading_calendar(equity_prices: pd.DataFrame) -> pd.DatetimeIndex:
    """The reference trading calendar: every date the equity panel trades on."""
    return pd.DatetimeIndex(sorted(equity_prices["date"].unique()))


def align_to_trading_day(dates: pd.Series, trading_calendar: pd.DatetimeIndex) -> pd.Series:
    """Map each date to itself if it's a trading day, otherwise to the next one.

    Uses searchsorted on the sorted trading calendar: for each input date,
    find the first calendar date >= that date. This handles weekends AND
    holidays (any date absent from the equity calendar), not only weekends.
    Dates beyond the last trading day in the calendar map to NaT (there is
    no "next" trading day) and are flagged by the caller.
    """
    calendar = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_calendar).unique()))
    positions = calendar.searchsorted(dates.values, side="left")
    out_of_range = positions >= len(calendar)
    positions_clipped = positions.clip(max=len(calendar) - 1)
    aligned = pd.Series(calendar[positions_clipped], index=dates.index)
    aligned[out_of_range] = pd.NaT
    return aligned


def assemble_headline_panel(headlines: pd.DataFrame,
                             trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Assemble the headlines into a daily panel per ticker and sector.

    Station 2 (Part A) is assembly only: structure the text and date-align it to
    the trading calendar (a weekend or holiday headline maps to the next trading
    day). The panel stays at headline-level granularity (one row per headline,
    raw title text kept intact - no stopword stripping, since Part B's VADER
    model needs the full text) with an added n_articles column showing how many
    headlines share the same (ticker, trading_date). Scoring the text - and
    lagging the signal - is the Station 3 sentiment model, built in Part B,
    not here.
    """
    df = normalize_news_timezone(headlines)
    df["trading_date"] = align_to_trading_day(df["date"], trading_calendar)

    n_unmapped = int(df["trading_date"].isna().sum())
    if n_unmapped > 0:
        print(f"Warning: {n_unmapped} headlines fall after the last trading "
              f"date in the calendar and were dropped.")
        df = df.dropna(subset=["trading_date"])

    panel = df.rename(columns={"date": "original_publish_date"})
    panel = panel[["ticker", "sector", "trading_date", "original_publish_date",
                    "title", "url", "publisher"]].copy()

    article_counts = (
        panel.groupby(["ticker", "trading_date"])
        .size()
        .reset_index(name="n_articles")
    )
    panel = panel.merge(article_counts, on=["ticker", "trading_date"], how="left")

    panel = panel.sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    return panel


def save_headline_panel(panel: pd.DataFrame, sample_rows: int = 500) -> Path:
    """Save a small SAMPLE of the derived headline panel under results/data/
    (not the full ~146,830-row panel), per the brief's guidance to submit
    "a small sample of derived data" rather than the full pipeline output.

    The full panel is always rebuilt on demand from source via
    assemble_headline_panel() - this saved file exists only as evidence of
    the output schema for markers, not as the working dataset itself.
    Sampled with a fixed random_state for reproducibility, then re-sorted
    into (ticker, trading_date) order for readability.
    """
    n = min(sample_rows, len(panel))
    sample = panel.sample(n=n, random_state=42).sort_values(["ticker", "trading_date"])
    path = RESULTS_DATA_DIR / "headline_panel.csv"
    sample.to_csv(path, index=False)
    return path
