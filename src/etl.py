"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md). Add your own
integrity checks. Do not commit data files.
"""
from pathlib import Path

import pandas as pd

from src import data_access

RESULTS_TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers (Phase 3 - integrity checks)
# ---------------------------------------------------------------------------

def _audit_missing_dates_equity(df):
    """Missing-date audit for equities.

    Design choice: the expected trading calendar is the union of all dates
    observed across all 50 tickers, since equities share a single market
    calendar (NYSE). This is fully reproducible from the data itself, with
    no dependency on an external holiday calendar.
    """
    expected_calendar = pd.Index(sorted(df["date"].unique()))
    rows = []
    for ticker, g in df.groupby("ticker"):
        actual_dates = pd.Index(g["date"].unique())
        missing = expected_calendar.difference(actual_dates)
        rows.append({
            "ticker": ticker,
            "missing_dates": len(missing),
            "first_date": g["date"].min().date(),
            "last_date": g["date"].max().date(),
        })
    return pd.DataFrame(rows).sort_values("missing_dates", ascending=False)


def _audit_missing_dates_crypto(df):
    """Missing-date audit for crypto.

    Design choice: crypto trades every calendar day (7 days/week), so the
    expected calendar per coin is every day between that coin's own min
    and max observed date.
    """
    rows = []
    for ticker, g in df.groupby("ticker"):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        actual_dates = pd.Index(g["date"].unique())
        missing = full_range.difference(actual_dates)
        rows.append({
            "ticker": ticker,
            "missing_dates": len(missing),
            "first_date": g["date"].min().date(),
            "last_date": g["date"].max().date(),
        })
    return pd.DataFrame(rows).sort_values("missing_dates", ascending=False)


def _check_duplicates(df, subset):
    """Return (n_duplicate_rows, duplicate_rows_df) for the given key."""
    dup_mask_all = df.duplicated(subset=subset, keep=False)
    n_dupes = int(df.duplicated(subset=subset, keep="first").sum())
    return n_dupes, df[dup_mask_all].sort_values(subset)


def _flag_return_outliers(df, price_col="adjClose", z_thresh=5):
    """Compute a preliminary daily return per ticker and flag extreme moves.

    NOTE: this return is for outlier screening only. Station 2 (feature
    engineering, Phase 4) builds the official return feature used downstream.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = df.groupby("ticker")[price_col].pct_change()
    zscore = df.groupby("ticker")["daily_return"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    df["return_zscore"] = zscore
    df["is_return_outlier"] = df["return_zscore"].abs() > z_thresh
    return df


def _check_price_consistency(df):
    """Flag internal price/volume inconsistencies. Does NOT drop rows."""
    issues = pd.DataFrame(index=df.index)
    issues["high_lt_low"] = df["high"] < df["low"]
    issues["close_out_of_range"] = (df["close"] < df["low"]) | (df["close"] > df["high"])
    issues["open_out_of_range"] = (df["open"] < df["low"]) | (df["open"] > df["high"])
    issues["negative_volume"] = df["volume"] < 0
    issues["non_positive_price"] = (
        (df["open"] <= 0) | (df["high"] <= 0) |
        (df["low"] <= 0) | (df["close"] <= 0)
    )
    # Guarded: the unit-test synthetic dataframe in
    # test_price_consistency_catches_bad_rows() has no adjClose column,
    # so this check is skipped (returns False) rather than raising a
    # KeyError when the column is absent.
    if "adjClose" in df.columns:
        issues["missing_adj_close"] = df["adjClose"].isna()
    else:
        issues["missing_adj_close"] = False

    # Zero volume is tracked SEPARATELY, not folded into any_issue: a
    # zero-volume day can be a genuine low-liquidity observation rather
    # than a data error, so it should be visible but not automatically
    # counted as a "problem" row.
    issues["zero_volume"] = df["volume"] == 0

    issues["any_issue"] = issues[[
        "high_lt_low", "close_out_of_range", "open_out_of_range",
        "negative_volume", "non_positive_price", "missing_adj_close",
    ]].any(axis=1)
    return issues


def _save_integrity_exhibit(dataset_name, summary_row, missing_df=None,
                             outlier_df=None, dup_df=None):
    """Save/update the data-integrity summary exhibit files under results/tables/."""
    summary_path = RESULTS_TABLES_DIR / "data_integrity_summary.csv"
    summary_row_df = pd.DataFrame([summary_row])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["dataset"] != dataset_name]  # avoid dupes on re-run
        combined = pd.concat([existing, summary_row_df], ignore_index=True)
    else:
        combined = summary_row_df
    combined.to_csv(summary_path, index=False)

    if missing_df is not None:
        missing_df.to_csv(RESULTS_TABLES_DIR / f"missing_dates_{dataset_name}.csv", index=False)
    if outlier_df is not None and len(outlier_df) > 0:
        outlier_df.to_csv(RESULTS_TABLES_DIR / f"return_outliers_{dataset_name}.csv", index=False)
    if dup_df is not None and len(dup_df) > 0:
        dup_df.to_csv(RESULTS_TABLES_DIR / f"duplicates_{dataset_name}.csv", index=False)


# ---------------------------------------------------------------------------
# Main loaders
# ---------------------------------------------------------------------------

def load_clean_equities():
    """Load equity prices and run Station 1 integrity checks."""
    df = data_access.load_equity_prices()

    # 1. Missing-date audit
    missing_summary = _audit_missing_dates_equity(df)
    total_missing = int(missing_summary["missing_dates"].sum())

    # 2. Duplicate check on (ticker, date)
    n_dupes, dupe_rows = _check_duplicates(df, subset=["ticker", "date"])
    if n_dupes > 0:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # 3. Outlier / extreme-value screen + internal consistency checks
    df = _flag_return_outliers(df, price_col="adjClose", z_thresh=5)
    n_outliers = int(df["is_return_outlier"].sum())
    df = _flag_return_outliers_mad(df, mad_thresh=3.5)
    n_outliers_mad = int(df["is_return_outlier_mad"].sum())

    consistency = _check_price_consistency(df)
    n_consistency_issues = int(consistency["any_issue"].sum())

    print("--- equity_prices integrity check ---")
    print(f"total missing dates (all tickers): {total_missing}")
    print(f"duplicate (ticker, date) rows:      {n_dupes}")
    print(f"return outliers (|z| > 5):           {n_outliers}")
    print(f"return outliers (MAD > 3.5):         {n_outliers_mad}")
    print(f"price consistency issues:            {n_consistency_issues}")

    # 4. Save exhibit
    summary_row = {
        "dataset": "equity_prices",
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "total_missing_dates": total_missing,
        "n_duplicate_rows": n_dupes,
        "n_return_outliers": n_outliers,
        "n_price_consistency_issues": n_consistency_issues,
        "n_return_outliers_mad": n_outliers_mad,
    }
    _save_integrity_exhibit(
        "equity", summary_row,
        missing_df=missing_summary,
        outlier_df=df[df["is_return_outlier"]],
        dup_df=dupe_rows,
    )

    return df


def load_clean_crypto():
    """Load crypto prices (365-day calendar) and run Station 1 integrity checks."""
    df = data_access.load_crypto_prices()

    # cap at 2023-12-31 (from Phase 2)
    df = df[df["date"] <= "2023-12-31"].copy()

    # 1. Missing-date audit (7-day calendar)
    missing_summary = _audit_missing_dates_crypto(df)
    total_missing = int(missing_summary["missing_dates"].sum())

    # 2. Duplicate check on (ticker, date)
    n_dupes, dupe_rows = _check_duplicates(df, subset=["ticker", "date"])
    if n_dupes > 0:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # 3. Outlier / extreme-value screen + internal consistency checks
    df = _flag_return_outliers(df, price_col="adjClose", z_thresh=5)
    n_outliers = int(df["is_return_outlier"].sum())
    df = _flag_return_outliers_mad(df, mad_thresh=3.5)
    n_outliers_mad = int(df["is_return_outlier_mad"].sum())

    consistency = _check_price_consistency(df)
    n_consistency_issues = int(consistency["any_issue"].sum())

    print("--- crypto_prices integrity check ---")
    print(f"total missing dates (all tickers): {total_missing}")
    print(f"duplicate (ticker, date) rows:      {n_dupes}")
    print(f"return outliers (|z| > 5):           {n_outliers}")
    print(f"return outliers (MAD > 3.5):         {n_outliers_mad}")
    print(f"price consistency issues:            {n_consistency_issues}")

    # 4. Save exhibit
    summary_row = {
        "dataset": "crypto_prices",
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "total_missing_dates": total_missing,
        "n_duplicate_rows": n_dupes,
        "n_return_outliers": n_outliers,
        "n_price_consistency_issues": n_consistency_issues,
        "n_return_outliers_mad": n_outliers_mad,
    }
    _save_integrity_exhibit(
        "crypto", summary_row,
        missing_df=missing_summary,
        outlier_df=df[df["is_return_outlier"]],
        dup_df=dupe_rows,
    )

    return df


def load_clean_news():
    """Load news headlines and dedup on (ticker, date, title).

    NOTE: full timezone alignment to the equity trading calendar happens in
    Phase 4 (feature engineering / calendar merge), not here.
    """
    df = data_access.load_news_headlines()

    n_dupes, dupe_rows = _check_duplicates(df, subset=["ticker", "date", "title"])
    if n_dupes > 0:
        df = df.drop_duplicates(subset=["ticker", "date", "title"], keep="first")

    print("--- news_headlines integrity check ---")
    print(f"duplicate (ticker, date, title) rows: {n_dupes}")

    summary_row = {
        "dataset": "news_headlines",
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "total_missing_dates": None,
        "n_duplicate_rows": n_dupes,
        "n_return_outliers": None,
        "n_price_consistency_issues": None,
    }
    _save_integrity_exhibit("news", summary_row, dup_df=dupe_rows)

    return df

def _flag_return_outliers_mad(df, mad_thresh=3.5):
    """Robust outlier screen using the median absolute deviation (MAD), for
    comparison against the classic mean/std z-score screen.

    The classic screen suffers from a masking effect: a few very extreme
    observations inflate the standard deviation used to detect outliers,
    which can hide OTHER genuine outliers that would otherwise stand out.
    The median and MAD are far less sensitive to a small number of extreme
    values, since both are based on ranks/medians rather than means.

    Uses the standard 0.6745 scaling constant so the modified z-score is
    comparable in magnitude to a normal z-score (0.6745 = the 75th
    percentile of the standard normal distribution).
    """
    def _mod_zscore(s):
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0:
            return pd.Series(0.0, index=s.index)
        return 0.6745 * (s - median) / mad

    df = df.copy()
    df["mod_zscore"] = df.groupby("ticker")["daily_return"].transform(_mod_zscore)
    df["is_return_outlier_mad"] = df["mod_zscore"].abs() > mad_thresh
    return df


def compare_outlier_detection_methods(df, dataset_name):
    """Compare classic z-score vs robust MAD outlier flags on the SAME
    data. Returns a one-row summary of how many days each method flags,
    and how much they overlap.
    """
    n_z = int(df["is_return_outlier"].sum())
    n_mad = int(df["is_return_outlier_mad"].sum())
    n_both = int((df["is_return_outlier"] & df["is_return_outlier_mad"]).sum())
    return {
        "dataset": dataset_name,
        "n_flagged_zscore": n_z,
        "n_flagged_mad": n_mad,
        "n_flagged_by_both": n_both,
        "n_flagged_zscore_only": n_z - n_both,
        "n_flagged_mad_only": n_mad - n_both,
    }
def detect_corporate_action_dates(equity, ratio_change_threshold=0.05):
    """Detect likely corporate-action dates (stock splits, large dividend
    adjustments) by finding day-over-day jumps in the close/adjClose
    ratio, rather than assuming specific tickers had specific events.

    adjClose retroactively adjusts historical prices for splits and
    dividends so that returns computed from it are economically correct;
    close does not. Between corporate actions, the close/adjClose ratio
    for a ticker is roughly constant; a stock split or large dividend
    shows up as a sharp, discrete jump in this ratio. This is the
    empirical justification for using adjClose (not close) in
    daily_returns() - not just an assumption.
    """
    df = equity.sort_values(["ticker", "date"]).copy()
    df["close_adj_ratio"] = df["close"] / df["adjClose"]
    df["ratio_pct_change"] = df.groupby("ticker")["close_adj_ratio"].pct_change()

    flagged = df[df["ratio_pct_change"].abs() > ratio_change_threshold].copy()
    flagged = flagged[["ticker", "date", "close", "adjClose",
                       "close_adj_ratio", "ratio_pct_change"]]
    return flagged.sort_values(["ticker", "date"]).reset_index(drop=True)
