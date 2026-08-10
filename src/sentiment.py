"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.

NOTE on the lag: sector_sentiment_index() and market_fear_greed_index() below are
DESCRIPTIVE exhibits - they show sentiment as observed on each date, with no
look-ahead concern of their own. The >=1 trading-day lag the brief requires applies
specifically when a sentiment value is used to make a TRADING decision - that lag is
applied in src/fusion.py (Fase 6), not here.
"""
from pathlib import Path

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from finvader.finvader import Merge
from finvader.SentiBignomics import lexicon1
from finvader.Henry import lexicon2

from src.custom_lexicon import CUSTOM_FINANCE_LEXICON

RESULTS_DATA_DIR = Path(__file__).resolve().parent.parent / "results" / "data"
RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _build_analyzers():
    """One-time setup: three analyzers sharing the same underlying VADER
    algorithm, so any difference between them is attributable ONLY to their
    lexicon:
        plain   - stock VADER, no finance lexicon
        finance - "finVADER": VADER + two PUBLISHED finance lexicons
                  (SentiBignomics, Henry 2008) merged in - matches the
                  `finvader` package used in the course's own reference
                  material (Loughran & McDonald, 2011, motivate why a
                  finance-specific lexicon matters at all: about half of
                  finance headlines score neutral under plain VADER).
        finance_custom - finance + OUR OWN small set of additional finance
                  terms (see src/custom_lexicon.py and
                  ai/lexicon_extension_log.md for the two-pass AI rating
                  and agreement process that produced it) - this is the
                  innovation layer, distinct from the third-party finvader
                  package: terms we proposed and rated ourselves.

    Built ONCE and reused for every headline, rather than calling the
    `finvader` package's own finvader() function per headline: that
    function rebuilds the merged lexicon and a fresh
    SentimentIntensityAnalyzer on EVERY call, which benchmarks at
    ~6ms/headline (~15 minutes for the full ~146,830-headline panel).
    Reusing one instance gives IDENTICAL scores (verified against
    finvader() directly) at ~0.08ms/headline (~10-20 seconds).
    """
    nltk.download("vader_lexicon", quiet=True)
    plain = SentimentIntensityAnalyzer()

    sentibignomics = lexicon1()
    constant = 0.1  # matches finvader's own scaling constant
    sentibignomics = {k: v * constant for k, v in sentibignomics.items()}
    henry = lexicon2()
    finance_lexicon = Merge(sentibignomics, henry)
    finance = SentimentIntensityAnalyzer()
    finance.lexicon.update(finance_lexicon)

    finance_custom = SentimentIntensityAnalyzer()
    finance_custom.lexicon.update(finance_lexicon)
    finance_custom.lexicon.update(CUSTOM_FINANCE_LEXICON)

    return plain, finance, finance_custom


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score every headline with VADER, finVADER, and finVADER+custom, then
    average to ONE score per (ticker, trading_date) - the first step in
    building the sector index (score -> ticker-day average -> sector
    average).

    Do NOT strip casing/punctuation first - VADER's rules rely on them
    (capitals, negation, booster words). `panel` is the full headline-level
    panel from features.assemble_headline_panel (~146,830 rows), not the
    500-row sample saved to disk.
    """
    plain, finance, finance_custom = _build_analyzers()

    titles = panel["title"].fillna("")
    scored = panel.copy()
    scored["sentiment_vader"] = titles.apply(lambda t: plain.polarity_scores(t)["compound"])
    scored["sentiment_finvader"] = titles.apply(lambda t: finance.polarity_scores(t)["compound"])
    scored["sentiment_finvader_custom"] = titles.apply(
        lambda t: finance_custom.polarity_scores(t)["compound"])

    daily = (scored.groupby(["ticker", "sector", "trading_date"])
             .agg(sentiment_vader=("sentiment_vader", "mean"),
                  sentiment_finvader=("sentiment_finvader", "mean"),
                  sentiment_finvader_custom=("sentiment_finvader_custom", "mean"),
                  n_articles=("n_articles", "first"))
             .reset_index())
    return daily


def sector_sentiment_index(ticker_day_scores: pd.DataFrame,
                            trading_calendar: pd.DatetimeIndex,
                            score_col: str = "sentiment_finvader") -> pd.DataFrame:
    """Daily sentiment index per sector: equal-weight the tickers within
    each sector (every ticker with a score that day counts once,
    regardless of how many headlines it had - that headline-level
    averaging already happened in score_headlines).

    No-headline-day treatment (stated and justified, per the brief):
    ticker-days with no headline are left as NaN here, not filled with a
    neutral 0 or carried forward. A missing headline is not the same claim
    as "sentiment was exactly neutral that day" - filling it would invent
    information. The report/app exhibit instead smooths over these gaps
    with a rolling average (matching the course's own reference exhibit),
    which is a form of treatment that doesn't fabricate a specific daily
    value. (Fase 6's fusion step, which needs a complete daily series to
    compute a rolling z-score, makes its OWN explicit fill decision -
    documented separately there.)

    Returns a WIDE panel: one row per equity trading day ('date' column),
    one column per sector.
    """
    sector_daily = (ticker_day_scores.groupby(["sector", "trading_date"])[score_col]
                     .mean().reset_index())
    wide = sector_daily.pivot(index="trading_date", columns="sector", values=score_col)
    wide = wide.reindex(pd.DatetimeIndex(trading_calendar))
    wide.index.name = "date"
    return wide.reset_index()


def market_fear_greed_index(ticker_day_scores: pd.DataFrame,
                             trading_calendar: pd.DatetimeIndex,
                             score_col: str = "sentiment_finvader",
                             smoothing_window: int = 21) -> pd.DataFrame:
    """A market-wide fear-and-greed gauge: average sentiment across ALL
    tickers each day (not grouped by sector), smoothed with a rolling
    window, then rescaled onto 0-100 and separately standardised
    (z-score) - matching the course's own reference exhibit. The raw
    level sits above "neutral" (50) on the large majority of days (news
    coverage has a positive baseline), so the STANDARDISED series is what
    actually reveals the fear spikes; both are returned.
    """
    market_daily = ticker_day_scores.groupby("trading_date")[score_col].mean()
    market_daily = market_daily.reindex(pd.DatetimeIndex(trading_calendar))
    smoothed = market_daily.rolling(smoothing_window, min_periods=1).mean()

    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo < 1e-9:
        # Degenerate case (near-zero variance in the smoothed series, e.g. a
        # very short or unusually uniform sample): min-max rescaling would
        # divide by ~0. Fall back to a flat neutral reading rather than NaN.
        level_0_100 = pd.Series(50.0, index=smoothed.index)
    else:
        level_0_100 = (smoothed - lo) / (hi - lo) * 100.0

    std = smoothed.std(ddof=0)
    if std < 1e-9:
        standardised = pd.Series(0.0, index=smoothed.index)
    else:
        standardised = (smoothed - smoothed.mean()) / std

    out = pd.DataFrame({
        "date": pd.DatetimeIndex(trading_calendar),
        "level_0_100": level_0_100.values,
        "standardised": standardised.values,
    })
    return out
