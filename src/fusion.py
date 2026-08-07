"""Station 3 - fusion: fold equity sentiment into the Equity Min-Variance fund.

Base fund: Equity Min-Variance, matching the course's own reference design (Week
10 revision lecture, slides 32-33). We keep the base fund's weights and move them
only a little, based on lagged news sentiment - a naive linear tilt is the
baseline; the intensity-confidence extension below (Fase 7) is the innovation.
"""
import numpy as np
import pandas as pd

from src import portfolios


# ---------------------------------------------------------------------------
# Lagged, look-ahead-safe sentiment/intensity panels
# ---------------------------------------------------------------------------

def build_lagged_ticker_sentiment(ticker_day_scores: pd.DataFrame,
                                   trading_calendar: pd.DatetimeIndex,
                                   score_col: str = "sentiment_finvader",
                                   lag_days: int = 1) -> pd.DataFrame:
    """Wide per-ticker daily sentiment panel (NaN on no-news days), shifted
    forward by `lag_days` TRADING days. The value stored at date t is really
    the score observed at date (t - lag_days) or earlier - every downstream
    calculation in this module reads from this already-lagged panel, so
    nothing later needs to remember to apply a lag again.
    """
    wide = ticker_day_scores.pivot_table(index="trading_date", columns="ticker",
                                          values=score_col, aggfunc="mean")
    wide = wide.reindex(pd.DatetimeIndex(trading_calendar))
    return wide.shift(lag_days)


def build_lagged_intensity(ticker_day_scores: pd.DataFrame,
                            trading_calendar: pd.DatetimeIndex,
                            lag_days: int = 1) -> pd.DataFrame:
    """Same lag treatment, for headline VOLUME (n_articles) rather than
    sentiment polarity - feeds the intensity-confidence extension (Fase 7).
    """
    wide = ticker_day_scores.pivot_table(index="trading_date", columns="ticker",
                                          values="n_articles", aggfunc="sum")
    wide = wide.reindex(pd.DatetimeIndex(trading_calendar))
    return wide.shift(lag_days)

# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def rolling_zscore(lagged_panel: pd.DataFrame, window: int = 63,
                    min_periods: int = 10) -> pd.DataFrame:
    """Rolling z-score per ticker, using the already-lagged panel (so no
    further shift is needed here - the window itself only ever looks
    backward from each date). NaN wherever a ticker has fewer than
    min_periods non-null observations in the window: not enough evidence
    to say anything, which is a different claim from "neutral".
    """
    mean = lagged_panel.rolling(window, min_periods=min_periods).mean()
    std = lagged_panel.rolling(window, min_periods=min_periods).std(ddof=0)
    return (lagged_panel - mean) / std.replace(0, np.nan)


def intensity_confidence(lagged_intensity: pd.DataFrame, window: int = 63,
                          min_periods: int = 10, lo: float = 0.5,
                          hi: float = 2.0) -> pd.DataFrame:
    """Innovation (Fase 7): a confidence multiplier per ticker/day, built
    from how a ticker's RECENT headline volume compares to its own
    EXPANDING historical baseline volume - both computed on the already-
    lagged intensity panel, so this stays look-ahead-safe by construction.

    This directly extends the Part A finding that headline INTENSITY (not
    just presence) carries information - outlier-return days had
    significantly more headlines than matched normal days. Here that same
    idea is reused as a trading-signal CONFIDENCE gate: a sentiment reading
    backed by unusually heavy coverage is treated as more informative than
    the same reading based on a single stray headline, so the tilt is
    amplified when coverage is elevated and dampened when it is thin.

    Ratio is clipped to [lo, hi] so one very quiet or very noisy ticker
    can't dominate; undefined ratios (a ticker with no baseline history
    yet) default to 1.0 (neutral - falls back to the plain tilt) rather
    than 0 (which would silently switch the tilt off).
    """
    recent = lagged_intensity.rolling(window, min_periods=1).sum()
    baseline = lagged_intensity.expanding(min_periods=min_periods).mean() * window
    ratio = (recent / baseline.replace(0, np.nan)).clip(lo, hi)
    return ratio.fillna(1.0)

# ---------------------------------------------------------------------------
# Tilt mechanics
# ---------------------------------------------------------------------------

def tilt_weights(base_weights: pd.Series, zscore_row: pd.Series, lam: float,
                  confidence_row: pd.Series = None) -> pd.Series:
    """w_tilted_i = w_base_i * (1 + lam * z_i * confidence_i).

    z is treated as 0 (no tilt for that name) wherever it's NaN -
    insufficient sentiment evidence defers to the base optimiser's own
    view rather than guessing. Result is clipped long-only and renormalised
    to sum to 1, exactly matching the course's own reference formula
    (slide 33), extended with an optional confidence multiplier.
    """
    z = zscore_row.reindex(base_weights.index).fillna(0.0)
    conf = (confidence_row.reindex(base_weights.index).fillna(1.0)
            if confidence_row is not None else 1.0)
    tilted = base_weights * (1.0 + lam * z * conf)
    tilted = tilted.clip(lower=0.0)
    total = tilted.sum()
    if total <= 0:
        return base_weights  # degenerate: tilt zeroed every name out, fall back to base
    return tilted / total

def apply_sentiment(base_result: dict, equity_returns: pd.DataFrame,
                     ticker_day_scores: pd.DataFrame,
                     trading_calendar: pd.DatetimeIndex, lam: float = 1.0,
                     zscore_window: int = 63, lag_days: int = 1,
                     use_intensity_confidence: bool = False,
                     score_col: str = "sentiment_finvader") -> dict:
    """Re-simulate the base fund's REALISED daily returns with sentiment-
    tilted weights at each of the base fund's own rebalance dates.

    `score_col` selects which sentiment column drives the tilt - by
    default finVADER, but passing "sentiment_finvader_custom" isolates the
    effect of the custom lexicon extension (Fase 7) alone: every other
    setting (lam, window, confidence) stays identical, so any difference
    in the result is attributable ONLY to the lexicon.

    Returns a dict in exactly the same shape as portfolios.oos_backtest(),
    so exhibits.py's fact-sheet functions work unchanged on a fused fund.
    Setting lam=0 must reproduce the base fund's returns exactly (a tilt
    of zero is a no-op) - see tests.
    """
    lagged_sent = build_lagged_ticker_sentiment(ticker_day_scores, trading_calendar,
                                                 score_col=score_col, lag_days=lag_days)
    zscore = rolling_zscore(lagged_sent, window=zscore_window)

    confidence = None
    if use_intensity_confidence:
        lagged_intensity = build_lagged_intensity(ticker_day_scores, trading_calendar,
                                                    lag_days=lag_days)
        confidence = intensity_confidence(lagged_intensity, window=zscore_window)

    base_weights = base_result["weights"]
    rebalance_dates = base_weights.index
    eq = equity_returns.sort_values("date").set_index("date")

    daily_rows, weight_rows = [], []
    for i, date in enumerate(rebalance_dates):
        z_row = (zscore.loc[date] if date in zscore.index
                 else pd.Series(dtype=float))
        conf_row = (confidence.loc[date] if confidence is not None and date in confidence.index
                    else None)
        current_weights = tilt_weights(base_weights.loc[date], z_row, lam, conf_row)
        weight_rows.append({"date": date, **current_weights.to_dict()})

        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        mask = (eq.index >= date) if next_date is None else (eq.index >= date) & (eq.index < next_date)
        for d in eq.index[mask]:
            ret = float(current_weights.reindex(eq.columns).fillna(0.0).to_numpy()
                        @ eq.loc[d].fillna(0.0).to_numpy())
            daily_rows.append({"date": d, "return": ret})

    daily_returns = pd.DataFrame(daily_rows).set_index("date")["return"]
    weights = pd.DataFrame(weight_rows).set_index("date")
    growth_of_dollar = (1.0 + daily_returns).cumprod()

    return {
        "daily_returns": daily_returns,
        "weights": weights,
        "growth_of_dollar": growth_of_dollar,
        "first_oos_date": daily_returns.index[0],
        "method": f"{base_result['method']}_tilt(lam={lam})",
        "periods_per_year": base_result["periods_per_year"],
    }

# ---------------------------------------------------------------------------
# Discovery / holdout evaluation (avoids the overfitting trap - slide 36)
# ---------------------------------------------------------------------------

def evaluate_on_window(result: dict, start: str, end: str,
                        periods_per_year: int = 252) -> dict:
    """performance_metrics() restricted to a date sub-window of an already-
    computed fund/tilt result - used to score a config on the DISCOVERY
    window without touching the HOLDOUT window until a config is chosen.
    """
    r = result["daily_returns"]
    r_window = r[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
    return portfolios.performance_metrics(r_window, periods_per_year=periods_per_year)
