# Prompt log - Combined fund leading-NaN calendar mismatch

## What I wanted
Sanity-check the very first real `run_part_b.py` output before trusting it.

## Prompt(s)
Pasted the real terminal output from `python scripts/run_part_b.py` and asked
Claude to confirm the fund-grid numbers were correct.

## What the assistant produced
`oos_backtest()`'s NaN-handling dropped the panel's leading row using
`dropna(how="all")` - correct for the single-universe (equity-only, crypto-only)
panels, which each have exactly one all-NaN leading row.

## What was wrong or risky
For the COMBINED panel specifically, equity's leading-NaN row and crypto's
leading-NaN row fall on DIFFERENT calendar dates (crypto's raw data starts one
calendar day before equity's first trading day). At equity's first date, the
crypto columns were already valid (non-NaN), so the row was NOT all-NaN and
`dropna(how="all")` silently kept it - letting 50 equity assets enter the first
estimation window with an artificial "0% return" that day. This surfaced as
`Combined` funds showing a different `first_oos_date`/`n_days` than `Equity` funds
in the real run output, when they should have matched exactly.

## What I changed and why
Replaced the leading-row check with "drop rows until every asset column has a
value" (`df[asset_cols].notna().all(axis=1).idxmax()`), which correctly handles a
panel whose different sub-universes have leading NaNs on different dates. Verified
with a hand-built reproduction of the exact scenario, then confirmed on the real
data: Equity and Combined now report identical `first_oos_date` (2021-01-04) and
`n_days` (753).
