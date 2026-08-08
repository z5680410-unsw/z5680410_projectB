# Prompt log - rolling vs expanding window backtest design

## What I wanted
Verify the Part B build actually matches the course's own reference material (Week
10 revision lecture slides), after building funds using a design that was valid
per the written brief but not yet cross-checked against the lecture itself.

## Prompt(s)
Asked Claude to re-check every step-by-step decision so far against the uploaded
week10_revision_fins5545.pdf slide deck specifically, not just the written
PROJECT_BRIEF.md.

## What the assistant produced
The original `oos_backtest()` used a fixed 252-day ROLLING window (re-estimate on
only the trailing 252 days at each rebalance) - a legitimate choice under the
brief's "choose your own window type" freedom, already tested and working.

## What was wrong or risky
Slides 11-12 of the lecture explicitly describe an EXPANDING window ("the training
window grows to include all data up to that point... re-estimate on ALL data seen
so far"), with monthly rebalancing and a stated first live date of January 2021 -
this is the course's own reference design, and diverges from what was already built
and delivered.

## What I changed and why
Rewrote `oos_backtest()` to use an expanding window (estimate on every row strictly
before the current rebalance date, not just the trailing N) with rebalancing on the
first trading day of each new calendar month. Simplified the function signature at
the same time (one shared `first_live_date="2021-01-01"` instead of separate
`estimation_window`/`rebalance_every` per universe). This also meant the resulting
`performance_metrics.csv` could be directly compared against the lecture's own
Sharpe reference table (slide 22) as a validity check, since the underlying dataset
is confirmed identical (matching row counts and duplicate counts) - see
prompt_log_05.
