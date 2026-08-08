# Prompt log - independent validation of fund performance numbers

## What I wanted
After matching the backtest design to the lecture (prompt_log_01), sanity-check the
resulting `performance_metrics.csv` against the lecture's own reference Sharpe
table (slide 22), since the underlying raw dataset is confirmed identical (same
row counts, same duplicate counts as the lecture's stated figures).

## Prompt(s)
Asked Claude to compare my performance_metrics.csv against the reference table and
explain any large differences.

## What the assistant produced
Flagged that Crypto and Combined fund rankings diverged meaningfully from the
slide - most strikingly, Combined Max-Sharpe ranked BEST in my results (0.96)
but WORST in the lecture's reference (0.40).

## What was wrong or risky
A divergence like this is genuinely ambiguous from the metrics table alone - it
could mean a real bug in `oos_backtest()`/`performance_metrics()`, or it could
just mean crypto behaved differently in this specific out-of-sample window than
in the illustrative lecture example. The two explanations are not distinguishable
without an independent check.

## What I changed and why
Wrote `scripts/sanity_check_crypto.py`: recomputes Crypto Equal-Weight's growth of
$1 DIRECTLY from raw adjusted-close prices, completely bypassing `src/portfolios.py`
(equal-weight needs no estimation at all, so this is a genuinely independent
calculation, not just re-running the same code). The result matched
`performance_metrics.csv` almost exactly (CAGR 0.342 vs 0.341880, Sharpe 0.425 vs
0.425301), which confirms the pipeline is computing correctly, and that the
divergence from the lecture's illustrative numbers is a real characteristic of this
dataset's crypto returns (high volatility drag between arithmetic mean and realised
CAGR) - not a bug. Kept as report material for the critical-reflection section
rather than something to "fix".
