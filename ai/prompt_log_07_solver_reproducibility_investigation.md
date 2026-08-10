# Prompt log - backtest results changed between two identical-code runs

## What I wanted
Understand why `run_part_b.py`, run twice on the same machine with (as far
as I could tell) unchanged code, produced meaningfully different fund
performance numbers - Equal-Weight identical both times, but every
solver-based method (Min-Variance, Max-Sharpe, Risk Parity, Mean-CVaR)
diverged, and the `max_sharpe optimiser did not converge` warning that had
appeared in every prior run disappeared entirely.

## Prompt(s)
Asked Claude to help diagnose the discrepancy rather than just re-running
the pipeline and hoping the numbers matched.

## What the assistant produced
An initial hypothesis that a scipy version difference was responsible
(`pip show scipy` showed 1.18.0 on my machine vs 1.17.1 in Claude's own test
environment), with a suggested downgrade.

## What was wrong or risky
Claude tested this hypothesis directly - installing scipy 1.18.0 in its own
sandbox and re-running the exact same synthetic backtest under both 1.17.1
and 1.18.0 - and the results came out byte-identical on synthetic
(uncorrelated random) data. That result did not match what I was seeing on
my real, correlated financial data, so the scipy-version hypothesis, while
plausible, was not actually confirmed - accepting it without the direct A/B
test would have been an unverified guess presented as an answer. I then
downgraded scipy on my own machine and confirmed via a direct `python.exe -m
pip` check (to rule out a multi-Python-install mismatch) that 1.17.1 was
genuinely active, and the results still did not change - ruling scipy out
completely.

## What I changed and why
Rather than keep guessing at individual library versions indefinitely,
Claude and I agreed on a practical stopping point: run the pipeline twice in
a row with no changes to confirm it was at least internally reproducible on
my machine (it was - both runs matched exactly), adopt that stable run as
the canonical set of numbers for the report, and document the investigation
itself as a finding rather than a solved mystery. This is recorded in Part 6
as evidence that solver behaviour on ill-conditioned real covariance
matrices is genuinely environment-sensitive (matching the brief's own
"solver scaling" warning), and `requirements.txt` was updated to pin exact
scipy/pandas versions so the reported numbers are reproducible going
forward, even though the root cause of the original divergence was not
fully isolated.
