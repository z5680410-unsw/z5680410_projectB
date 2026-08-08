# Prompt log - finVADER scoring performance

## What I wanted
Score all ~146,830 headlines with finVADER (VADER + published finance lexicons),
matching the tool the course's own reference material uses.

## Prompt(s)
Asked Claude to build `sentiment.score_headlines()` using the `finvader` package.

## What the assistant produced
An initial design that called the `finvader` package's own `finvader()` function
once per headline.

## What was wrong or risky
Benchmarked at ~6ms/headline, which extrapolates to ~15 minutes for the full
headline panel - `finvader()` rebuilds the merged SentiBignomics+Henry lexicon and
instantiates a brand-new `SentimentIntensityAnalyzer` on EVERY call, all of which
is pure repeated setup work.

## What I changed and why
Built the merged lexicon and a single `SentimentIntensityAnalyzer` ONCE, then
reused that same instance for every headline via `.polarity_scores()`. First
verified this gives byte-identical scores to calling `finvader()` directly on a
sample of headlines (confirms no behaviour change), then benchmarked the reused
version at ~0.08ms/headline - about 80x faster, full panel in ~10-20 seconds
instead of ~15 minutes.
