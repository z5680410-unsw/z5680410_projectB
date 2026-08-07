# CLAUDE.md - Quantvestment, Part B

Instructions for Claude when working in this folder (FINS5545 Part B - Sandy, z5680410).

## Project context
Quantvestment: systematically managed multi-asset funds (US equities + crypto),
out-of-sample backtested, plus a news-sentiment index and a fusion tilt, served
through a Streamlit app. Full brief: PROJECT_BRIEF.md. Data schema and known
quirks: context/DATA_GUIDE.md.

## How I work with Claude on this project
- I paste actual file contents before asking for changes - don't assume file
  contents from memory or from earlier in the conversation.
- I run everything myself in PowerShell (`python scripts/run_part_b.py`,
  `streamlit run streamlit_app.py`) and paste back the real terminal output.
  Wait for that output before proposing the next step.
- Prefer complete file replacements or clearly marked find/replace blocks over
  vague instructions - I copy-paste directly into PyCharm.
- Test new logic (edge cases, look-ahead safety, weight validity) before
  handing it to me, and say what was tested.
- Work in small, verifiable steps rather than large multi-file changes at once,
  especially for src/portfolios.py, src/fusion.py, and src/sentiment.py, where a
  silent look-ahead bug is the most costly mistake to ship.

## Non-negotiable project conventions
- No look-ahead: weights/signals at date t use only data strictly before t.
- Long-only, fully invested funds: weights sum to 1, w >= 0.
- Annualise with periods_per_year=252 for equity/combined, 365 for crypto.
- Reuse src/plotstyle.py (FT design system) for every figure - do not
  introduce a different visual style.
- Exact required filenames (do not rename): results/data/fund_returns.csv,
  fund_weights.csv, sector_sentiment_index.csv, results/tables/performance_metrics.csv.
- streamlit_app.py reads ONLY precomputed results/ files - never import nltk,
  never recompute a backtest or sentiment score inside the app.
- Data loads only via src/data_access.py - never commit raw .parquet data.

## AI workflow documentation
Detailed prompt logs (what was asked, what Claude produced, what was wrong,
what I changed) live in ai/ - see ai/lexicon_extension_log.md for the custom
finance-lexicon rating process specifically.
