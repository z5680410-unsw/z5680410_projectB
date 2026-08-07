"""Custom finance-term lexicon extension for VADER (Station 3 innovation,
unstructured side - see the brief and slide 31: "Extend the VADER lexicon
with finance terms, then have your AI agent rate them and keep the ones
raters agree on.").

This is DELIBERATELY separate from finvader (SentiBignomics + Henry, already
used elsewhere in src/sentiment.py) - finvader is a third-party package;
this is a small, independently curated set of finance-specific terms, rated
through two independent passes and kept only where both passes agreed. The
full rating record (including REJECTED terms and why) is in
ai/lexicon_extension_log.md - this file holds only the final, agreed dict.

VADER scores individual TOKENS (its tokenizer does not match multi-word
lexicon keys - verified empirically before building this list), so every
entry here is a single word, not a phrase. Scores are on VADER's own
-4..+4 scale (checked against vader_lexicon.txt's own range before choosing
magnitudes).
"""

CUSTOM_FINANCE_LEXICON = {
    "beat": 2.6,
    "upgrade": 1.9,
    "upgraded": 1.9,
    "downgrade": -1.9,
    "downgraded": -1.9,
    "buyback": 1.1,
    "writedown": -2.25,
    "impairment": -2.25,
    "dilutive": -1.35,
    "accretive": 1.4,
    "delisting": -3.55,
    "delisted": -3.55,
    "tailwind": 1.7,
    "headwind": -1.7,
    "outperform": 2.1,
    "underperform": -2.1,
    "shortfall": -2.4,
    "windfall": 2.4,
    "breach": -3.0,
}
