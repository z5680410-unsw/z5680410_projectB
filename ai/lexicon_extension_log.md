# Prompt log - custom finance lexicon extension (Station 3 innovation, unstructured side)

## What I wanted
Extend VADER's lexicon with our own finance-specific terms (brief, and slide 31:
"Extend the VADER lexicon with finance terms, then have your AI agent rate them and
keep the ones raters agree on") - separate from `finvader` (SentiBignomics + Henry),
which is a third-party package already used elsewhere, not something we curated
ourselves.

## Prompt(s)
Asked Claude to: (1) propose a candidate list of finance-specific words common in
headlines but likely missing from VADER's general-purpose lexicon, (2) rate each
candidate TWICE independently - once from a plain dictionary-definition framing,
once from a "how would markets typically react to a headline containing this word"
framing - and (3) keep a term only where both passes agree in sign and are within
1.5 points of each other on VADER's -4..+4 scale, discarding the rest.

## What the assistant produced
First pass included several multi-word phrases (e.g. "guidance cut", "record
revenue", "going concern") alongside single words.

## What was wrong or risky
Checked empirically (`SentimentIntensityAnalyzer().lexicon['multi word key'] = ...`
then compared `polarity_scores()` with and without it) whether VADER's tokenizer
actually looks up multi-word keys - it does not; the score was identical either way.
VADER only ever scores individual tokens. Every phrase-based candidate would have
silently done nothing if left in.

## What I changed and why
Dropped every multi-word candidate outright rather than trying to force a
single-word substitute for concepts like "going concern" (no single word captures
that meaning without misleading in other contexts). Re-ran the two-pass rating on
single-word candidates only. Verified against VADER's own base lexicon that the kept
terms were genuinely ABSENT (not silently overwriting an existing, possibly better,
VADER score) before finalising.

---

## Two-pass rating table

Pass 1 = plain dictionary-definition framing. Pass 2 = "how would markets typically
react to a headline containing this word" framing. Kept only if same sign AND
|pass1 - pass2| <= 1.5.

| term | pass 1 | pass 2 | agree? | final score |
|---|---|---|---|---|
| beat | +2.5 | +2.8 | yes | **+2.6** |
| upgrade / upgraded | +2.0 | +1.8 | yes | **+1.9** |
| downgrade / downgraded | -2.0 | -1.8 | yes | **-1.9** |
| buyback | +1.5 | +0.8 | yes | **+1.1** |
| writedown / impairment | -2.5 | -2.0 | yes | **-2.25** |
| dilutive | -1.5 | -1.2 | yes | **-1.35** |
| accretive | +1.5 | +1.3 | yes | **+1.4** |
| delisting / delisted | -3.5 | -3.6 | yes | **-3.55** |
| tailwind | +1.8 | +1.6 | yes | **+1.7** |
| headwind | -1.8 | -1.6 | yes | **-1.7** |
| outperform | +2.2 | +2.0 | yes | **+2.1** |
| underperform | -2.2 | -2.0 | yes | **-2.1** |
| shortfall | -2.3 | -2.5 | yes | **-2.4** |
| windfall | +2.5 | +2.3 | yes | **+2.4** |
| breach | -2.8 | -3.2 | yes | **-3.0** |
| restructuring | -1.5 | -0.5 | **no** (same sign, gap 1.0 - kept as REJECTED anyway: too context-dependent, see note) | rejected |
| layoffs | -1.8 | -1.0 | **no** (gap 0.8, but genuinely mixed in practice - cost discipline vs distress signal) | rejected |
| recall | -2.0 | -0.3 | **no** (gap 1.7 - "recall" is at least as often a generic memory verb as a product recall) | rejected |
| raised / raises | +1.5 | -0.3 | **no** (sign flips - "raised guidance" is positive, "raised concerns"/"raised doubts" is negative; word alone can't tell) | rejected |

`restructuring` and `layoffs` are flagged as rejected DESPITE passing the numeric
gap threshold, because on reflection both are genuinely context-dependent in real
financial reporting (can be framed as cost discipline/streamlining, positive, or as
a distress signal, negative) in a way a single-word score cannot resolve - the
numeric agreement was closer than the underlying semantic agreement, so the more
conservative call is to exclude them rather than force a number.

19 terms kept out of 23 reviewed (finalised list: `src/custom_lexicon.py`). This
custom set is merged into a THIRD analyzer alongside plain VADER and finVADER, so
`sentiment.py` can report all three side by side (`sentiment_vader`,
`sentiment_finvader`, `sentiment_finvader_custom`) and show whether our own
additions move the needle beyond the published finance lexicon.
