# Prompt log - .gitignore silently excluding required CSVs

## What I wanted
Confirm the provided starter `.gitignore` actually protects raw data while still
allowing the required `results/` CSVs to be committed, before running `git add`
for real on the Part B repo.

## Prompt(s)
Asked Claude to check the `.gitignore` was safe before the first commit.

## What the assistant produced
Reproduced the exact starter `.gitignore` in an isolated throwaway git repo (not
the real project) with placeholder `results/data/*.csv` and `results/tables/*.csv`
files, then ran `git check-ignore -v` and `git status` against it.

## What was wrong or risky
Two bugs found by testing, not by reading the file:
1. `*.csv` blocks every CSV; the starter's `!results/data` negation only
   un-ignores the DIRECTORY `results/data` itself - it does NOT rescue the
   individual `.csv` files inside it from the separate `*.csv` rule, and
   `results/tables/` (where `performance_metrics.csv` - a REQUIRED filename -
   lives) had no negation at all.
2. The `data/` pattern (meant to block a root-level raw-data cache folder) has no
   leading slash, so it matches a directory named "data" at ANY depth - including
   `results/data/`, blocking it a second, different way even after fixing #1.

Both would have caused the required output files to be silently missing from the
GitHub submission with no error or warning at commit time.

## What I changed and why
Changed the CSV rule to `*.csv` + `!results/**/*.csv` (rescues every CSV under
results/, both data/ and tables/), and anchored the raw-data rule to `/data/`
(leading slash = project root only, so it no longer matches nested `results/data/`).
Verified the fix the same way - fresh isolated test repo, `git add -A`, `git status`
- before touching the real repository.[prompt_log_template.md](prompt_log_template.md)[prompt_log_template.md](prompt_log_template.md)
