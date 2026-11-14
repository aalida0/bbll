# Prompt-only SR candidate screening — 14 November 2026

**Data source:** Analyst-reported `prompt_screen_campaign.json` outputs
from UChicago AF for both eras. Covers 16 resonance masses,
400–4000 GeV. These are counts of threshold candidates satisfying
preliminary template validity, **not expected limits**.

Acceptance requirements:
- Signed aggregate prompt-background yield > 0 in each retained bin;
- (N_{\\mathrm{eff}}=(\\sum w)^2/(\\sum w^2)\\ge 10) per bin;
- At least two retained (m_T(HH)) shape bins;
- Resolved and boosted use distinct model banks, with boosted 1-tag and
  2-tag treated as alternatives to the inclusive boosted category.

## Run 2: surviving thresholds

| mX (GeV) | Resolved /11 | Boosted /11 | Boosted 1-tag /11 | Boosted 2-tag /9 |
|---:|---:|---:|---:|---:|
| 400 | 11 | 11 | 11 | 9 |
| 500 | 11 | 11 | 11 | 9 |
| 600 | 11 | 11 | 11 | 9 |
| 700 | 11 | 11 | 11 | 9 |
| 800 | 11 | 11 | 11 | 9 |
| 900 | 10 | 11 | 11 | 9 |
| 1000 | 9 | 11 | 11 | 8 |
| 1200 | 7 | 11 | 11 | 7 |
| 1400 | 5 | 10 | 10 | 6 |
| 1600 | 6 | 8 | 8 | 5 |
| 1800 | 4 | 7 | 7 | 5 |
| 2000 | 5 | 6 | 6 | 5 |
| 2500 | 3 | 4 | 4 | 3 |
| 3000 | 3 | 3 | 2 | 3 |
| 3500 | 4 | 2 | 2 | 2 |
| 4000 | 3 | 2 | 1 | 2 |

## Run 3: surviving thresholds

| mX (GeV) | Resolved /11 | Boosted /11 | Boosted 1-tag /11 | Boosted 2-tag /9 |
|---:|---:|---:|---:|---:|
| 400 | 11 | 11 | 11 | 9 |
| 500 | 11 | 11 | 11 | 9 |
| 600 | 11 | 11 | 11 | 9 |
| 700 | 11 | 11 | 11 | 9 |
| 800 | 11 | 11 | 11 | 9 |
| 900 | 11 | 11 | 11 | 9 |
| 1000 | 9 | 11 | 11 | 9 |
| 1200 | 8 | 11 | 11 | 7 |
| 1400 | 6 | 11 | 11 | 7 |
| 1600 | 6 | 10 | 10 | 6 |
| 1800 | 5 | 9 | 9 | 5 |
| 2000 | 4 | 11 | 9 | 4 |
| 2500 | 4 | 7 | 7 | 3 |
| 3000 | 3 | 6 | 5 | 3 |
| 3500 | 3 | 4 | 3 | 2 |
| 4000 | 3 | 4 | 4 | 2 |

### Interpretation and required follow-ups

All configured threshold candidates survive through 800 GeV in both eras.
At high masses, acceptance falls rapidly: Run-2 boosted 1-tag retains
**1/11** at 4000 GeV, while Run-3 inclusive boosted retains **4/11**.
A surviving candidate does not establish useful sensitivity; some may be
severely merged in (m_T(HH)), and per-process background MC statistics,
fake backgrounds, signal MC-stat, and detector/theory uncertainties are
still untested.

Next: perform separate **two-channel versus three-channel prompt-only**
Asimov-CLs comparisons in `sr_optimization.compare_architectures`.
Only after the full fake and systematic model is available can SR
thresholds be frozen for a publication likelihood.
