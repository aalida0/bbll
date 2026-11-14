# Prompt-only CLs architecture comparison, full Run-2/Run-3 scan

**Analysis date:** 14 November 2026. **Status: 32/32 completed.**

Analyst-reported outputs from
`sr_optimization/prompt_pyhf_top3_campaign.json`. This reproduces
the **prompt-background-only** HistFactory model with aggregate
background MC-stat `shapesys` nuisances. No fake/nonprompt backgrounds,
signal MC-stat nuisances, or correlated detector/theory systematics.
`mu95` scales the input signal template and is **not independently
certified in femtobarns**.

Each architecture chooses the best combination of the **three best
single-channel** BDT score candidates. `exhaustive=True` means all
surviving candidates happened to fit within the shortlist for that mass.
The two architectures are mutually exclusive:
- two-channel: resolved + inclusive boosted;
- three-channel: resolved + boosted 1-tag + boosted 2-tag.

Define `R = mu95_two / mu95_three`; `R > 1` favors the split.
The reduction in upper limit is `100*(1-1/R)` percent.

| Era | Mass GeV | 2ch mu95 | 3ch mu95 | Ratio | Reduction % | Exhaustive |
|---|---:|---:|---:|---:|---:|---|
| Run 2 | 400 | 1250.5 | 1239.7 | 1.0087 | +0.87 | No |
| Run 2 | 500 | 292.49 | 275.71 | 1.0609 | +5.74 | No |
| Run 2 | 600 | 137.54 | 99.281 | 1.3853 | +27.81 | No |
| Run 2 | 700 | 84.893 | 49.523 | 1.7142 | +41.66 | No |
| Run 2 | 800 | 53.814 | 29.398 | 1.8305 | +45.37 | No |
| Run 2 | 900 | 38.969 | 22.534 | 1.7293 | +42.17 | No |
| Run 2 | 1000 | 30.308 | 19.199 | 1.5786 | +36.65 | No |
| Run 2 | 1200 | 18.8 | 12.331 | 1.5246 | +34.41 | No |
| Run 2 | 1400 | 13.514 | 11.963 | 1.1296 | +11.47 | No |
| Run 2 | 1600 | 11 | 9.9701 | 1.1033 | +9.37 | No |
| Run 2 | 1800 | 10.119 | 9.6423 | 1.0494 | +4.71 | No |
| Run 2 | 2000 | 8.3595 | 8.2889 | 1.0085 | +0.84 | No |
| Run 2 | 2500 | 5.2193 | 5.7771 | 0.9034 | -10.69 | No |
| Run 2 | 3000 | 4.1123 | 4.7079 | 0.8735 | -14.48 | **Yes** |
| Run 2 | 3500 | 3.5415 | 3.4226 | 1.0348 | +3.36 | No |
| Run 2 | 4000 | 2.9611 | 3.0946 | 0.9568 | -4.51 | **Yes** |
| Run 3 | 400 | 1095.1 | 1087 | 1.0074 | +0.74 | No |
| Run 3 | 500 | 252.47 | 238.41 | 1.0590 | +5.57 | No |
| Run 3 | 600 | 95.683 | 62.248 | 1.5371 | +34.94 | No |
| Run 3 | 700 | 62.266 | 38.939 | 1.5990 | +37.46 | No |
| Run 3 | 800 | 43.384 | 24.403 | 1.7778 | +43.75 | No |
| Run 3 | 900 | 29.34 | 16.999 | 1.7260 | +42.06 | No |
| Run 3 | 1000 | 21.01 | 12.297 | 1.7085 | +41.47 | No |
| Run 3 | 1200 | 12.697 | 7.8023 | 1.6274 | +38.55 | No |
| Run 3 | 1400 | 9.6647 | 6.9573 | 1.3892 | +28.01 | No |
| Run 3 | 1600 | 9.0554 | 7.3868 | 1.2259 | +18.43 | No |
| Run 3 | 1800 | 7.5735 | 6.6674 | 1.1359 | +11.96 | No |
| Run 3 | 2000 | 5.6709 | 5.7082 | 0.9935 | -0.66 | No |
| Run 3 | 2500 | 3.431 | 3.24 | 1.0589 | +5.57 | No |
| Run 3 | 3000 | 2.2955 | 2.4182 | 0.9492 | -5.35 | No |
| Run 3 | 3500 | 1.7488 | 1.7393 | 1.0055 | +0.54 | No |
| Run 3 | 4000 | 1.4693 | 1.399 | 1.0503 | +4.79 | No |

## Physics interpretation

- The split has a strong **prompt-only** advantage around
  **600–1200 GeV**, peaking at 800 GeV in both eras.
- At 400–500 GeV the gain is negligible to modest.
- Above about 2000 GeV, it is small, fluctuates in sign,
  and may be dominated by sparse weighted MC and adaptive bin merges.
- Run-2 3000/4000 GeV fits are **exhaustive** over surviving candidates:
  their negative ratios cannot be explained solely by a top-three shortlist.

## Required review before publication

1. Run the matched-score-threshold/common-`m_T(HH)`-bin category
   ablation (`sr_optimization.matched_split_control`) to isolate the
   category split from independent threshold/binning optimization.
2. Do not force a three-channel fit in mass regions where either
   1-tag or 2-tag cannot satisfy `N_eff >= 10` with >=2 bins.
3. Add FakeSim/nonprompt backgrounds, correlated experimental/theory
   systematics and signal MC-stat, then repeat the architecture choice.
4. Audit sidecar signal normalization before assigning `fb` to `mu95`.
5. Avoid choosing the final mass-dependent architecture using only this
   prompt-only table; predefine and validate the selection policy.
