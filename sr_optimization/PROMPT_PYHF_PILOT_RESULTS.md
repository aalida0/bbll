# Run-3 prompt-only pyhf comparison — pilot, 14 November 2026

These numbers were reported from two complete `pyhf==0.7.6`
diagnostic fits of the validated Run-3 SR candidate JSON.

**Model:** Median expected asymptotic CLs, aggregate prompt-background
`shapesys` MC-stat nuisance parameters only. Each channel is required to
have positive signed background per bin, (N_{\\mathrm{eff}}\\ge10) and
at least two (m_T(HH)) bins. No FakeSim, detector/theory systematics or
signal MC-stat uncertainty. `mu` is a factor multiplying the input
signal template; physical fb normalization is not independently certified.

| Mass GeV | 2-channel mu95 | 3-channel mu95 | two/three | Upper-limit decrease | Fits |
|---:|---:|---:|---:|---:|---|
| 600 | 95.68261248710787 | 62.24789709846798 | 1.5371220065 | 34.9434% | 9/9 and 27/27 |
| 4000 | 1.4693397999695779 | 1.3989799540299475 | 1.0502936770 | 4.7885% | 9/9 and 18/18 |

All single-channel fits also completed:
- 600 GeV: resolved 11/11, inclusive boosted 11/11,
  boosted 1tag 11/11, boosted 2tag 9/9;
- 4000 GeV: resolved 3/3, inclusive boosted 4/4,
  boosted 1tag 4/4, boosted 2tag 2/2.

**Interpretation:** The 3-channel (resolved, boosted_1tag, boosted_2tag)
architecture is favored in both *prompt-only* pilot points; the gain
is substantially larger at 600 GeV.

**Limitations:** The combined fit selects only the best three
single-channel candidates per channel, i.e. it is not an exhaustive
global minimum. Low-mass gains may shrink after FakeSim and correlated
systematics, or when MC-stat/shape uncertainty is revisited. The 
statistical preference cannot justify freezing publication SRs.
A cross-era/mass campaign, robust nuisance treatment and fit stability
checks are mandatory before a final result.
