# Run-3 FakeSim: post-merge projection and frozen-threshold CLs diagnostics

## Verified input status

On 2026-11-15 the 400-file / 4,000,000-event Run-3 FakeSim strict merge
reported PASS with public e-mu proxy anchor closure. The four calibrated
normalization factors computed in that complete source campaign were:

| model | kappa |
|---|---:|
| dror2016_global | 4.851040122190293 |
| dror2016_bonly | 12.127906666872477 |
| curtin2015_mu_A__dror_alpha | 0.2430399448921402 |
| curtin2015_mu_B__dror_alpha | 0.24444972451602703 |

These factors reproduce the historical source/model calculation. This is
an internal computation and normalization consistency closure, NOT proof
of detector-level fake-lepton efficiency or SR transfer accuracy.

Frozen input files:

    /data/aalida/bbll_rerun_20261114/fake_backgrounds/merged/run3_full_raw_fakesim.json
    /data/aalida/bbll_rerun_20261114/fake_backgrounds/merged/run3_full_raw_fakesim.npz
    /data/aalida/bbll_rerun_20261114/fake_backgrounds/run3_candidate_grid.json

These files are READ ONLY. Do not rerun FakeSim or regenerate raw sources.

## Step 1 — project all benchmark fake yields onto current mT bins

    cd ~/bbll
    git pull --ff-only
    source .venv39/bin/activate

    python -m fake_backgrounds.project_merged \
      --output-root /data/aalida/bbll_rerun_20261114 \
      --mass 800

This projects ALL valid frozen SR score candidates for the selected mass
(the complete Run-3 score inventory contains 515 candidates over 16 masses)
onto their exact existing prompt mT(HH) edges. It refuses nonmaster edges,
mismatched score thresholds, stale candidate grids, corrupt merged NPZ SHA,
or inconsistent normalization factors. Source-event master covariance
is transformed as G @ C @ G.T, preserving within-event correlations
across the newly merged mT bins. No clipping of negative MC yields.

Output:
    /data/aalida/bbll_rerun_20261114/fake_backgrounds/diagnostics/run3/m800_fake_candidates.json

## Step 2 — evaluate current prompt-only winner cuts with fakes

    python -m fake_backgrounds.fixed_cut_diagnostic \
      --output-root /data/aalida/bbll_rerun_20261114 \
      --mass 800

Output:
    /data/aalida/bbll_rerun_20261114/fake_backgrounds/fixed_cut_diagnostics/run3/m800_fake_aware_fixed_cut.json

This uses the prior exhaustive expected-CLs chosen cuts for the two
exclusive architectures: resolved+boosted (two channels) or
resolved+boosted_1tag+boosted_2tag (three channels). It compares the
unchanged prompt-only median mu95 to a CONDITIONAL fake-aware median mu95
for the exact same signal templates, score cuts and mT edges.

Provisional MC-only nuisance model:

- Nominal fake = public-anchor-calibrated dror2016_global;
- 50% common fake normalization histosys (public e-mu anchor uncertainty);
- single common histosys from binwise min/max of all four calibrated
  benchmark total-background templates; this is a conservative envelope
  and does not represent a physical continuous benchmark interpolation;
- MC-stat shapesys from prompt sumw2 plus exact event-level FakeSim
  projected covariance diagonal. Off-diagonal correlation gate must PASS.

Hard blockers: any nonpositive bkg or histosys endpoint, <2 mT bins,
total background Neff<10, max off-diagonal background MC correlation >5%,
or an integrated covariance uncertainty difference >5%. In such cases
the script writes BLOCKED_TEMPLATE_GATE/BLOCKED_FIT and does not clip,
assign zero uncertainty, or manufacture a fake-aware expected limit.

The script prints per-channel prompt yield, calibrated fake yield,
fake fraction, min combined Neff and covariance correlation, plus
two- and three-channel expected-mu95 ratios when valid.

## Limitations before publication

These are NOT final cross-section limits in fb. The frozen signal
template strength units are not independently certified as fb in this
workflow. The study also lacks remaining detector/theory nuisances,
signal MC statistics and quantified shared-source covariance between
exclusive channels. We have not yet audited possible physical
double-counting with prompt ordinary background MC samples.

The ATLAS e-mu calibration is a model-dependent inclusive public
anchor. It does not automatically validate Delphes fake efficiencies
in a b-tagged, high-BDT-score bbll signal region. Report the four
benchmark spread and high-score MCstat honestly.

## Step 3 — once the fixed cuts are audited

Rebuild adaptive mass-dependent mT binning on prompt+fake covariance
and optimize the full 515 candidate-grid thresholds using a nuisance-aware
likelihood. Compare robust combined expected limits to the existing
prompt-only exhaustive results. Run-2 FakeSim requires separate 13 TeV
source validation/production; do not transfer 13.6 TeV source yields.
