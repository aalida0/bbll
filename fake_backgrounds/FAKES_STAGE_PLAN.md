# Fake/nonprompt leptons — bbll phenomenology Stage 4

> **2026-11-14 correction — PRIMARY PATH IS LEGACY FAKESIM REUSE.**
> The previous F0–F4 SS/OS transfer plan below is retained as an optional
> diagnostic only; it is not the proposed primary fake implementation.
> The existing `aalida0/bbll-ml-sr` repository already contains a full
> simulation-based jet-to-lepton FakeSim chain and historical 4-million-event
> Run-3 production. Reuse its actual object-level jet fake mechanics,
> calibrations, benchmark model variations, covariance and fake-aware
> pyhf machinery. **Do not replace it with an invented or new SS estimator.**
>
> Verified legacy modules:
>
> - `scripts/data/fakesim_reco.py` and
>   `scripts/data/fakesim_hbb_topology.py` — jet-to-fake lepton injection
>   and exclusive Higgs reconstruction.
> - `scripts/data/fakesim_public_benchmarks.py` — Dror/Curtin
>   public-literature benchmarks; source-flavour effects and charge weights.
> - `scripts/data/produce_run3_fakesim_benchmark_shard.py` — 4M-source
>   Run-3 signal-side replica and sample-scale template producer.
> - `scripts/stats/fakesim_background.py` — event-level second moments,
>   bin merging, master-template statistics.
> - `scripts/stats/fakesim_likelihood.py` — correlated fake normalization
>   and model-envelope histosys.
>
> **Blocking difference confirmed in the actual producer**:
> `produce_run3_fakesim_benchmark_shard.py`, line ~621, contains
> `if not mass_window or category == "none" or score is None: continue`.
> The current nominal `bbll` selection explicitly has
> `hbb_mass_window_required: false`. Historical Run-3 FakeSim *templates*
> therefore select the wrong event population for our analysis. The old
> progress log itself identifies this physics bug (PROGRESS.md §47).
> Correct the gate and rerun **the existing reconstruction/templating** as
> needed, not event generation or FakeSim framework development.
>
> Required adaptation:
>
> 1. Inventory legacy executable modules and historical Run-3 raw sources
>    `Wenu_b`, `Wmunu_b`, `ttbar_semileptonic`, plus Run-2 raw
>    FakeSim availability. Historical notes sometimes say Run-2 raw inputs
>    were unavailable; check the present AF filesystem before inferring.
> 2. Remove the obsolete Hbb-window veto in the FakeSim producer, keeping
>    the resolved/boosted category and all other frozen nominal cuts.
>    Add an explicit regression asserting no Hbb mass-window requirement.
> 3. Score FakeSim hypotheses with the **current** `bbll` frozen OOF
>    5-fold model bank and features, not old trained-model score maps.
>    Check model provenance, score features, and category closure.
> 4. Calibrate to the cited *public* Run-3 ATLAS 13.6-TeV e-mu inclusive
>    fake/nonprompt anchor, with correct luminosity and source coverage.
>    Historical FakeSim normalization kappas cannot be presumed unchanged
>    once object selection changes.
> 5. Rebuild matched Run-2/Run-3, mass-dependent fake templates and
>    fake variation envelopes at the exhaustive candidate thresholds.
>    Track source-level signed weights and event-level sumw2/covariance.
> 6. Validate full/partial closure and non-overlap with the 10 ordinary
>    prompt background samples before constructing **new** fake-aware fits.
>    Preserve all previous prompt-only outputs without modification.
>
> This is the basis of the upcoming implementation. The former SS
> transfer-factor prototype `fake_backgrounds/fakes.py` remains a
> non-nominal standalone diagnostic until explicitly justified; its
> synthetic tests do not validate real fake templates.
>

**Status: MC_TRANSFER_PROTOTYPE_TESTED; PHYSICS INPUTS AND CLOSURE NOT YET VALIDATED.**

This is a pure-MC Delphes phenomenology study (no observed collision data).
The existing 32/32-mass, two-or-more-mT-bin **prompt-only** SR results are
immutable. Do not call this a data-driven fake estimate or append these
prototype transfer factors to the nominal HistFactory workspace.

## Source and exact meaning

ATLAS internal note `ATL-COM-PHYS-2024-862` (draft 23 January 2026):

- Section 7.3, pp. 89–90, estimates fakes in OS regions from an SS
  control region, subtracting SS prompt MC from observed SS data and
  multiplying by a nonprompt-MC-derived OS/SS transfer factor.
- It requires the prompt/nonprompt overlap to be explicitly removed from
  the prompt samples to avoid double counting, retaining one fake estimate.
- Its stated FF pT bins have 12 **edges**:
  `[10,12,14,16,18,20,25,30,35,40,60,80]` GeV, i.e. **11 intervals**.
  Section 9.1.1 refers to 12 independent FF bin variations. This internal
  inconsistency must be audited against the actual FF histogram before
  transferring a nuisance or claiming the published FF bin count.
- The note's ~90% ttbar nonprompt contribution is specific to that
  ATLAS detector/selection and not automatically our Delphes SR.
- The ATLAS note's DD normalization and Run-2+Run-3 combination are
  **not** input fake efficiencies for our fast-simulation analysis.

## Current input blockers

1. `preselection/selection.py` takes `pass_common_preselection` which
   upstream already requires opposite-sign (OS) dileptons, so no SS event
   can be manufactured from the saved selected events.
2. `preselection/selection.py`'s retained `META` and BDT feature
   branches have neither lepton charges nor prompt-truth origin flags.
   `bdt_training/score.py` retains even fewer columns.
3. The existing `bbll/analysis.yaml` background inventory excludes
   W+jets/nonprompt semileptonic ttbar, which must be audited as candidate
   fake sources in separate source samples.
4. No data SS control region exists in this Delphes-only study. Any SS
   pseudo-data study must be labelled **MC closure**, with independent MC
   calibration and validation partitions (e.g. disjoint event UID sets).
5. Delphes object misidentification and isolation are card-specific.
   A fake lepton from heavy-flavor decay is not the same as a jet→lepton
   misidentification. Do not simulate missing jet-fakes by simply assigning
   arbitrary rates without a calibrated source and uncertainty envelope.

## Physics implementation sequence

**Gate F0 — raw-ntuple input audit**

- Inspect representative raw `Analysis_NOSYS` ntuple schemas, with
  `python -m fake_backgrounds.audit --ntuple /ABSOLUTE/PATH/sample.root`.
- Confirm if charges, lepton truth origin, b-decay origin, lepton pT,
  isolation, and source event identifiers exist. Audit both run periods.
- Explicitly map upstream branches to a canonical schema; don't infer
  prompt/nonprompt labels from process names, PDG species or reco alone.

**Gate F1 — reconstruct SS and OS baseline**

- Rerun **only** the nonprompt-relevant background ntuples through an
  alternative sign-independent preselection, keeping `event_uid`,
  lepton charges, truth-origin, lepton isolation, subleading pT, and all
  20/25 frozen BDT features. Use boosted-first exclusive categories.
- Apply the already frozen OOF 5-fold model bank to these events using
  `event_uid % 5` (no retraining). Keep per-process weights and signed
  `sumw2`. For nominal OS prompt backgrounds, truth-filter fake events
  if the fake estimate includes them.
- Validate charge closure, category partition, and signal-free CR overlaps.

**Gate F2 — Monte Carlo composition and MC statistical feasibility**

- For W+jets, semileptonic/dileptonic ttbar, tW, and relevant other
  processes: report reconstructed truth-nonprompt rates by lepton flavor,
  pT, era, and category. Count photon conversions separately where possible.
- Confirm whether the Delphes card models relevant misidentifications.
- Evaluate signed effective statistics for each source; do not infer
  a usable fake model solely from the total 1M generated events.

**Gate F3 — independent MC fake closure**

- On **calibration MC only**, derive `f_i = N_nonprompt,OS,i/N_nonprompt,SS,i`
  with pT bins and MC-weight uncertainties. The present prototype in
  `fake_backgrounds/fakes.py` refuses absent/nonpositive or low-Neff
  denominator/numerator bins.
- On **disjoint validation MC**, build SS pseudo-data = SS nonprompt + SS
  prompt. Subtract SS prompt, apply factors to the mT templates and compare
  to independently reconstructed OS truth nonprompt. Compute MC-stat and
  transfer-factor covariance, shape closure and process-composition checks.
- Do not clip negative SS-subtracted bins; validate physical modeling or
  explicitly merge/fit sparse bins with recorded uncertainty treatment.
- Test transfer-factor dependence on top process, lepton flavor, era,
  jet/tag topology, and high BDT score, not just overall yield closure.
- A pure-MC closure by itself does not calibrate fake rates to the detector.

**Gate F4 — physically justified final fake template**

- Select a documented fast-simulation strategy: direct truth-matched
  nonprompt OS MC **if** relevant fake mechanisms are simulated; or a
  transferable rate model externally calibrated to the approximated
  reconstruction. Any extrapolation requires a sensitivity envelope.
- Only after these checks: construct mass/channel-specific, positive,
  statistically adequate fake templates, and remove the same nonprompt
  contributions from prompt OS MC to avoid overlap.
- Integrate into a **new** pyhf likelihood with calibrated uncertainties;
  never overwrite `prompt_pyhf_exhaustive_campaign.json` or existing
  `prompt_pyhf_exhaustive_thresholds.csv`.

## Prototype status

`fake_backgrounds/fakes.py` currently implements only:

- signed-weight pT-bin transfer factors and delta-method MC variances;
- a simulation-only SS prompt subtraction, OS template projection and
  correlated transfer-factor covariance across mT bins;
- independent-validation expected-vs-truth closure diagnostics;
- explicit errors for invalid effective statistics, unphysical negative
  fake subtraction, incompatible shapes, and outside-pT bins.

All tests are **synthetic** and do not certify Delphes fake efficiencies
or physics closure. The fake-template integration switch is intentionally
set to `false` in `fake_backgrounds/config.yaml`.

## Run a safe initial schema audit on AF

```bash
cd ~/bbll
git pull --ff-only
source .venv39/bin/activate
python -m pytest -q tests/test_fake_backgrounds.py

# Choose one original raw Analysis_NOSYS ROOT file for each era.
python -m fake_backgrounds.audit --ntuple /ABSOLUTE/PATH/Run2_sample.root
python -m fake_backgrounds.audit --ntuple /ABSOLUTE/PATH/Run3_sample.root

# Also inspect a previously reduced selected ROOT file:
python -m fake_backgrounds.audit --ntuple /ABSOLUTE/PATH/selected_sample.root
```

If the raw ntuple does not contain charge and truth-origin branches, the
upstream Delphes-to-ntuples code (not the OOF BDT, SR scan or pyhf scripts)
must be extended. No additional MC generation should be authorized before
this input inventory establishes whether existing nonprompt-relevant
samples can support the required closure tests.
