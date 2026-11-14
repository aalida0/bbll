# FakeSim from raw Delphes ROOT — migration pilot

Status: DIRECT_RAW_DELPHES_PILOT_IMPLEMENTED; synthetic tests pass. A real AF raw-file pilot is still required. No fake template has been certified.

## Source contract

There are no intermediate W+jets or semileptonic ttbar analysis ntuples, and none are needed. The new module fake_backgrounds/legacy_raw.py imports the tested raw Delphes reconstruction from a separate checkout of a alida0/bbll-ml-sr:

- raw_delphes_adapter.py decodes the Delphes TTree;
- association_cache.py and build_exact_largeR_overlap_cache.py provide exact track and smallR/LargeR identities;
- offline_btag_model.py uses provenance-dependent deterministic b-tags;
- fakesim_hypotheses.py, fakesim_reco.py, fakesim_hbb_topology.py and fakesim_analysis_observables.py rebuild lepton, jet, MET, and Hbb objects;
- fakesim_public_benchmarks.py supplies the four public-paper fake-hypothesis models.

The new adapter loads CURRENT bbll model banks (20 resolved / 25 boosted inputs, five event-UID-fold models) and the CURRENT exhaustive SR candidate winners and mT(HH) bin edges. It does NOT impose the obsolete Hbb mass-window veto from the old full producer. It stores separate two-channel and three-channel resolved thresholds, so the two architectures are never mixed.

## Prepare and inspect the first Run-3 raw source

Run on UChicago AF:

    cd ~/bbll
    git pull --ff-only
    source .venv39/bin/activate
    python -m pytest -q tests/test_legacy_raw_fakesim.py

    python -m fake_backgrounds.legacy_inventory \
      --manifest /data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json \
      --sample Wenu_b --ordinal 1 --era run3 --mass 800 --events 100 \
      --legacy-repo ~/bbll-ml-sr \
      --output-root /data/aalida/bbll_rerun_20261114

The inventory command does not run reconstruction. It reports the original raw Delphes ROOT path, cache paths, per-sample normalization summary, sample_id, and source_id. It prints a command for the executable pilot. Do not guess sample_id and source_id: these control event UID, deterministic tagging and fold selection.

After Wenu_b, repeat with --sample Wmunu_b and --sample ttbar_semileptonic.

The pilot reads a bounded prefix of the raw event file (default 100 events) and writes a NON-NOMINAL diagnostic JSON under:

    /data/aalida/bbll_rerun_20261114/fake_backgrounds/raw_pilots/run3/

## Normalization and event-level covariance

Use the old full-source normalization summary, with n_entries and norm_factor checked. The pilot weights each fake hypothesis by

    Event.Weight * norm_factor * (current_lumi/source_lumi)
      * epsilon(jet -> lepton) * alpha_weight * charge_weight.

The current Run-3 target luminosity is 308 fb^-1, replacing the old 312 fb^-1; Run-2 is 140.1 fb^-1.

One source event can generate many fake hypotheses. Sum all hypothesis weights within each raw source event BEFORE accumulating the squared event weight or master-bin covariance. A source event counts as one independent MC-statistical unit.

No historical FakeSim kappa is applied in this pilot, and the public ATLAS 13.6 TeV e-mu inclusive anchor is not used as a universal fake rate.

## Physics safeguards and current limitations

- Exact track association and LargeR overlap caches are REQUIRED; their source basename and event-number closure are checked by the old adapter. Raw Delphes files do not eliminate this reconstruction requirement.
- Incomplete semileptonic ttbar source events require the original exact skip manifest when producing full files. Skip without re-normalizing the source dataset.
- Historical Run-3 FakeSim predictions had a genuine extra Hbb mass-window veto. Historical templates and calibration should not be copied directly. The new pilot records how many scored hypotheses fall outside the Hbb mass window.
- The user reports raw Delphes outputs. Run-2 raw inputs and their proper provenance/cache/normalization are not yet verified; do not infer availability from the old Run-3 manifest.
- Synthetic tests prove numerical mechanics and window removal, not physics closure on actual Run-2/Run-3 Delphes events.
- A full campaign must evaluate all score candidate thresholds with master mT(HH) bins, not merely the prompt-only winners used for this pilot, and must validate source composition, prompt overlap, full correlated source-event moments, kappa re-calibration and model envelope.
- Current prompt-only mass scans, BDT models, and source event files remain unchanged. Fake templates are not integrated into the nominal pyhf likelihood yet.

Next gate: run the inventory for one file from each of the three Run-3 source classes, then run the printed bounded pilot commands. Once real raw-source closure passes, adapt the existing full FakeSim Condor producer, including removal of the old mass-window veto, score inputs, normalization and fake-aware likelihood.
