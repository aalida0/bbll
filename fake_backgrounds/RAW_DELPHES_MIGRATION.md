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
- Historical Run-3 FakeSim predictions had a genuine extra Hbb mass-window veto. Historical templates must not be copied directly. The historical public e-mu anchor is evaluated before the old mass-window gate, so its kappa may remain applicable if independently reproduced with the same sources and benchmark model. The new pilot records how many scored hypotheses fall outside the Hbb mass window.
- The user reports raw Delphes outputs. Run-2 raw inputs and their proper provenance/cache/normalization are not yet verified; do not infer availability from the old Run-3 manifest.
- Synthetic tests prove numerical mechanics and window removal, not physics closure on actual Run-2/Run-3 Delphes events.
- A full campaign must evaluate all score candidate thresholds with master mT(HH) bins, not merely the prompt-only winners used for this pilot, and must validate source composition, prompt overlap, full correlated source-event moments, verification of the public-anchor kappa against the historical source normalization, and model envelope.
- Current prompt-only mass scans, BDT models, and source event files remain unchanged. Fake templates are not integrated into the nominal pyhf likelihood yet.

Next gate: run the inventory for one file from each of the three Run-3 source classes, then run the printed bounded pilot commands. Once real raw-source closure passes, adapt the existing full FakeSim Condor producer, including removal of the old mass-window veto, score inputs, normalization and fake-aware likelihood.


## If the inventory reports missing exact-association caches

These are NOT analysis ntuples. They are compact, exact ROOT-identity
maps built by the old framework from the existing raw Delphes ROOT file.
Never use a cache belonging to a different raw filename or event order.

The old repository already provides these documented builders:

    python -m scripts.data.build_exact_association_cache RAW_ROOT OUTPUT_ASSOC_NPZ \
      --delphes-lib /home/aalida/tools/Delphes/libDelphes.so

    python -m scripts.data.build_exact_largeR_overlap_cache RAW_ROOT OUTPUT_LARGER_NPZ \
      --delphes-lib /home/aalida/tools/Delphes/libDelphes.so

Run from the old bbll-ml-sr checkout in a Python/ROOT environment with
PyROOT and libDelphes.so available. Build the FULL raw-file caches, not only
the first 100 pilot entries, if the cache will later be reused for complete
source processing. Their loaders validate raw basename, event counts and
per-entry identity. Do not blindly overwrite an already validated cache.

If a Run-2 raw FakeSim source or its original normalization summary is
missing, record that as a physics input blocker rather than substituting
Run-3 data. We must not equate a 13.6 TeV source ensemble with a 13 TeV one.

## After all three Run-3 files show FOUND

The inventory CLI now supports --execute. It checks the required source, cache
and normalization paths, passes provenance IDs without shell reinterpretation,
uses the active Python and refuses to overwrite existing pilot outputs.

Run the following three-source 100-event pilot with --execute:

    cd ~/bbll
    git pull --ff-only
    source .venv39/bin/activate

    for sample in Wenu_b Wmunu_b ttbar_semileptonic; do
      python -m fake_backgrounds.legacy_inventory \
        --manifest /data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json \
        --sample "$sample" --ordinal 1 --era run3 --mass 800 --events 100 \
        --legacy-repo ~/bbll-ml-sr \
        --output-root /data/aalida/bbll_rerun_20261114 \
        --execute
      if [ "$?" -ne 0 ]; then
        echo "FAKESIM PILOT FAILED for $sample — inspect traceback"
        break
      fi
    done

Success means the real raw Delphes objects were decoded and scored using
the current model and category. Zero high-BDT fake yield after only 100 source
events is not a physics failure and cannot be extrapolated to full production.
Full-source calibration, fake-aware optimization, and separate Run-2 input
verification remain separate validation gates.

## Real raw-file pilots (reported 2026-11-14)

All three ordinal-1 files processed a 100-event real Delphes pilot without errors:

| source | events | scored hypotheses | scored outside old Hbb window |
| --- | ---: | ---: | ---: |
| Wenu_b | 100 | 39 | 39 |
| Wmunu_b | 100 | 122 | 79 |
| ttbar_semileptonic | 100 | 420 | 210 |
| TOTAL | 300 | 581 | 328 |

These are hypothesis-level counts, not independent MC events or weighted yields. The 56.5% outside-window fraction cannot be used as a fractional fake-background correction. The selected 800 GeV pilot JSON files were written under the raw_pilots/run3 directory, but not directly inspected by this versioned source audit.

**Next gate: full 400-file input audit before HTCondor.**

    cd ~/bbll
    git pull --ff-only
    source .venv39/bin/activate
    python -m fake_backgrounds.full_source_audit \
      --manifest /data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json \
      --source-root /data/aalida/bbll_ml/fakes \
      --check-root-entries \
      --output /data/aalida/bbll_rerun_20261114/fake_backgrounds/run3_full_source_inventory.json

This checks 50 Wenu_b files + 50 Wmunu_b files + 300 semileptonic ttbar files, their exact track and LargeR caches, original normalization and two skipped ttbar events. Expect exactly 400 raw Delphes files and 4,000,000 generated source entries. If any are missing, it raises an error before submitting production.
