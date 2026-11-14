# Run-3 FakeSim production from 400 raw Delphes ROOT files

Status: source audit PASS (400 files / 4,000,000 entries, 2 skip entries),
three 100-event legacy reconstruction pilots PASS; full-source current-model
FakeSim producer implemented and synthetically tested, but NO full-source
Condor production has yet been run on AF. Do NOT label fake-aware limits final.

## Principles

- Use existing raw Delphes files and validated exact associations; no Wjets or
  semileptonic ttbar intermediate ntuples are required.
- Keep current OOF sklearn five-fold models, resolved 20 and boosted 25
  features, current 308 fb^-1 Run-3 normalization and source UID rule.
- The full-score scan covers all surviving prompt-only candidate thresholds
  at all 16 resonance masses, not only prompt-only winning cuts.
- NO Hbb mass-window veto in FakeSim SR selection. The public 29 fb^-1
  ATLAS-inspired e-mu normalization proxy is evaluated before SR cuts.
- One raw MC event is the unit for sumw2: sum all alpha, jet, charge, and
  fake-flavour hypotheses at event level before accumulating second moments.
- Four public-literature fake-model benchmarks are kept separate. The
  historical kappa values are never silently reused: strict merged
  calibration validates against the original 4M-source proxy calculation.
- Running/incomplete files are protected by atomic .running locks; failed
  jobs leave locks requiring manual inspection before resubmission.

## 1. Freeze full Run-3 score candidate grid

    cd ~/bbll
    git pull --ff-only
    source .venv39/bin/activate
    python -m pytest -q tests/test_fake_full_production.py
    python -m fake_backgrounds.full_grid \
      --output-root /data/aalida/bbll_rerun_20261114

Repeat checks on an existing grid with --resume, never overwrite manually:

    python -m fake_backgrounds.full_grid \
      --output-root /data/aalida/bbll_rerun_20261114 --resume

Expected output: a candidate-count summary and
/data/aalida/bbll_rerun_20261114/fake_backgrounds/run3_candidate_grid.json.
The candidate count must match the current frozen score-screen inventory.

## 2. Run a real all-mass 100-event producer smoke on each source class

    ROOT=/data/aalida/bbll_rerun_20261114
    MANIFEST=/data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json
    for sample in Wenu_b Wmunu_b ttbar_semileptonic; do
      python -m fake_backgrounds.full_shard \
        --manifest "$MANIFEST" \
        --source-root /data/aalida/bbll_ml/fakes \
        --legacy-repo /home/aalida/bbll-ml-sr \
        --model-root "$ROOT" \
        --grid "$ROOT/fake_backgrounds/run3_candidate_grid.json" \
        --output-root "$ROOT" \
        --sample "$sample" --ordinal 1 \
        --output-prefix "$ROOT/fake_backgrounds/full_shards_smoke/run3/$sample/ord001" \
        --max-events 100 --alpha-nodes 8 --chunk-size 100
      if [ "$?" -ne 0 ]; then
        echo "Raw full-producer smoke failed for $sample"
        break
      fi
    done

These files are PARTIAL diagnostics. They do NOT satisfy full-shard merge
criteria. Verify nonzero scored event counts, external-window records,
source signed weights, and anchor yields. The 100-event sample can still
have zero high-score candidate yields without invalidating reconstruction.

## 3. Submit one complete 10k-event ROOT file per source class

    python -m fake_backgrounds.full_submit --one-per-sample
    python -m fake_backgrounds.full_submit --one-per-sample --submit

This creates three Condor jobs, one Wenu_b, one Wmunu_b and one ttbar
source. The code requests 1 CPU, 6 GB RAM and 4 GB disk per job with the
shared /home and /data AF filesystem, 500-event decoding chunks and the
current 3-fold architecture candidate grids. Each job preserves one
source event as one MC-statistical unit.

Inspect:

    condor_q -nobatch
    find ~/bbll/condor/logs/run3_fakesim_current -type f -name '*.err' -size +0c
    find /data/aalida/bbll_rerun_20261114/fake_backgrounds/full_shards/run3 \
      -name 'ord001.json' -o -name 'ord001.npz'

Verify each full output JSON says COMPLETE_RUN3_RAW_FAKESIM_SHARD,
full_file=true, events_processed=10000, old Hbb window required=false,
public anchor evaluated before SR. If any job fails, DO NOT delete its
.running marker until the log and failure are understood.

## 4. Remaining 397 Run-3 raw source files

Only after the three complete-source validations pass:

    python -m fake_backgrounds.full_submit
    python -m fake_backgrounds.full_submit --submit

Alternatively cap a batch with --max-jobs 16 and submit more verified batches.
Do not submit a new batch while the previous one remains queued or running
without checking their state. This is a reproducibility/duplicate-prevention
requirement, not a recommendation to regenerate input events.

## 5. Strict merge and public-anchor check

After 400 complete shards:

    python -m fake_backgrounds.full_merge \
      --manifest /data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json \
      --source-root /data/aalida/bbll_ml/fakes \
      --grid /data/aalida/bbll_rerun_20261114/fake_backgrounds/run3_candidate_grid.json \
      --output-root /data/aalida/bbll_rerun_20261114

Output prefix:
/data/aalida/bbll_rerun_20261114/fake_backgrounds/merged/run3_full_raw_fakesim

The merge verifies 400 exact source IDs / 4 million event entries,
two exact skips, all input/calibration/model hashes, every expected
master-bin array dimension, and the 29 fb^-1 e-mu proxy calibration.

Expected historical comparison kappas for the same original public proxy:

    dror2016_global                 4.851040122
    dror2016_bonly                12.12790667
    curtin2015_mu_A__dror_alpha    0.2430399449
    curtin2015_mu_B__dror_alpha    0.2444497245

These are comparison targets; they are NOT automatically imposed.
The new merged kappa must match within 1% unless a documented difference
in the input, reconstruction or proxy explains and revalidates the change.

## 6. Still required for physics publication

- Compare the direct fake mT(HH) shapes and selected yields to the old
  historical templates to quantify the former Hbb veto impact.
- Study Run-3 fake source composition, missing processes, flavour,
  high-score and tagging-category dependence, and public-anchor proxy
  acceptance differences versus ATLAS detector selections.
- Derive per-candidate fake-model envelope and fake_norm nuisance without
  double-counting the dedicated source samples in ordinary prompt MC.
- Rebuild adaptive mT(HH) binning and MC stat at each prompt candidate with
  fake added; optimize both architectures with fake-aware likelihood.
- Audit Run-2 sources separately. Do not use Run-3 13.6-TeV fake samples as
  if they were Run-2 13-TeV MC.
- Quantify cross-category shared-source covariance and calibration-kappa
  statistical correlations if material. The source merger retains
  intra-candidate master-bin correlations, not all inter-candidate covariance.

Neither the current 61+ unit test suite nor the full-source inventory check
validates final Delphes fake probabilities or publication-grade limits.
