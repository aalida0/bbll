# BDT training validation — 14 November 2026

**Status: PASS (training reproducibility).** The four model banks were trained on
UChicago AF, and the user-provided `training_metrics.json` summaries reproduce
the previously frozen mean weighted AUC benchmarks to six decimal places.

Analysis: Run 2 (13 TeV) and Run 3 (13.6 TeV), resolved/boosted topologies.
Classifier: scikit-learn `HistGradientBoostingClassifier` 1.6.1; Python 3.9;
NumPy 2.0.2. Five held-out folds assigned by `event_uid % 5`.
Training: 1/10 deterministic signal/Z+jets/top sampling; all rare prompt
backgrounds; absolute weights with separate S/B, three background-family,
and 48 signal mass×decay cell balancing. No fakes in classifier training.

| Era | Topology | Variant | Events | Weighted AUC (mean) | AUC std | Models |
|---|---|---|---:|---:|---:|---|
| Run 2 | Resolved | conservative | 898,552 | 0.9924740823222533 | 0.00012250177773177073 | 5/5 |
| Run 2 | Boosted | safer | 636,047 | 0.9699637713399474 | 0.0005986055541673587 | 5/5 |
| Run 3 | Resolved | conservative | 902,360 | 0.9919109550905686 | 0.00044672318595304224 | 5/5 |
| Run 3 | Boosted | safer | 632,233 | 0.9684678841871335 | 0.0011666275035771387 | 5/5 |

Total training-pool rows across four jobs: **3,069,192**.
Total persisted model files: **20**.

## Fold-level evidence

| Era | Topology | Fold | Weighted AUC | Iterations |
|---|---|---:|---:|---:|
| Run 2 | Resolved | 0 | 0.992482 | 402 |
| Run 2 | Resolved | 1 | 0.992282 | 604 |
| Run 2 | Resolved | 2 | 0.992430 | 436 |
| Run 2 | Resolved | 3 | 0.992659 | 520 |
| Run 2 | Resolved | 4 | 0.992518 | 376 |
| Run 2 | Boosted | 0 | 0.969864 | 265 |
| Run 2 | Boosted | 1 | 0.969025 | 269 |
| Run 2 | Boosted | 2 | 0.970898 | 290 |
| Run 2 | Boosted | 3 | 0.969907 | 302 |
| Run 2 | Boosted | 4 | 0.970125 | 305 |
| Run 3 | Resolved | 0 | 0.991600 | 976 |
| Run 3 | Resolved | 1 | 0.992371 | 1113 |
| Run 3 | Resolved | 2 | 0.992021 | 930 |
| Run 3 | Resolved | 3 | 0.992348 | 786 |
| Run 3 | Resolved | 4 | 0.991214 | 962 |
| Run 3 | Boosted | 0 | 0.968144 | 681 |
| Run 3 | Boosted | 1 | 0.967475 | 421 |
| Run 3 | Boosted | 2 | 0.967762 | 554 |
| Run 3 | Boosted | 3 | 0.970738 | 727 |
| Run 3 | Boosted | 4 | 0.968220 | 222 |

Maximum observed iteration count: **1113/1200**; no fold hit the cap.
Training summaries were reported by the analyst; per-model SHA-256 and
independent per-event score equivalence to the historical campaign have not
yet been separately certified.

## Freeze and limitations

- Freeze the **training** algorithm, features (20 resolved; 25 boosted),
  hyperparameters, five-fold split, deterministic sampling and balancing.
- These AUC results **do not** establish stable fit sensitivity, absence of
  `mT(HH)` sculpting, good high-score background effective statistics or
  publication readiness.
- The next gate is full-event **out-of-fold** scoring (no subsampling).
  Validate exact selected-row closure and signed weights, compare score shapes
  with historical results, and stress-test high-score `N_eff` before any
  fit-bin optimization.
- The original sidecar-sourced `physics_weight` is retained during scoring.
  Optional Run-3 luminosity correction `308/312` belongs to downstream
  expected yields and statistical templates, never to the classifier loss.
