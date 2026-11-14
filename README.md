# bbll

Minimal, platform-independent reproduction code for expected limits on
`pp -> X -> HH -> bbll + MET`
at 13 TeV (Run 2) and 13.6 TeV (Run 3).

The pipeline is built incrementally: **1. Event selection**, 2. BDT,
3. signal-region optimization, 4. fake backgrounds, 5. systematic templates,
6. `pyhf` expected limits. Only validated code is added to `main`.

## Luminosity and normalized ROOT weights

The aligned `Weights/weight_nominal` sidecars **already include the sample cross
sections and the reference luminosity**; do not multiply by cross section again.
The single `bbll/analysis.yaml` specifies both the reference and target
luminosities. For Run 3, sidecars are normalized to **312 fb⁻¹** and the
publication target is **308 fb⁻¹**.

```yaml
run3:
  sqrt_s_TeV: 13.6
  luminosity_fb: 308.0
  weights_luminosity_fb: 312.0
  rescale_weights: true
```

With `rescale_weights: true`, later yield/template code uses
`analysis_weights(weight_nominal, "run3")` (scale = **308/312**).
With `false`, it uses the original weights (scale = 1).
Run 2 currently has `rescale_weights: false` because its weight reference
luminosity has not been independently confirmed. The selected ROOT
`physics_weight` and `train_weight_base` remain **raw** in either mode;
the luminosity correction is applied **once** when constructing expected
yields, systematics templates, and limits.

## Preselection and category assignment

All Run-2/Run-3 physics settings, sample names, frozen BDT inputs, and
selection branch names are in **[bbll/analysis.yaml](bbll/analysis.yaml)** (eras, normalization, samples, features) and **[preselection/config.yaml](preselection/config.yaml)** (cuts, ROOT schema).
The same configuration is used on AF, lxplus, or a laptop. Paths are CLI
arguments, not site-dependent settings.

Requires a nominal `Analysis_NOSYS` tree and an aligned normalized-weight
sidecar `Weights/weight_nominal` for each signal or prompt-background sample.
The ntuples must already contain `pass_common_preselection` and reconstructed
Higgs-candidate/tagging flags; this stage **does not reconstruct raw Delphes
objects** or redefine the upstream dilepton selection.

Selection (boosted topology has priority):

- **Boosted:** `pass_common_preselection`, `largeR_candidate_valid`,
  `largeR_trim_valid`, at least one `n_tags_inside_largeR`.
- **Resolved:** `pass_common_preselection`, **not boosted**,
  `resolved_candidate_valid`, `resolved_b1_tag`, `resolved_b2_tag`.

No `pass_hbb_mass_window` or `passes_analysis` requirement is applied.
The trimmed large-R jet is nominal. The fit observable `mt_hh_mH125` is stored
but **not used as a classifier input**.

The same selected events carry two *exclusive* category labels:

| `category3` | Three-category fit | `category2` | Two-category fit |
|---:|---|---:|---|
| 0 | resolved | 0 | resolved |
| 1 | boosted, exactly 1 associated b-tag | 1 | boosted, at least 1 tag |
| 2 | boosted, at least 2 associated b-tags | 1 | boosted, at least 1 tag |

For the two-category analysis, the 1-tag and ≥2-tag events are **merged at
template construction**, before fitting. Never combine *limits* from separate
categories numerically. Identical events, signal/background models, and
nuisance conventions are required for the final sensitivity comparison.

### Install (AF, lxplus, laptop)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

### Reduce one sample

```bash
python -m preselection.selection \
  /path/to/signal/run3/X600_run3.root \
  /path/to/output/run3/X600_run3.root
```

The sidecar is discovered as `X600_run3.weights.root` alongside the input.
You can also pass `--weights /path/to/weights.root`. Repeat for prompt
backgrounds and signals in both eras. The resulting ROOT `events` tree retains
`category3`, `category2`, `event_uid`, `source_entry`, signed `physics_weight`,
`train_weight_base=abs(physics_weight)`, signal labels, fitted `mt_hh_mH125`,
and the final 20 resolved/25 boosted BDT input variables. The original entry index allows later cross-checks against nominal and varied trees. A `.selection.json`
report provides category counts, signed yields, and exact count closure.

### Full campaign (16 signals + 10 prompt backgrounds per era)

```bash
python -m preselection.campaign \
  --input-root /path/to/bbll_samples \
  --output-root /path/to/selected \
  --era both \
  --resume
```

Use `--era run2` or `--era run3` for one period. `--resume` validates both existing ROOT and
per-sample JSON outputs before reusing them, so completed samples are not
regenerated after an interruption.

For a **one-time, exact comparison** with the older validated reduction, add:

```bash
  --reference-dir /path/to/previous/reduced
```

This verifies event order, IDs, signed weights, fit observable, and every
frozen BDT input for the resolved and boosted samples. The reference directory
is optional: a third-party reproduction does not depend on the older repo.

Reference selected-event counts (not physical yields) from the validated
reduction:

| Era | Resolved | Boosted |
|---|---:|---:|
| Run 2 | 8,891,378 | 6,294,727 |
| Run 3 | 8,899,170 | 6,225,647 |

The campaign checks the exact frozen per-era counts from YAML and writes `<output-root>/<era>/selection_campaign.json` with
per-sample and combined counts. The `source_entry` field identifies the
entry index in the **input Analysis_NOSYS tree** (not necessarily the original
generator or raw-Delphes entry).

Input and output paths are CLI arguments: no AF, lxplus, EOS, AFS, Condor,
or hostname assumptions are embedded in the code. Large ROOT inputs and model
artifacts are intentionally not stored in this repository.

**Scope of reproduction:** starting from the prepared ROOT ntuples and aligned
weight sidecars. Publishing the ntuple construction or the inputs themselves
is a separate requirement for reproduction starting from generator/Delphes
output.


## Publication stages

| Stage | Entry script | Configuration | State |
|---|---|---|---|
| Preselection | `preselection/selection.py` and `preselection/campaign.py` | `preselection/config.yaml` | **Validated for both eras** |
| BDT training | `bdt_training/train.py` | `bdt_training/config.yaml` | Implemented; awaiting AF validation |
| SR optimization | `sr_optimization/optimize.py` | `sr_optimization/config.yaml` | Not implemented yet |
| Fake backgrounds | `fake_backgrounds/fakes.py` | `fake_backgrounds/config.yaml` | Not implemented yet |
| Systematics | `systematics/templates.py` | `systematics/config.yaml` | Not implemented yet |
| Statistical inference | `statistical_inference/limits.py` | `statistical_inference/config.yaml` | Not implemented yet |

Preselection is fully validated for both eras. BDT training is implemented and awaits
an AF training campaign; the remaining stages are explicit nonfunctional placeholders. Shared eras, luminosity policy (optional 308/312
Run-3 scaling), samples, and feature names live in `bbll/analysis.yaml`.
All Stage-1 event weights remain stored exactly as in the input sidecars.

### Normalization provenance

The preselection reader uses **only canonical merged files** matching
`<stem>.root` and `<stem>.weights.root`, for instance
`ttbar_dil_run2.root` and `ttbar_dil_run2.weights.root`.
It deliberately does not open `_part1`/`_part2` files because those are
intermediate pieces and counting both merged and split files would duplicate
events. The nominal selected weights are copied from
`Weights/weight_nominal`; the reader does not infer or apply any separate
cross-section factor. The provenance and contents of any *cross-section
metadata* inside the sidecar still need explicit independent verification.

## BDT training

### Reproducible Python environment

**Model training uses Python 3.9**, NumPy 2.0.2 and scikit-learn 1.6.1.
The training entry point rejects other Python versions or leaked Key4hep
site-packages; tests may run on other Python versions.

Create a **new** isolated environment rather than reusing an activated
Key4hep/Python 3.13 environment:

```bash
unset PYTHONPATH PYTHONHOME
python3.9 -m venv .venv39
source .venv39/bin/activate
python -m pip install -e ".[test]"
python -m pip check
python -m bdt_training.environment
```

If `python3.9` is unavailable on PATH, use another verified Python 3.9
interpreter to create the venv. On Condor, we do not inherit the login-shell
environment; Python must be accessible from all workers via the shared
filesystem. The environment check is run before submission and on every
worker before model training.

The final all-mass model uses the published frozen 20-feature resolved
(\`conservative\`) and 25-feature boosted (\`safer\`) classifiers. Training uses
five deterministic folds (\`event_uid % 5\`), one model per held-out fold,
raw absolute event weights rebalanced per background family and signal mass
× decay cell, and the original 1/10 deterministic bulk sampling.

The four independent jobs are Run 2/Run 3 × resolved/boosted, configured in
\`bdt_training/config.yaml\`. Each reads the already-selected ROOT files from
\`preselection\` (no second reduction or regeneration).

Preflight one job without writing models:

\`\`\`bash
python -m bdt_training.train run3 resolved \
  --selected-root /path/to/selected \
  --output-root /path/to/results --dry-run
\`\`\`

Train interactively (repeat for all four combinations if not using Condor):

\`\`\`bash
python -m bdt_training.train run3 resolved \
  --selected-root /path/to/selected \
  --output-root /path/to/results
\`\`\`

For HTCondor on a shared filesystem, generate one four-job submit file without
hardcoding site paths, then submit:

\`\`\`bash
python -m bdt_training.submit \
  --selected-root /path/to/selected \
  --output-root /path/to/results
condor_submit /path/to/repo/condor/logs/bdt_training.sub
\`\`\`

Or add \`--submit\` to the first command. The generated submit file and
HTCondor event/STDOUT/STDERR logs default to \`<repo>/condor/logs/\`, **not**
\`<output-root>/logs/\`. On UChicago AF, Condor cannot log to Ceph (\`/data\`).
ROOT inputs and trained models may stay on \`/data\`. To choose an alternative
approved log directory, use \`--log-dir\`.

The submission uses `should_transfer_files = YES` to avoid HTCondor's
`FileSystemDomain` requirement when the login node and workers differ.
It transfers only `bdt_training/worker.sh` and runs it through `/bin/bash`.
The launcher calls the selected Python interpreter by its absolute path:
therefore the repository, virtual environment and selected ROOT files **must
still be accessible from worker nodes**, as in the historical UChicago AF
workflow. It requests **1 CPU, 16 GB memory and 4 GB disk** per job, with
BLAS/OpenMP threads limited to one.

If a previous submission remains idle because of the filesystem-domain
requirement, remove that old cluster before resubmitting to avoid duplicates.

Outputs: \`results/models/<era>/<topology>/<variant>/fold0.pkl\` through
\`fold4.pkl\`, \`features.txt\`, and \`training_metrics.json\`.

**Not a new physics definition:** the training pool, input features, and
hyperparameters reproduce the validated development methodology. This fresh
public training implementation still requires real-data AUC and event-count
comparison with the previous model banks before the new models are considered
publication-validated. Full-event OOF scoring is a separate follow-up within
\`bdt_training\`, after this model-bank validation.


### Full-event out-of-fold BDT scores

After confirming the 20 classifier models are present and match the historical
AUC benchmarks, score **every selected event** with its own held-out model:
\`fold = event_uid % 5\` and \`score = model[fold].predict_proba(features)[1]\`.
Unlike training, scoring does **not** apply 1/10 subsampling and does not
reweight events. The compact ROOT trees preserve \`physics_weight\` (signed,
uncorrected), \`mt_hh_mH125\`, both category labels, \`source_entry\`,
\`event_uid\`, the signal labels, \`ml_score\` and \`oof_fold\`.

Pilot on a single sample, then submit the full four-job campaign:

\`\`\`bash
python -m bdt_training.score run3 resolved \
  --selected-root /data/aalida/bbll_rerun_20261114/selected \
  --output-root /data/aalida/bbll_rerun_20261114 \
  --sample X600_run3

python -m bdt_training.submit \
  --mode score \
  --selected-root /data/aalida/bbll_rerun_20261114/selected \
  --output-root /data/aalida/bbll_rerun_20261114 \
  --submit
\`\`\`

Scores are stored separately by era/topology/sample:
\`<output-root>/bdt_scores/<era>/<topology>/<sample>.root\`. A per-sample
\`.oof.json\` report records the model hashes and raw weight sum, and
\`oof_campaign.json\` verifies exact preselection event-count closure. Worker
jobs use \`--resume\` to reuse validated partial campaigns, and write Condor
logs under \`<repo>/condor/logs/\` (not on Ceph).

The first scoring run must be validated against the historical scores,
per-category \(m_T(HH)\) shapes, and signed background effective statistics
before using BDT thresholds in a physics fit.

### OOF validation gate

Once the scored sample files exist, audit them against every selected source
event using `bdt_training.validate_scores`. This compares event IDs, source
entries, categories, fit variable, **raw signed physics weights**, and
`event_uid % 5` assignments. It also checks that the saved scores are finite,
within [0, 1], and exactly reproduce the held-out model for up to 100
events per sample. Model SHA-256 values are verified against the saved OOF
metadata.

Pilot after the single-sample score:

```bash
python -m bdt_training.validate_scores run3 resolved \
  --selected-root /data/aalida/bbll_rerun_20261114/selected \
  --output-root /data/aalida/bbll_rerun_20261114 \
  --sample X600_run3
```

Following the four full scoring jobs, repeat for every era/topology without
`--sample`; this creates `oof_validation.json` only after exact campaign
closure. OOF validation is a *necessary* correctness check, not by itself
evidence of safe high-score fit templates or absent mass sculpting.
