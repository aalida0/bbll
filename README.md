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

## 1. Event selection

All Run-2/Run-3 physics settings, sample names, frozen BDT inputs, and
selection branch names are in **[bbll/analysis.yaml](bbll/analysis.yaml)**.
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
python -m bbll.selection \
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
python -m bbll.campaign \
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
