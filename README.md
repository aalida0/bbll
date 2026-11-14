# bbll

Minimal, platform-independent reproduction code for expected limits on
(pp\to X\to HH\to b\bar b\ell\ell + E_\mathrm{T}^\mathrm{miss})
at 13 TeV (Run 2) and 13.6 TeV (Run 3).

The pipeline is built incrementally: **1. Event selection**, 2. BDT,
3. signal-region optimization, 4. fake backgrounds, 5. systematic templates,
6. `pyhf` expected limits. Only validated code is added to `main`.

## 1. Event selection

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
`category3`, `category2`, `event_uid`, signed `physics_weight`,
`train_weight_base=abs(physics_weight)`, signal labels, fitted `mt_hh_mH125`,
and the final 20 resolved/25 boosted BDT input variables. A `.selection.json`
report provides category counts, signed yields, and exact count closure.

Input and output paths are CLI arguments: no AF, lxplus, EOS, AFS, Condor,
or hostname assumptions are embedded in the code. Large ROOT inputs and model
artifacts are intentionally not stored in this repository.

**Scope of reproduction:** starting from the prepared ROOT ntuples and aligned
weight sidecars. Publishing the ntuple construction or the inputs themselves
is a separate requirement for reproduction starting from generator/Delphes
output.
