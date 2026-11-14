"""Merge/validate exhaustive mass-point CLs results and export chosen BDT cuts.

The final outputs remain explicitly prompt-only; this stage certifies
exhaustive threshold combinations, not a full systematic likelihood.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from .aggregate import ERAS, MASSES
from .exhaustive_inputs import load_inputs, merged_path
from .exhaustive_merge import merge_mass


def validate_mass(root, era, mass):
    screen, baseline, _, identity = load_inputs(root, era, mass)
    path = merged_path(root, era, mass)
    if not path.is_file():
        raise FileNotFoundError(path)
    doc = json.loads(path.read_text())
    for key, expected in identity.items():
        if doc.get(key) != expected:
            raise ValueError(f"{path}: stale exhaustive result field {key}")
    if (doc.get("status") != "COMPLETE_PROMPT_ONLY_EXHAUSTIVE"
            or doc.get("n_combined_tested") != identity["total_combinations"]
            or doc.get("n_combined_failed") != 0
            or doc.get("fake_backgrounds_included") is not False
            or doc.get("detector_theory_systematics_included") is not False
            or doc.get("signal_MCstat_included") is not False
            or doc.get("signal_normalization_in_fb_certified") is not False):
        raise ValueError(f"{path}: not a complete prompt-only exhaustive result")
    if not (isinstance(doc.get("n_shards"), int) and doc["n_shards"] >= 1):
        raise ValueError(f"{path}: missing shard count")

    architectures = doc.get("architectures", {})
    if set(architectures) != {"two_channel", "three_channel"}:
        raise ValueError(f"{path}: invalid architecture inventory")
    limits = {}
    from .prompt_likelihood import ARCHITECTURES
    for architecture, channels in ARCHITECTURES.items():
        item = architectures[architecture]
        if (item.get("n_combined_tested")
                != identity["architecture_sizes"][architecture]):
            raise ValueError(f"{path}: incomplete architecture fits")
        best = item["best_candidates"]
        if set(best) != set(channels):
            raise ValueError(f"{path}: missing/extra best threshold channels")
        for ch in channels:
            candidate = best[ch]
            index = candidate["index"]
            if type(index) is not int or index not in range(
                    len(screen["candidates"][ch])):
                raise ValueError(f"{path}: invalid best candidate index")
            source = screen["candidates"][ch][index]
            if (candidate["score_threshold"] != source["score_threshold"]
                    or candidate["efficiency_target"]
                    != source["efficiency_target"]
                    or candidate["mt_edges_GeV"] != source["edges_GeV"]):
                raise ValueError(f"{path}: stale best candidate {ch}")
        value = float(item["exhaustive_best_mu95"])
        pilot = float(baseline["architectures"][architecture]["best"]["expected_mu95"])
        if not math.isfinite(value) or value <= 0 or value > pilot * (1 + 5e-5):
            raise ValueError(f"{path}: invalid/non-exhaustive CLs best")
        limits[architecture] = value
    ratio = limits["two_channel"] / limits["three_channel"]
    if not math.isclose(ratio, doc["two_over_three_expected_mu95_ratio"], rel_tol=1e-10):
        raise ValueError(f"{path}: invalid ratio")
    return doc


def run_campaign(root, eras=ERAS, fits_per_shard=50, max_shards=32, mu_max=1e5):
    from .exhaustive_submit import calculate_shards

    root = Path(root)
    records = []
    for era in eras:
        for mass in MASSES:
            _, _, _, identity = load_inputs(root, era, mass)
            n = calculate_shards(identity["total_combinations"],
                                 fits_per_shard, max_shards)
            path = merged_path(root, era, mass)
            if path.exists():
                doc = validate_mass(root, era, mass)
            else:
                doc = merge_mass(root, era, mass, n, mu_max)
                validate_mass(root, era, mass)
            a = doc["architectures"]
            two = a["two_channel"]["exhaustive_best_mu95"]
            three = a["three_channel"]["exhaustive_best_mu95"]
            records.append({
                "era": era, "mass_GeV": mass,
                "two_mu95": two, "three_mu95": three,
                "ratio": two / three,
                "limit_reduction_percent": 100*(1-three/two),
                "exhaustive": True,
                "source_sha256": doc["screen_sha256"],
                "best_candidates": {
                    arch: a[arch]["best_candidates"] for arch in a
                },
            })
    outputs = root / "sr_optimization"
    summary = outputs / "prompt_pyhf_exhaustive_campaign.json"
    csv_path = outputs / "prompt_pyhf_exhaustive_thresholds.csv"
    if summary.exists() or csv_path.exists():
        raise FileExistsError(
            f"summary or CSV already exists: {summary} / {csv_path}"
        )
    payload = {
        "schema_version": 1, "status": "COMPLETE_PROMPT_ONLY_EXHAUSTIVE",
        "eras": list(eras), "masses_GeV": list(MASSES),
        "all_surviving_combinations_evaluated": True,
        "nominal_minimum_mt_bins": 2,
        "fakes_included": False, "systematics_included": False,
        "signal_normalization_in_fb_certified": False,
        "results": records,
    }
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "era", "mass_GeV", "architecture", "channel", "score_threshold",
        "efficiency_target", "n_mt_bins", "min_background_neff",
        "mu95_architecture", "two_over_three_ratio"
    ])
    for r in records:
        for arch in ("two_channel", "three_channel"):
            mu = r["two_mu95"] if arch == "two_channel" else r["three_mu95"]
            for ch, sel in r["best_candidates"][arch].items():
                writer.writerow([
                    r["era"], r["mass_GeV"], arch, ch,
                    f"{sel['score_threshold']:.12g}",
                    f"{sel['efficiency_target']:.3f}",
                    sel["n_mt_bins"],
                    f"{sel['min_background_neff']:.6g}",
                    f"{mu:.10g}", f"{r['ratio']:.8g}"
                ])
    outputs.mkdir(parents=True, exist_ok=True)
    ct = csv_path.with_suffix(".csv.tmp")
    jt = summary.with_suffix(".json.tmp")
    if ct.exists() or jt.exists():
        raise FileExistsError("unfinished campaign aggregation")
    try:
        ct.write_text(buf.getvalue())
        jt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        ct.replace(csv_path)
        jt.replace(summary)
    finally:
        ct.unlink(missing_ok=True)
        jt.unlink(missing_ok=True)
    print("era   mass   two_mu95    three_mu95    ratio    reduction(%)")
    for r in records:
        print(f"{r['era']:4s} {r['mass_GeV']:5d} "
              f"{r['two_mu95']:11.5g} {r['three_mu95']:12.5g} "
              f"{r['ratio']:8.4f} {r['limit_reduction_percent']:11.2f}")
    print(f"PASS: {len(records)} exhaustive comparisons; "
          f"threshold CSV: {csv_path}\nsummary JSON: {summary}", flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--era", choices=("run2", "run3", "both"), default="both")
    p.add_argument("--fits-per-shard", type=int, default=50)
    p.add_argument("--max-shards", type=int, default=32)
    p.add_argument("--mu-max", type=float, default=1e5)
    a = p.parse_args()
    eras = ERAS if a.era == "both" else (a.era,)
    run_campaign(a.output_root, eras,
                 a.fits_per_shard, a.max_shards, a.mu_max)


if __name__ == "__main__":
    main()
