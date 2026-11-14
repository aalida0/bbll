"""Strict exhaustive-shard validation and mass-point CLs merge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .exhaustive_inputs import (
    iter_combinations, load_inputs, merged_path, shard_path, sha256,
)
from .prompt_likelihood import ARCHITECTURES


def validate_shard(root, era, mass, index, shards, identity, mu_max=1e5):
    path = shard_path(root, era, mass, index, shards)
    if not path.exists():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text())
    for key, expected in identity.items():
        if row.get(key) != expected:
            raise ValueError(f"{path}: stale/wrong input identity field {key}")
    if (row.get("shard_index") != index or row.get("shard_count") != shards
            or row.get("mu_max") != float(mu_max)):
        raise ValueError(f"{path}: mismatched shard parameters")
    if (row.get("status") != "COMPLETE_EXHAUSTIVE_SHARD"
            or row.get("failures") != []
            or row.get("n_failed") != 0):
        raise ValueError(f"{path}: incomplete or failed shard; inspect before retrying")
    items = row.get("results")
    expected = len(range(index, identity["total_combinations"], shards))
    if (not isinstance(items, list) or row.get("n_assigned") != expected
            or row.get("n_success") != expected or len(items) != expected):
        raise ValueError(f"{path}: incomplete shard fit accounting")
    indices = []
    for result in items:
        ix = result.get("index")
        if type(ix) is not int or ix % shards != index:
            raise ValueError(f"{path}: result assigned to wrong shard")
        limit = float(result.get("expected_mu95", float("nan")))
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"{path}: invalid fit limit")
        indices.append(ix)
    if sorted(indices) != list(range(index, identity["total_combinations"], shards)):
        raise ValueError(f"{path}: duplicate or missing combination index")
    return row


def selection_reference(screen, arch, best):
    channels = ARCHITECTURES[arch]
    items = best["candidate_indices"]
    if set(items) != set(channels):
        raise ValueError("incomplete or overlapping best architecture")
    selected = {}
    for ch in channels:
        index = items[ch]
        src = screen["candidates"][ch][index]
        selected[ch] = {
            "index": int(index),
            "score_threshold": float(src["score_threshold"]),
            "efficiency_target": float(src["efficiency_target"]),
            "n_mt_bins": int(src["n_mt_bins"]),
            "mt_edges_GeV": src["edges_GeV"],
            "min_background_neff": float(src["min_background_neff"]),
            "signal_yield": float(src["signal_yield"]),
            "background_yield": float(src["background_yield"]),
        }
    return selected


def merge_mass(root, era, mass, shards, mu_max=1e5):
    screen, baseline, baseline_info, identity = load_inputs(root, era, mass)
    target = merged_path(root, era, mass)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    if shards < 1 or type(shards) is not int:
        raise ValueError("shards must be positive")

    documents = [
        validate_shard(root, era, mass, index, shards, identity, mu_max)
        for index in range(shards)
    ]
    found = {}
    for doc in documents:
        for row in doc["results"]:
            index = row["index"]
            if index in found:
                raise ValueError(f"duplicate combined fit index {index}")
            found[index] = row

    combos = list(iter_combinations(screen))
    if len(combos) != identity["total_combinations"]:
        raise RuntimeError("combination enumeration did not close")
    if set(found) != set(range(len(combos))):
        raise ValueError("missing or duplicate exhaustive combination")
    for task in combos:
        actual = found[task["index"]]
        if (actual["architecture"] != task["architecture"]
                or actual["candidate_indices"] != task["candidate_indices"]):
            raise ValueError(f"candidate identification mismatch at {task['index']}")

    architectures = {}
    for arch, channels in ARCHITECTURES.items():
        pool = [row for row in found.values() if row["architecture"] == arch]
        if len(pool) != identity["architecture_sizes"][arch]:
            raise RuntimeError(f"exhaustive count mismatch for {arch}")
        best = min(pool, key=lambda r: (r["expected_mu95"], r["index"]))
        short_best = baseline["architectures"][arch]["best"]
        short_limit = float(short_best["expected_mu95"])
        if best["expected_mu95"] > short_limit * (1.0 + 5e-5):
            raise RuntimeError(
                f"{arch}: exhaustive mu95={best['expected_mu95']} worse "
                f"than shortlisted {short_limit}; check numerical reproducibility"
            )
        architectures[arch] = {
            "n_combined_tested": len(pool),
            "shortlist_best_mu95": short_limit,
            "exhaustive_best_mu95": float(best["expected_mu95"]),
            "solver": best["solver"],
            "best_combination_index": best["index"],
            "best_candidates": selection_reference(screen, arch, best),
            "shortlist_to_exhaustive_ratio":
                short_limit / float(best["expected_mu95"]),
        }

    two = architectures["two_channel"]["exhaustive_best_mu95"]
    three = architectures["three_channel"]["exhaustive_best_mu95"]
    ratio = two / three
    result = {
        **identity,
        "status": "COMPLETE_PROMPT_ONLY_EXHAUSTIVE",
        "mu_max": float(mu_max), "n_shards": shards,
        "n_combined_tested": len(found), "n_combined_failed": 0,
        "shortlist_baseline_verified": True,
        "shortlist_baseline_file_sha256": sha256(baseline_info["file"]),
        "architectures": architectures,
        "two_over_three_expected_mu95_ratio": ratio,
        "percent_limit_reduction_three_vs_two": 100.0 * (1.0 - 1.0/ratio),
        "fake_backgrounds_included": False,
        "detector_theory_systematics_included": False,
        "signal_MCstat_included": False,
        "signal_normalization_in_fb_certified": False,
        "inference_scope": "prompt-only background shapesys MCstat; not final",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".json.tmp")
    if temp.exists():
        raise FileExistsError(temp)
    try:
        temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"PASS: {era}/m{mass}, {len(found)} exhaustive fits / {shards} shards; "
          f"2ch={two:.6g}, 3ch={three:.6g}, ratio={ratio:.5f}", flush=True)
    print(f"Saved {target}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--era", choices=("run2", "run3"), required=True)
    p.add_argument("--mass", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--mu-max", type=float, default=1e5)
    a = p.parse_args()
    merge_mass(a.output_root, a.era, a.mass, a.shards, a.mu_max)


if __name__ == "__main__":
    main()
