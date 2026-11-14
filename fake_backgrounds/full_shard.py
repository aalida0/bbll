"""Full-source Run-3 FakeSim shard, using original raw Delphes and current bbll.

Processes ONE provenance-identified raw Delphes file once, constructs shared
jet->lepton hypotheses, scores current five-fold BDTs and fills all surviving
prompt-only SR candidate thresholds at every mass. No old Hbb mass-window
veto, no intermediate sample ntuples, no historical precomputed fake template.

Public e-mu anchor is computed BEFORE any SR threshold/mass-window logic.
Correlated hypotheses from one source event are summed BEFORE squaring.
Calibration is stored unscaled; the 29/fb ATLAS-inspired kappa is determined
only after strict full-source merge. Nothing is added to nominal pyhf yet.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .full_grid import pack_candidates, sha256, validate_grid
from .legacy_inventory import source_from_manifest
from .legacy_raw import (
    SOURCES, current_models, legacy_modules, load_normalization,
    score_hypothesis, selected_bin, validate_raw_schema,
)
from .full_source_audit import EXPECTED_EVENTS_PER_FILE, raw_paths


def accumulate_source_event(event_vectors, sumw, moment, active_events):
    """Exact event-level second moments, including across mT master bins."""
    for (benchmark, candidate), cells in event_vectors.items():
        if not cells:
            continue
        pairs = [(int(bin_id), float(w)) for bin_id, w in cells.items()
                 if float(w) != 0.0]
        if not pairs:
            continue
        active_events[benchmark, candidate] += 1
        for i, wi in pairs:
            sumw[benchmark, candidate, i] += wi
        for i, wi in pairs:
            for j, wj in pairs:
                moment[benchmark, candidate, i, j] += wi * wj


def load_benchmarks(legacy, n_alpha):
    bench_mod = legacy["bench"]
    names = list(bench_mod.benchmark_names())
    configs = [bench_mod.get_benchmark(name) for name in names]
    if len(names) != 4 or len(set(names)) != len(names):
        raise ValueError("expected 4 distinct legacy public fake benchmarks")
    nodes = [
        legacy["hyp"].truncated_gaussian_quadrature(
            b["alpha"]["mu"], b["alpha"]["sigma"], n_alpha
        )
        for b in configs
    ]
    alpha = np.asarray([x[0] for x in nodes[0]], dtype=np.float64)
    weights = np.asarray([[x[1] for x in row] for row in nodes])
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("invalid fake alpha quadrature weights")
    if not np.allclose(np.sum(weights, axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("alpha weights not normalized")
    for row in nodes[1:]:
        if not np.array_equal(alpha, np.asarray([x[0] for x in row])):
            raise ValueError("benchmark alpha nodes cannot be shared")
    charge_maps = []
    for b in configs:
        raw = {int(q): float(v) for q, v in b["charge_weights"].items()}
        total = sum(raw.values())
        if (any(q not in (-1, 1) or not math.isfinite(v) or v < 0
                for q, v in raw.items()) or total <= 0):
            raise ValueError("invalid fake charge model")
        charge_maps.append({-1: raw.get(-1, 0.0) / total,
                            1: raw.get(1, 0.0) / total})
    return names, configs, alpha, weights, charge_maps


def chosen_indices(category, score, banks):
    if category == "resolved":
        channels = ("resolved",)
    elif category in ("boosted_1tag", "boosted_2tag"):
        channels = ("boosted", category)
    else:
        return np.empty(0, dtype=np.int32)
    accepted = []
    for ch in channels:
        indices = banks[ch]["index"][score >= banks[ch]["score"]]
        if len(indices):
            accepted.append(indices)
    return (np.concatenate(accepted) if accepted else
            np.empty(0, dtype=np.int32))


def _candidate_master_edges(grid):
    edges = np.asarray(grid["master_mt_edges_GeV"], dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 3 or np.any(np.diff(edges) <= 0):
        raise ValueError("invalid master-bin edges")
    return edges


def run_source_file(*, manifest, source_root, legacy_repo, model_root,
                    grid_path, output_root, sample, ordinal, output_prefix,
                    n_alpha=8, chunk_size=500, max_events=None):
    manifest = Path(manifest)
    source_root = Path(source_root)
    output_root = Path(output_root)
    output_prefix = Path(output_prefix)
    out_json = Path(str(output_prefix) + ".json")
    out_npz = Path(str(output_prefix) + ".npz")
    tmp_json = Path(str(output_prefix) + ".json.tmp")
    tmp_npz = Path(str(output_prefix) + ".npz.tmp")
    for path in (out_json, out_npz, tmp_json, tmp_npz):
        if path.exists():
            raise FileExistsError(f"refuse to overwrite existing/incomplete shard: {path}")
    if sample not in SOURCES or ordinal <= 0:
        raise ValueError("invalid raw FakeSim source")
    if n_alpha < 1 or chunk_size < 1 or (max_events is not None and max_events < 1):
        raise ValueError("invalid FakeSim quadrature/chunk/entry settings")
    row = source_from_manifest(manifest, sample, ordinal)
    paths = raw_paths(row, source_root)
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {key} source: {path}")
    normalization_path = source_root / f"run3_{sample}_raw_weight_summary.json"
    skip_manifest = source_root / "run3_association_complete_skip_manifest.json"
    if not skip_manifest.is_file():
        raise FileNotFoundError(skip_manifest)
    old = legacy_modules(legacy_repo)
    expected_entries = validate_raw_schema(paths["raw"], old["raw"].LOGICAL)
    if expected_entries != EXPECTED_EVENTS_PER_FILE:
        raise ValueError(
            f"raw source must have {EXPECTED_EVENTS_PER_FILE} entries, got {expected_entries}"
        )
    norm = load_normalization(normalization_path, "run3", expected_entries)
    grid = validate_grid(grid_path, output_root, check_source=True)
    score_banks, master_edges = pack_candidates(grid)
    scores, model_shas = current_models(model_root, "run3")
    names, benchmarks, alpha, alpha_w, charge_maps = load_benchmarks(old, n_alpha)
    n_models, n_candidates = len(names), grid["n_candidates"]
    n_mt = len(master_edges)-1

    assoc = old["assoc"].load_association_cache(
        paths["association"], expected_raw_basename=paths["raw"].name
    )
    larger = old["larger"].load_cache(
        paths["largeR"], expected_raw_basename=paths["raw"].name
    )
    gates_module = importlib.import_module("scripts.data.fakesim_source_gates")
    gates = gates_module.load_association_skip_manifest(skip_manifest)
    expected_skips = gates_module.expected_skip_entries(
        gates, sample, str(row["source_id"])
    )
    # Reuse exactly the public e-mu anchor used in the old 4M production;
    # its selection does NOT depend on Hbb mass-window membership.
    old_producer = importlib.import_module(
        "scripts.data.produce_run3_fakesim_benchmark_shard"
    )
    anchor_pass = old_producer.anchor_proxy_pass

    y = np.zeros((n_models, n_candidates, n_mt), dtype=np.float64)
    M = np.zeros((n_models, n_candidates, n_mt, n_mt), dtype=np.float64)
    active = np.zeros((n_models, n_candidates), dtype=np.int64)
    anchor_y = np.zeros(n_models, dtype=np.float64)
    anchor_second = np.zeros(n_models, dtype=np.float64)
    anchor_active = np.zeros(n_models, dtype=np.int64)
    diagnostics = Counter()
    skips_seen = set()
    upper = expected_entries if max_events is None else min(max_events, expected_entries)
    print(f"Source {sample} ord{ordinal:03d}: {upper}/{expected_entries} raw events, "
          f"4 models, {n_candidates} SR thresholds, {n_mt} mT master bins",
          flush=True)
    for first in range(0, upper, chunk_size):
        stop = min(first+chunk_size, upper)
        events, _ = old["raw"].load_reco_events(
            paths["raw"], entry_start=first, entry_stop=stop,
            association_cache=assoc, largeR_overlap_cache=larger,
            btag_sample_id=row["sample_id"], btag_source_id=row["source_id"],
        )
        if len(events) != stop-first:
            raise RuntimeError("raw Delphes event count mismatch in chunk")
        for event in events:
            diagnostics["raw_events"] += 1
            if gates_module.should_skip_association_incomplete(
                gates, sample, str(row["source_id"]), event
            ):
                diagnostics["association_skipped"] += 1
                skips_seen.add(int(event["entry"]))
                continue
            nominal = old["hyp"].one_prompt_lepton_source(event)
            if nominal is None:
                continue
            diagnostics["one_prompt_events"] += 1
            uid = old["tags"].event_uid(
                row["sample_id"], row["source_id"], int(event["entry"])
            )
            source_weight = float(event["event_weight"]) * norm["scale"]
            if not math.isfinite(source_weight):
                raise ValueError("nonfinite raw FakeSim event weight")
            per_event = defaultdict(lambda: defaultdict(float))
            anchor_event = np.zeros(n_models, dtype=np.float64)
            for jet in old["hyp"].eligible_source_jets(event, nominal):
                diagnostics["eligible_jets"] += 1
                for flavour in ("electron", "muon"):
                    probabilities = np.asarray([
                        old["bench"].fake_probability(model, jet, flavour)
                        for model in benchmarks
                    ], dtype=np.float64)
                    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
                        raise ValueError("invalid fake efficiency")
                    if not np.any(probabilities):
                        continue
                    for ia, aval in enumerate(alpha):
                        for charge in (-1, 1):
                            factors = probabilities * alpha_w[:, ia] * np.asarray(
                                [m[charge] for m in charge_maps], dtype=np.float64
                            )
                            if not np.any(factors):
                                continue
                            hypo = old["hyp"].build_fake_hypothesis(
                                event, jet, flavour, float(aval), charge
                            )
                            diagnostics["hypotheses_built"] += 1

                            # Public normalization anchor is independent of
                            # SR score and Hbb mass. It is evaluated first.
                            if anchor_pass(
                                hypo, nominal, jet, flavour, float(aval), charge
                            ):
                                anchor_event += source_weight * factors * (
                                    29.0 / norm["target_lumi_fb"]
                                )
                                diagnostics["anchor_passing_hypotheses"] += 1

                            scored = score_hypothesis(hypo, uid, scores)
                            if scored is None:
                                continue
                            category, score = scored
                            diagnostics["scored_hypotheses"] += 1
                            if not hypo["hbb_topology"]["pass_hbb_mass_window"]:
                                diagnostics["scored_outside_old_mass_window"] += 1
                            indices = chosen_indices(category, score, score_banks)
                            if not len(indices):
                                continue
                            mt = float(hypo["observables"]["mt_hh_mH125"])
                            mt_idx = selected_bin(mt, master_edges)
                            diagnostics["scored_pass_at_least_one_cut"] += 1
                            for b, factor in enumerate(factors):
                                if factor == 0:
                                    continue
                                w = source_weight * float(factor)
                                for index in indices:
                                    per_event[(b, int(index))][mt_idx] += w
            # Do not count alpha quadrature, charge hypotheses or source
            # jets as statistically independent events. One source -> one
            # contribution vector (and exact outer product) per candidate.
            accumulate_source_event(per_event, y, M, active)
            anchor_y += anchor_event
            anchor_second += anchor_event*anchor_event
            anchor_active += anchor_event != 0
        print(f"progress {sample}/{ordinal}: {stop}/{upper}", flush=True)

    expected_seen = set(
        int(x) for x in expected_skips if int(x) < upper
    )
    if skips_seen != expected_seen:
        raise RuntimeError(
            f"association skip mismatch {sample}/{ordinal}: "
            f"got={sorted(skips_seen)}, expected={sorted(expected_seen)}"
        )
    if diagnostics["raw_events"] != upper:
        raise RuntimeError("raw Delphes entry count did not close")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(M))
            and np.all(np.isfinite(anchor_y))
            and np.all(np.isfinite(anchor_second))):
        raise ValueError("nonfinite fake shard arrays")
    if np.any(np.diagonal(M, axis1=2, axis2=3) < 0):
        raise ValueError("negative fake MC second-moment diagonal")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp_npz.open("wb") as stream:
            np.savez_compressed(
                stream, template_sumw=y, template_second_moment=M,
                template_nonzero_events=active, anchor_sumw_29fb=anchor_y,
                anchor_second_moment_29fb=anchor_second,
                anchor_nonzero_events=anchor_active,
                master_mt_edges=master_edges,
            )
        tmp_npz.replace(out_npz)
        result = {
            "schema_version": 1,
            "status": ("COMPLETE_RUN3_RAW_FAKESIM_SHARD" if upper == expected_entries
                       else "PARTIAL_RUN3_RAW_FAKESIM_DIAGNOSTIC"),
            "source_sample": sample, "ordinal": int(ordinal),
            "raw_file": str(paths["raw"]),
            "sample_id": str(row["sample_id"]),
            "source_id": str(row["source_id"]),
            "raw_file_events": expected_entries, "events_processed": upper,
            "full_file": upper == expected_entries,
            "source_normalization": norm,
            "n_alpha": int(n_alpha),
            "benchmarks": names,
            "benchmark_configurations": benchmarks,
            "n_candidates": n_candidates, "n_mt_master_bins": n_mt,
            "current_model_sha256": model_shas,
            "grid_sha256": sha256(grid_path),
            "legacy_source_sha256": old["provenance"],
            "raw_association_cache": str(paths["association"]),
            "raw_largeR_cache": str(paths["largeR"]),
            "skip_manifest_sha256": sha256(skip_manifest),
            "source_manifest_sha256": sha256(manifest),
            "normalization_summary_sha256": sha256(normalization_path),
            "npz_path": str(out_npz), "npz_sha256": sha256(out_npz),
            "expected_skips": sorted(expected_seen),
            "observed_skips": sorted(skips_seen),
            "diagnostics": dict(diagnostics),
            "hbb_mass_window_required": False,
            "anchor_applied_before_sr_selection": True,
            "historical_kappa_applied": False,
            "lumi_run3_fb": norm["target_lumi_fb"],
            "anchor_lumi_fb": 29.0,
            "fakeaware_likelihood_built": False,
            "physics_validated": False,
        }
        tmp_json.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        tmp_json.replace(out_json)
    finally:
        tmp_npz.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
    print(f"PASS: {result['status']} {sample} ord{ordinal:03d} "
          f"scored={diagnostics['scored_hypotheses']} "
          f"outside_old_Hbb_window={diagnostics['scored_outside_old_mass_window']}",
          flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--legacy-repo", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--sample", choices=SOURCES, required=True)
    p.add_argument("--ordinal", type=int, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--alpha-nodes", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--max-events", type=int,
                   help="Diagnostic only; full production processes entire source file")
    a = p.parse_args()
    run_source_file(
        manifest=a.manifest, source_root=a.source_root,
        legacy_repo=a.legacy_repo, model_root=a.model_root,
        grid_path=a.grid, output_root=a.output_root,
        sample=a.sample, ordinal=a.ordinal, output_prefix=a.output_prefix,
        n_alpha=a.alpha_nodes, chunk_size=a.chunk_size,
        max_events=a.max_events,
    )


if __name__ == "__main__":
    main()
