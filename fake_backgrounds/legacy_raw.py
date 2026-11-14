"""Read-only raw Delphes FakeSim pilot using the tested legacy reconstruction.

No intermediate background ntuples, no obsolete Hbb mass-window veto,
no historical calibration kappa, no modification of nominal prompt results.
This is a bounded pilot; the full FakeSim campaign requires separate review.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from bbll.config import load_config
from bdt_training.score import model_bank
from sr_optimization.exhaustive_campaign import validate_mass

SOURCES = ("Wenu_b", "Wmunu_b", "ttbar_semileptonic")
CHANNELS = ("resolved", "boosted", "boosted_1tag", "boosted_2tag")
CATEGORIES = ("resolved", "boosted_1tag", "boosted_2tag")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_modules(repo):
    repo = Path(repo).expanduser().resolve()
    files = (
        "raw_delphes_adapter.py", "fakesim_hypotheses.py",
        "fakesim_reco.py", "fakesim_hbb_topology.py",
        "fakesim_analysis_observables.py", "fakesim_public_benchmarks.py",
        "offline_btag_model.py", "association_cache.py",
        "build_exact_largeR_overlap_cache.py", "delphes_refs.py",
    )
    missing = [str(repo / "scripts/data" / name) for name in files
               if not (repo / "scripts/data" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"legacy FakeSim checkout incomplete: {missing}")
    if "scripts" in sys.modules:
        raise RuntimeError("conflicting scripts namespace imported before FakeSim")
    sys.path.insert(0, str(repo))
    def imp(name):
        return importlib.import_module("scripts.data." + name)
    return {
        "raw": imp("raw_delphes_adapter"),
        "hyp": imp("fakesim_hypotheses"),
        "bench": imp("fakesim_public_benchmarks"),
        "tags": imp("offline_btag_model"),
        "assoc": imp("association_cache"),
        "larger": imp("build_exact_largeR_overlap_cache"),
        "provenance": {name: sha256(repo / "scripts/data" / name)
                       for name in files},
        "root": str(repo),
    }


def validate_raw_schema(path, logical):
    import uproot
    with uproot.open(path) as root:
        if "Delphes" not in root:
            raise ValueError(f"{path}: Delphes tree not found")
        tree = root["Delphes"]
        existing = set(str(key).split("/")[-1] for key in tree.keys())
        missing = sorted(set(logical.values()) - existing)
        if missing:
            raise ValueError(f"{path}: missing FakeSim Delphes branches {missing}")
        if tree.num_entries < 1:
            raise ValueError("Delphes tree has no events")
        return int(tree.num_entries)


def load_normalization(path, era, raw_entries):
    cfg = load_config()
    row = json.loads(Path(path).read_text())
    for key in ("norm_factor", "lumi_fb", "n_entries"):
        if key not in row:
            raise ValueError(f"{path}: missing {key}")
    factor, lumi = float(row["norm_factor"]), float(row["lumi_fb"])
    n_total = int(row["n_entries"])
    if (not math.isfinite(factor) or factor <= 0
            or not math.isfinite(lumi) or lumi <= 0 or n_total < raw_entries):
        raise ValueError("invalid source normalization")
    target = float(cfg["eras"][era]["luminosity_fb"])
    return {
        "source_lumi_fb": lumi, "target_lumi_fb": target,
        "norm_factor": factor, "scale": factor * target / lumi,
        "n_total": n_total, "sha256": sha256(path), "source": str(path),
    }


def frozen_winners(root, era, mass):
    doc = validate_mass(root, era, mass)
    winners = {}
    for channel in CHANNELS:
        arch = "two_channel" if channel == "boosted" else "three_channel"
        row = doc["architectures"][arch]["best_candidates"][channel]
        edges = np.asarray(row["mt_edges_GeV"], dtype=np.float64)
        if len(edges) < 3 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"nominal {channel} mT bins invalid")
        winners[channel] = {
            "threshold": float(row["score_threshold"]), "edges": edges,
            "efficiency_target": float(row["efficiency_target"]),
        }
    return winners, doc["screen_sha256"]


def current_models(root, era):
    out, hashes = {}, {}
    for topology, nfeatures in (("resolved", 20), ("boosted", 25)):
        features, models, shas = model_bank(Path(root), era, topology)
        if len(features) != nfeatures or len(models) != 5:
            raise ValueError("unexpected current frozen BDT configuration")
        out[topology] = (features, models)
        hashes[topology] = shas
    return out, hashes


def score_hypothesis(hypothesis, uid, banks):
    category = str(hypothesis["hbb_topology"]["analysis_subcategory_name"])
    if category not in CATEGORIES:
        return None
    topo = "resolved" if category == "resolved" else "boosted"
    features, models = banks[topo]
    obs = hypothesis["observables"]
    absent = [name for name in features if name not in obs]
    if absent:
        raise ValueError(f"FakeSim {topo} missing BDT features {absent}")
    vector = np.asarray([[float(obs[name]) for name in features]], dtype=np.float32)
    if np.any(np.isinf(vector)):
        raise ValueError("infinite FakeSim feature; NaN is permitted by HGB")
    score = float(models[int(uid) % 5].predict_proba(vector)[0, 1])
    if not (math.isfinite(score) and 0.0 <= score <= 1.0):
        raise ValueError("invalid FakeSim score")
    return category, score


def selected_bin(mt, edges):
    if not math.isfinite(float(mt)):
        raise ValueError("nonfinite mT(HH)")
    value = min(max(float(mt), float(edges[0])+1e-6), float(edges[-1])-1e-6)
    return int(np.searchsorted(edges, value, side="right") - 1)


def run_pilot(*, raw_root, legacy_repo, model_root, output_root,
              norm_summary, association_cache, larger_cache, era, mass,
              sample, sample_id, source_id, output,
              events=100, chunk_size=100, alpha_nodes=4, skip_manifest=None):
    if era not in ("run2", "run3") or sample not in SOURCES:
        raise ValueError("unsupported Run era or FakeSim source")
    if any(type(x) is not int or x < 1 for x in (events, chunk_size, alpha_nodes)):
        raise ValueError("events, chunk_size and alpha_nodes must be positive integers")
    if not str(sample_id) or not str(source_id):
        raise ValueError("sample_id and source_id must come from original provenance")
    output = Path(output)
    temp = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temp.exists():
        raise FileExistsError(f"existing or incomplete output: {output}")
    for name, path in (("Delphes ROOT", raw_root),
                       ("exact association cache", association_cache),
                       ("exact LargeR cache", larger_cache),
                       ("normalization", norm_summary)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    old = legacy_modules(legacy_repo)
    n_raw = validate_raw_schema(raw_root, old["raw"].LOGICAL)
    norm = load_normalization(norm_summary, era, n_raw)
    winners, screen_hash = frozen_winners(output_root, era, mass)
    banks, model_hashes = current_models(model_root, era)
    assoc = old["assoc"].load_association_cache(
        association_cache, expected_raw_basename=Path(raw_root).name
    )
    larger = old["larger"].load_cache(
        larger_cache, expected_raw_basename=Path(raw_root).name
    )

    gates = None
    if skip_manifest is not None:
        gate_module = importlib.import_module("scripts.data.fakesim_source_gates")
        gates = (gate_module, gate_module.load_association_skip_manifest(skip_manifest))
    if sample == "ttbar_semileptonic" and gates is None and events >= n_raw:
        raise ValueError("full ttbar sample requires existing association skip manifest")

    benchmark = old["bench"]
    names = list(benchmark.benchmark_names())
    models = [benchmark.get_benchmark(name) for name in names]
    if not names:
        raise ValueError("no legacy FakeSim benchmark models")
    grids = [
        old["hyp"].truncated_gaussian_quadrature(
            model["alpha"]["mu"], model["alpha"]["sigma"], alpha_nodes
        ) for model in models
    ]
    alpha_values = [a for a, _ in grids[0]]
    if any(not np.allclose(alpha_values, [a for a, _ in row],
                           rtol=0, atol=1e-13) for row in grids[1:]):
        raise ValueError("legacy alpha quadrature grids disagree")

    hist = {(name, ch): np.zeros(len(x["edges"])-1, dtype=float)
            for name in names for ch, x in winners.items()}
    moments = {key: np.zeros((len(x), len(x)), dtype=float)
               for key, x in hist.items()}
    diagnostics = Counter()
    upper = min(events, n_raw)
    for first in range(0, upper, chunk_size):
        stop = min(first + chunk_size, upper)
        chunk, _ = old["raw"].load_reco_events(
            raw_root, entry_start=first, entry_stop=stop,
            association_cache=assoc, largeR_overlap_cache=larger,
            btag_sample_id=sample_id, btag_source_id=source_id
        )
        if len(chunk) != stop-first:
            raise RuntimeError("raw Delphes entry count mismatch")
        for event in chunk:
            diagnostics["raw_events"] += 1
            if gates is not None and gates[0].should_skip_association_incomplete(
                gates[1], sample, str(source_id), event
            ):
                diagnostics["association_skipped"] += 1
                continue
            nominal = old["hyp"].one_prompt_lepton_source(event)
            if nominal is None:
                continue
            diagnostics["one_prompt_events"] += 1
            uid = old["tags"].event_uid(sample_id, source_id, int(event["entry"]))
            event_hist = {key: np.zeros_like(value) for key, value in hist.items()}
            for jet in old["hyp"].eligible_source_jets(event, nominal):
                diagnostics["eligible_jets"] += 1
                for flavour in ("electron", "muon"):
                    probs = [benchmark.fake_probability(m, jet, flavour)
                             for m in models]
                    if not any(p > 0 for p in probs):
                        continue
                    for inode, alpha in enumerate(alpha_values):
                        for charge in (-1, 1):
                            hypo = old["hyp"].build_fake_hypothesis(
                                event, jet, flavour, alpha, charge
                            )
                            diagnostics["hypotheses_built"] += 1
                            scored = score_hypothesis(hypo, uid, banks)
                            if scored is None:
                                continue
                            cat, score = scored
                            diagnostics["hypotheses_scored"] += 1
                            if not hypo["hbb_topology"]["pass_hbb_mass_window"]:
                                diagnostics["outside_hbb_window_scored"] += 1
                            chs = ("resolved",) if cat == "resolved" else ("boosted", cat)
                            mt = float(hypo["observables"]["mt_hh_mH125"])
                            for ch in chs:
                                info = winners[ch]
                                if score < info["threshold"]:
                                    continue
                                k = selected_bin(mt, info["edges"])
                                for name, model, prob, grid in zip(
                                    names, models, probs, grids
                                ):
                                    if prob <= 0:
                                        continue
                                    charge_map = model["charge_weights"]
                                    qtotal = sum(float(x) for x in charge_map.values())
                                    if qtotal <= 0:
                                        raise ValueError("invalid charge weights")
                                    factor = (
                                        float(event["event_weight"]) * norm["scale"]
                                        * prob * grid[inode][1]
                                        * float(charge_map.get(charge, 0.0))/qtotal
                                    )
                                    event_hist[(name, ch)][k] += factor
            # The raw source event, not individual hypotheses, is the MC unit.
            for key, values in event_hist.items():
                hist[key] += values
                moments[key] += np.outer(values, values)

    rows = {}
    for name in names:
        rows[name] = {}
        for ch in CHANNELS:
            h = hist[(name, ch)]
            cov = moments[(name, ch)]
            var = np.diag(cov)
            neff = np.divide(h*h, var, out=np.zeros_like(h), where=var > 0)
            rows[name][ch] = {
                "score_threshold": winners[ch]["threshold"],
                "signal_efficiency_target": winners[ch]["efficiency_target"],
                "edges_GeV": winners[ch]["edges"].tolist(),
                "sumw": h.tolist(), "second_moment": cov.tolist(),
                "neff": neff.tolist(),
            }
    result = {
        "schema_version": 1, "status": "RAW_DELPHES_FAKESIM_PILOT_NOT_NOMINAL",
        "era": era, "mass_GeV": mass, "sample": sample,
        "sample_id": str(sample_id), "source_id": str(source_id),
        "raw_root": str(raw_root), "source_file_events": n_raw,
        "events_processed": upper, "partial_source": upper < n_raw,
        "normalization": norm, "diagnostics": dict(diagnostics),
        "legacy_repo": old["root"], "legacy_file_sha256": old["provenance"],
        "current_model_hashes": model_hashes, "screen_sha256": screen_hash,
        "association_cache": str(association_cache),
        "largeR_overlap_cache": str(larger_cache),
        "alpha_nodes": alpha_nodes, "benchmarks": names,
        "hbb_mass_window_required": False,
        "historical_fake_kappa_applied": False,
        "physics_validated": False, "nominal_modified": False,
        "templates": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"PASS raw Delphes FakeSim pilot: {sample} {era} m{mass}, "
          f"{upper}/{n_raw} events; scored={diagnostics['hypotheses_scored']}, "
          f"scored outside Hbb window={diagnostics['outside_hbb_window_scored']}")
    print(f"Saved {output}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("raw-root", "legacy-repo", "model-root", "output-root",
                "norm-summary", "association-cache", "larger-cache",
                "sample-id", "source-id", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--era", choices=("run2", "run3"), required=True)
    p.add_argument("--sample", choices=SOURCES, required=True)
    p.add_argument("--mass", type=int, required=True)
    p.add_argument("--events", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--alpha-nodes", type=int, default=4)
    p.add_argument("--skip-manifest")
    a = p.parse_args()
    opts = vars(a)
    for k in ("raw_root", "legacy_repo", "model_root", "output_root",
              "norm_summary", "association_cache", "larger_cache",
              "output"):
        opts[k] = Path(opts[k])
    if opts["skip_manifest"]:
        opts["skip_manifest"] = Path(opts["skip_manifest"])
    run_pilot(**opts)


if __name__ == "__main__":
    main()
