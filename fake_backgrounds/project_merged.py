"""Read-only projection of 4M merged Run-3 FakeSim onto current bbll SR bins.

Every fake candidate is aligned with an independently frozen prompt-only
candidate's exact score cut and mT(HH) edges. Event-level master-bin second
moments are projected as G C G^T; diagonal-only summation is incorrect for
within-source-event correlated hypotheses.

This is a DIAGNOSTIC, not a final fake-aware fit or detector validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .full_grid import sha256, validate_grid
from .full_merge import OLD_KAPPAS
from sr_optimization.scan import CHANNEL_TO_TOPOLOGY, validate_existing_screen

MERGED_STATUS = "RUN3_4M_RAW_FAKESIM_MERGED_ANCHOR_CLOSED_NOT_FINAL"


def load_merged(root):
    root = Path(root)
    grid_path = root / "fake_backgrounds" / "run3_candidate_grid.json"
    grid = validate_grid(grid_path, root, check_source=True)
    prefix = root / "fake_backgrounds" / "merged" / "run3_full_raw_fakesim"
    jpath = Path(str(prefix) + ".json")
    npz = Path(str(prefix) + ".npz")
    meta = json.loads(jpath.read_text())
    if (meta.get("status") != MERGED_STATUS
            or meta.get("schema_version") != 1
            or meta.get("n_source_files") != 400
            or meta.get("n_source_events") != 4000000
            or meta.get("n_association_skipped") != 2
            or meta.get("historical_kappa_pass") is not True
            or meta.get("n_candidates") != grid["n_candidates"]
            or meta.get("candidate_grid_sha256") != sha256(grid_path)
            or meta.get("npz_file") != str(npz)
            or meta.get("npz_sha256") != sha256(npz)
            or meta.get("benchmarks") != list(OLD_KAPPAS)
            or meta.get("hbb_mass_window_required") is not False):
        raise ValueError("stale/incomplete/invalid full-Run3 FakeSim merge")
    with np.load(npz, allow_pickle=False) as f:
        required = ("uncalibrated_sumw", "uncalibrated_second_moment",
                    "calibrated_sumw", "calibrated_second_moment",
                    "kappa", "master_mt_edges")
        if any(k not in f.files for k in required):
            raise ValueError("incomplete merged FakeSim numerical arrays")
        arrays = {k: np.asarray(f[k], dtype=float) for k in required}
    ny = (4, grid["n_candidates"], len(grid["master_mt_edges_GeV"]) - 1)
    if (arrays["calibrated_sumw"].shape != ny
            or arrays["uncalibrated_sumw"].shape != ny
            or arrays["calibrated_second_moment"].shape != ny + (ny[-1],)
            or arrays["uncalibrated_second_moment"].shape != ny + (ny[-1],)
            or arrays["kappa"].shape != (4,)
            or not np.array_equal(arrays["master_mt_edges"],
                                  grid["master_mt_edges_GeV"])):
        raise ValueError("merged FakeSim array/edge dimensions disagree with grid")
    if not all(np.all(np.isfinite(x)) for x in arrays.values()):
        raise ValueError("nonfinite merged FakeSim data")
    if np.any(arrays["kappa"] <= 0):
        raise ValueError("nonpositive fake normalization")
    for i, name in enumerate(OLD_KAPPAS):
        if not np.isclose(arrays["kappa"][i], meta["new_kappa"][name],
                          rtol=2e-12, atol=0):
            raise ValueError("npz fake calibration differs from JSON")
    if not np.allclose(
        arrays["calibrated_sumw"],
        arrays["uncalibrated_sumw"] * arrays["kappa"][:, None, None],
        rtol=1e-11, atol=1e-10
    ):
        raise ValueError("calibrated fake master yields do not close")
    return grid, meta, arrays


def grouping_matrix(master_edges, target_edges):
    master = np.asarray(master_edges, dtype=float)
    target = np.asarray(target_edges, dtype=float)
    if (master.ndim != 1 or target.ndim != 1 or len(target) < 3
            or np.any(np.diff(master) <= 0)
            or np.any(np.diff(target) <= 0)
            or target[0] != master[0]
            or target[-1] != master[-1]):
        raise ValueError("target mT edges must span full frozen master range")
    indices = np.searchsorted(master, target)
    if (np.any(indices >= len(master))
            or not np.array_equal(master[indices], target)):
        raise ValueError("target mT edge is not a frozen master edge")
    g = np.zeros((len(target)-1, len(master)-1), dtype=float)
    for i in range(len(g)):
        g[i, indices[i]:indices[i+1]] = 1.
    if not np.all(np.sum(g, axis=0) == 1):
        raise ValueError("target mT grouping loses or duplicates master bins")
    return g


def project(master_y, master_m, g):
    master_y = np.asarray(master_y, dtype=float)
    master_m = np.asarray(master_m, dtype=float)
    if (master_y.ndim != 1 or master_m.shape != (len(master_y),len(master_y))
            or g.ndim != 2 or g.shape[1] != len(master_y)
            or not np.all(np.isfinite(master_y))
            or not np.all(np.isfinite(master_m))):
        raise ValueError("fake master sum/covariance dimensions invalid")
    if not np.allclose(master_m, master_m.T, rtol=1e-10, atol=1e-8):
        raise ValueError("asymmetric FakeSim master covariance")
    result_y = g @ master_y
    result_cov = g @ master_m @ g.T
    if np.any(np.diag(result_cov) < -1e-10):
        raise ValueError("negative fake variance after bin merge")
    if not np.isclose(sum(result_y), sum(master_y), rtol=1e-9, atol=1e-8):
        raise ValueError("fake yield not conserved in bin projection")
    return result_y, result_cov


def project_candidate(grid, arrays, mass, channel, local_index, prompt_row):
    keys = [
        row for row in grid["fakesim_candidate_rows"]
        if (int(row["mass_GeV"]) == int(mass)
            and row["channel"] == channel
            and int(row["local_index"]) == int(local_index))
    ]
    if len(keys) != 1:
        raise ValueError(f"missing/duplicated frozen fake cut: {mass}/{channel}/{local_index}")
    src = keys[0]
    if (not np.isclose(float(src["score_threshold"]),
                       float(prompt_row["score_threshold"]), rtol=0, atol=1e-12)
            or not np.isclose(float(src["efficiency_target"]),
                              float(prompt_row["efficiency_target"]),
                              rtol=0, atol=1e-12)):
        raise ValueError("fake and prompt candidate cuts are not identical")
    g = grouping_matrix(grid["master_mt_edges_GeV"], prompt_row["edges_GeV"])
    ix = int(src["index"])
    result = {}
    for i, name in enumerate(OLD_KAPPAS):
        y, cov = project(
            arrays["calibrated_sumw"][i, ix],
            arrays["calibrated_second_moment"][i, ix], g
        )
        v = np.diag(cov)
        neff = np.divide(y*y, v, out=np.zeros_like(y), where=v > 0)
        result[name] = {
            "sumw": y.tolist(), "sumw2": v.tolist(),
            "event_level_covariance": cov.tolist(),
            "neff": neff.tolist(),
            "total_yield": float(np.sum(y)),
            "negative_bins": int(np.sum(y < 0)),
            "nonzero_bins": int(np.sum(y != 0)),
        }
    return {"channel": channel, "mass_GeV": int(mass),
            "candidate_index": int(local_index),
            "grid_index": ix,
            "score_threshold": float(src["score_threshold"]),
            "efficiency_target": float(src["efficiency_target"]),
            "mt_edges_GeV": list(map(float,prompt_row["edges_GeV"])),
            "prompt_background_bins": prompt_row["background_bins"],
            "prompt_background_sumw2_bins": prompt_row["background_sumw2_bins"],
            "signal_bins": prompt_row["signal_bins"],
            "benchmarks": result}


def mass_report(root, mass):
    root = Path(root)
    grid, merged, arrays = load_merged(root)
    screen = validate_existing_screen(
        root/"sr_optimization"/"run3"/f"m{int(mass)}"/"prompt_candidate_screen.json",
        "run3", int(mass)
    )
    rows = []
    for ch in CHANNEL_TO_TOPOLOGY:
        for i, candidate in enumerate(screen["candidates"][ch]):
            rows.append(project_candidate(grid, arrays, mass, ch, i, candidate))
    if len(rows) != sum(
            1 for row in grid["fakesim_candidate_rows"]
            if int(row["mass_GeV"]) == int(mass)):
        raise ValueError("fake candidate inventory has incomplete mass coverage")
    return {
        "schema_version": 1,
        "status": "RUN3_FAKESIM_MASS_PROJECTED_DIAGNOSTIC_NOT_FINAL",
        "era": "run3", "mass_GeV": int(mass),
        "n_candidates": len(rows), "merged_source_sha256":
        merged["npz_sha256"], "candidate_grid_sha256":
        merged["candidate_grid_sha256"],
        "lumi_fb": merged["current_luminosity_fb"],
        "nominal_model_for_diagnostics": "dror2016_global",
        "physics_validated": False,
        "fake_aware_limits_computed": False,
        "fake_model_nuisance_finalized": False,
        "warning": ("All fake rates are public-benchmark proxies calibrated "
                    "to an inclusive e-mu anchor; SR detector acceptance "
                    "and transfer remain model-dependent."),
        "candidates": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--mass", type=int, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    dst = (a.output or (a.output_root / "fake_backgrounds" / "diagnostics"
                        / "run3" / f"m{a.mass}_fake_candidates.json"))
    if dst.exists():
        raise FileExistsError(f"refuse to overwrite existing fake projection: {dst}")
    doc = mass_report(a.output_root, a.mass)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc, indent=2, sort_keys=True)+"\n")
    channels = list(CHANNEL_TO_TOPOLOGY)
    print(f"PASS Run3 m{a.mass}: projected {len(doc['candidates'])} "
          f"candidates to exact prompt SR mT binning")
    for channel in channels:
        selected = [r for r in doc["candidates"] if r["channel"] == channel]
        nneg = sum(r["benchmarks"]["dror2016_global"]["negative_bins"]
                   for r in selected)
        print(f"  {channel}: candidates={len(selected)} "
              f"nominal_fake_negative_bins={nneg}")
    print(f"Saved: {dst}")


if __name__ == "__main__":
    main()
