"""Strict full-400-file Run-3 raw FakeSim merger and public-anchor diagnostic.

Every file is checked against its original source identity, full event count,
frozen score grid, current BDT model hashes, old reconstruction source hashes,
exact skip manifest and saved numerical second moments.

The public inclusive 29/fb e-mu anchor is used only after summing all 400
source files. Previous 4M FakeSim kappa factors are comparison targets, not
silent or unverified inputs. Outputs remain physics-conditional (no pyhf yet).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .full_grid import sha256, validate_grid
from .full_source_audit import (
    EXPECTED_RUN3_FILES, EXPECTED_EVENTS_PER_FILE,
    raw_paths, verify_skip_manifest
)

OLD_KAPPAS = {
    "dror2016_global": 4.851040122,
    "dror2016_bonly": 12.12790667,
    "curtin2015_mu_A__dror_alpha": 0.2430399449,
    "curtin2015_mu_B__dror_alpha": 0.2444497245,
}
ATLAS_PUBLIC_EMU_TARGET = 2800.0
ATLAS_PUBLIC_EMU_TARGET_ERR = 1400.0


def shard_prefix(out_root, sample, ordinal):
    return Path(out_root) / "fake_backgrounds" / "full_shards" / "run3" / sample / (
        f"ord{int(ordinal):03d}"
    )


def expected_identity(src_row, source_root):
    return raw_paths(src_row, source_root)


def validate_full_shard(prefix, row, source_root, grid_sha, n_candidates, n_mt,
                        expected_models=None):
    prefix = Path(prefix)
    meta_path = Path(str(prefix)+".json")
    npz_path = Path(str(prefix)+".npz")
    if not meta_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"incomplete FakeSim output: {prefix}")
    meta = json.loads(meta_path.read_text())
    file_paths = expected_identity(row, source_root)
    expected = {
        "schema_version": 1,
        "status": "COMPLETE_RUN3_RAW_FAKESIM_SHARD",
        "source_sample": row["sample"],
        "ordinal": int(row["ordinal"]),
        "raw_file": str(file_paths["raw"]),
        "sample_id": str(row["sample_id"]),
        "source_id": str(row["source_id"]),
        "raw_file_events": EXPECTED_EVENTS_PER_FILE,
        "events_processed": EXPECTED_EVENTS_PER_FILE,
        "full_file": True,
        "n_alpha": 8,
        "n_candidates": n_candidates,
        "n_mt_master_bins": n_mt,
        "grid_sha256": grid_sha,
        "npz_path": str(npz_path),
        "hbb_mass_window_required": False,
        "anchor_applied_before_sr_selection": True,
        "historical_kappa_applied": False,
        "lumi_run3_fb": 308.0,
        "anchor_lumi_fb": 29.0,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{prefix}: wrong/stale physics identity: {key}")
    if meta.get("npz_sha256") != sha256(npz_path):
        raise ValueError(f"{prefix}: modified/corrupt npz")
    for cache_key, src_key in (
        ("raw_association_cache", "association"),
        ("raw_largeR_cache", "largeR"),
    ):
        if meta.get(cache_key) != str(file_paths[src_key]):
            raise ValueError(f"{prefix}: mismatched source association identity")
    for key in ("current_model_sha256", "legacy_source_sha256",
                "skip_manifest_sha256", "source_manifest_sha256",
                "benchmarks", "benchmark_configurations"):
        if key not in meta:
            raise ValueError(f"{prefix}: missing source/model provenance: {key}")
    if expected_models is not None and meta["benchmarks"] != expected_models:
        raise ValueError(f"{prefix}: incompatible benchmark model inventory")
    if meta.get("expected_skips") != meta.get("observed_skips"):
        raise ValueError(f"{prefix}: incomplete association skip closure")
    if int(meta.get("diagnostics", {}).get("raw_events", -1)) != EXPECTED_EVENTS_PER_FILE:
        raise ValueError(f"{prefix}: missing source events")
    with np.load(npz_path, allow_pickle=False) as blob:
        y = np.asarray(blob["template_sumw"], dtype=float)
        m = np.asarray(blob["template_second_moment"], dtype=float)
        nev = np.asarray(blob["template_nonzero_events"])
        ay = np.asarray(blob["anchor_sumw_29fb"], dtype=float)
        a2 = np.asarray(blob["anchor_second_moment_29fb"], dtype=float)
        an = np.asarray(blob["anchor_nonzero_events"])
        edges = np.asarray(blob["master_mt_edges"], dtype=float)
    n_models = len(meta["benchmarks"])
    if (
        y.shape != (n_models,n_candidates,n_mt)
        or m.shape != (n_models,n_candidates,n_mt,n_mt)
        or nev.shape != (n_models,n_candidates)
        or ay.shape != (n_models,) or a2.shape != (n_models,)
        or an.shape != (n_models,) or edges.shape != (n_mt+1,)
    ):
        raise ValueError(f"{prefix}: inconsistent source fake array shapes")
    if (
        not np.all(np.isfinite(y)) or not np.all(np.isfinite(m))
        or not np.all(np.isfinite(ay)) or not np.all(np.isfinite(a2))
        or np.any(np.diagonal(m,axis1=-2,axis2=-1) < -1e-12)
        or np.any(a2 < 0) or np.any(nev < 0) or np.any(an < 0)
    ):
        raise ValueError(f"{prefix}: nonphysical MC variance or nonfinite values")
    if not np.allclose(m, np.swapaxes(m, -1, -2), rtol=1e-10, atol=1e-8):
        raise ValueError(f"{prefix}: nonsymmetric fake event-level covariance")
    return meta, (y,m,nev,ay,a2,an,edges)


def merge_all(manifest, source_root, grid_path, output_root, target_prefix,
              old_kappa_rel_tol=0.01):
    manifest, source_root = Path(manifest), Path(source_root)
    grid_path, output_root = Path(grid_path), Path(output_root)
    target_prefix = Path(target_prefix)
    out_json = Path(str(target_prefix)+".json")
    out_npz = Path(str(target_prefix)+".npz")
    if any(path.exists() for path in (out_json,out_npz)):
        raise FileExistsError("refuse to overwrite existing complete/partial merge")
    grid = validate_grid(grid_path, output_root, check_source=True)
    grid_hash = sha256(grid_path)
    payload = json.loads(manifest.read_text())
    rows = [r for r in payload["files"] if r["sample"] in EXPECTED_RUN3_FILES]
    expected_total = sum(EXPECTED_RUN3_FILES.values())
    if len(rows) != expected_total:
        raise ValueError(f"provenance inventory contains {len(rows)} / {expected_total} raw files")
    unique = {(r["sample"],int(r["ordinal"])) for r in rows}
    if len(unique) != expected_total:
        raise ValueError("duplicate source file in manifest")
    for name, expected in EXPECTED_RUN3_FILES.items():
        if len([r for r in rows if r["sample"]==name]) != expected:
            raise ValueError(f"missing original raw files for {name}")
    nskip = verify_skip_manifest(
        source_root / "run3_association_complete_skip_manifest.json", rows
    )
    if nskip != 2:
        raise ValueError("wrong exact ttbar association-skip accounting")
    n_candidates = grid["n_candidates"]
    mt_edges = np.asarray(grid["master_mt_edges_GeV"],dtype=float)
    n_mt = len(mt_edges)-1
    names = list(OLD_KAPPAS)
    shape_y = (len(names),n_candidates,n_mt)
    shape_m = (len(names),n_candidates,n_mt,n_mt)
    total_y = np.zeros(shape_y)
    total_m = np.zeros(shape_m)
    total_n = np.zeros((len(names),n_candidates),dtype=np.int64)
    anchor = np.zeros(len(names))
    anchor_var = np.zeros(len(names))
    anchor_n = np.zeros(len(names),dtype=np.int64)
    by_sample = {
        s: {"source_files":0,"raw_events":0,"skips":0,
            "sumw":np.zeros(shape_y),"second":np.zeros(shape_m)}
        for s in EXPECTED_RUN3_FILES
    }
    identity = None
    completed=Counter()
    for src in sorted(rows,key=lambda r:(r["sample"],int(r["ordinal"]))):
        prefix = shard_prefix(output_root,src["sample"],src["ordinal"])
        meta,(y,m,n,ay,av,an,edges) = validate_full_shard(
            prefix,src,source_root,grid_hash,n_candidates,n_mt,
            expected_models=names
        )
        if not np.array_equal(edges,mt_edges):
            raise ValueError(f"{prefix}: wrong master mT edges")
        marker = {
            key: meta[key] for key in (
                "current_model_sha256","legacy_source_sha256",
                "grid_sha256","skip_manifest_sha256",
                "source_manifest_sha256","benchmark_configurations"
            )
        }
        if identity is None:
            identity=marker
        elif identity != marker:
            raise ValueError(f"{prefix}: models/config/source identity differs across shards")
        if meta["source_manifest_sha256"] != sha256(manifest):
            raise ValueError(f"{prefix}: outdated source manifest")
        if meta["skip_manifest_sha256"] != sha256(
                source_root/"run3_association_complete_skip_manifest.json"):
            raise ValueError(f"{prefix}: stale association manifest")
        total_y += y
        total_m += m
        total_n += n
        anchor += ay
        anchor_var += av
        anchor_n += an
        sample = src["sample"]
        by_sample[sample]["source_files"] += 1
        by_sample[sample]["raw_events"] += meta["events_processed"]
        by_sample[sample]["skips"] += len(meta["observed_skips"])
        by_sample[sample]["sumw"] += y
        by_sample[sample]["second"] += m
        completed[sample] += 1
    if sum(x["skips"] for x in by_sample.values()) != 2:
        raise ValueError("total source association skip count did not close")
    if any(by_sample[s]["source_files"] != count or
           by_sample[s]["raw_events"] != count*EXPECTED_EVENTS_PER_FILE
           for s,count in EXPECTED_RUN3_FILES.items()):
        raise ValueError("all-4M source completeness failed")
    if np.any(anchor<=0) or np.any(anchor_var<0):
        raise ValueError("nonpositive/invalid public e-mu proxy anchor")
    kappas = ATLAS_PUBLIC_EMU_TARGET / anchor
    relative_diff = {
        name: float(kappas[i]/OLD_KAPPAS[name]-1.0)
        for i,name in enumerate(names)
    }
    if old_kappa_rel_tol <= 0:
        raise ValueError("old kappa tolerance must be positive")
    closure = all(abs(x) <= old_kappa_rel_tol for x in relative_diff.values())
    if not closure:
        raise ValueError(
            "new full-source public-anchor calibration fails old 4M agreement "
            f"(rel_tol={old_kappa_rel_tol}): {relative_diff}; "
            "do not silently copy historical kappa factors"
        )
    calibrated_y = total_y * kappas[:,None,None]
    calibrated_m = total_m * kappas[:,None,None,None]**2
    diag = np.diagonal(calibrated_m,axis1=2,axis2=3)
    neff = np.divide(calibrated_y**2,diag,out=np.zeros_like(calibrated_y),where=diag>0)
    metadata = {
        "status": "RUN3_4M_RAW_FAKESIM_MERGED_ANCHOR_CLOSED_NOT_FINAL",
        "schema_version": 1, "source_era": "run3",
        "sources": dict(completed),
        "n_source_files": expected_total,
        "n_source_events": expected_total*EXPECTED_EVENTS_PER_FILE,
        "n_association_skipped": 2,
        "current_luminosity_fb": 308.0,
        "anchor_reference": "ATLAS PLB 848 (2024) 138376 arXiv:2308.09529 Table 1",
        "anchor_emu_target_yield_29fb": ATLAS_PUBLIC_EMU_TARGET,
        "anchor_emu_target_error_29fb": ATLAS_PUBLIC_EMU_TARGET_ERR,
        "source_proxy_emu_yield_29fb": anchor.tolist(),
        "source_proxy_emu_sumw2_29fb": anchor_var.tolist(),
        "anchor_effective_MC_events": np.divide(
            anchor**2,anchor_var,out=np.zeros_like(anchor),where=anchor_var>0
        ).tolist(),
        "benchmarks": names,
        "new_kappa": dict(zip(names,kappas.tolist())),
        "historical_kappa_comparison": relative_diff,
        "historical_kappa_max_relative_difference": old_kappa_rel_tol,
        "historical_kappa_pass": closure,
        "n_candidates": n_candidates,
        "n_mt_master_bins": n_mt,
        "source_manifest_sha256": sha256(manifest),
        "candidate_grid_sha256": grid_hash,
        "shared_identity": identity,
        "source_summary": {
            key:{
                "source_files":int(v["source_files"]),
                "raw_events":int(v["raw_events"]),
                "association_skipped":int(v["skips"]),
                "uncalibrated_integral_by_benchmark":np.sum(
                    v["sumw"],axis=(1,2)
                ).tolist(),
            } for key,v in by_sample.items()
        },
        "calibration_statistical_uncertainty_conditioned_on_kappa": True,
        "calibration_kappa_and_SR_yield_MC_correlation_not_parameterized": True,
        "fake_model_coverage_not_full_detector_calibration": True,
        "fakeaware_likelihood_built": False,
        "physics_validated": False,
        "hbb_mass_window_required": False,
        "full_source_covariance_preserved": True,
    }
    target_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz=Path(str(out_npz)+".tmp")
    tmp_json=Path(str(out_json)+".tmp")
    if tmp_npz.exists() or tmp_json.exists():
        raise FileExistsError("incomplete previous FakeSim full merge")
    try:
        with tmp_npz.open("wb") as stream:
            np.savez_compressed(
                stream,
                uncalibrated_sumw=total_y,
                uncalibrated_second_moment=total_m,
                calibrated_sumw=calibrated_y,
                calibrated_second_moment=calibrated_m,
                calibrated_neff=neff,
                nonzero_source_events=total_n,
                anchor_sumw_29fb=anchor,
                anchor_second_moment_29fb=anchor_var,
                anchor_nonzero_source_events=anchor_n,
                kappa=kappas,
                master_mt_edges=mt_edges,
            )
        tmp_npz.replace(out_npz)
        metadata["npz_file"]=str(out_npz)
        metadata["npz_sha256"]=sha256(out_npz)
        tmp_json.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
        tmp_json.replace(out_json)
    finally:
        tmp_npz.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
    print("PASS Run3 4M-source FakeSim merge with public-anchor closure")
    print("kappa: "+json.dumps(metadata["new_kappa"],sort_keys=True))
    print(f"outputs: {out_json} {out_npz}")
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--grid",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--output-prefix",type=Path)
    parser.add_argument("--old-kappa-rel-tol",type=float,default=.01)
    args=parser.parse_args()
    prefix=(args.output_prefix or (args.output_root/"fake_backgrounds"/"merged"/
                                   "run3_full_raw_fakesim"))
    merge_all(args.manifest,args.source_root,args.grid,args.output_root,prefix,
              old_kappa_rel_tol=args.old_kappa_rel_tol)


if __name__=="__main__":
    main()
