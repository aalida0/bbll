"""Validate the three complete 10k-file Run-3 FakeSim jobs before 397 more.

This is a read-only numerical gate. Uses the strict merger shard validator,
then checks the original 400-source manifest, current model and source hashes,
signed master-bin covariance PSD and public e-mu proxy contribution.

Absolute 29/fb proxy kappa closure requires ALL 400 sources, never extrapolate
from the first 3 files or from scored-hypothesis counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .full_grid import sha256, validate_grid
from .full_merge import OLD_KAPPAS, shard_prefix, validate_full_shard

SAMPLES = ("Wenu_b", "Wmunu_b", "ttbar_semileptonic")


def validate_three(root, source_root, provenance, grid_path=None):
    root, source_root, provenance = map(Path, (root, source_root, provenance))
    grid_path = (Path(grid_path) if grid_path is not None else
                 root / "fake_backgrounds" / "run3_candidate_grid.json")
    grid = validate_grid(grid_path, root, check_source=True)
    grid_sha = sha256(grid_path)
    original = json.loads(provenance.read_text())
    rows = [
        row for row in original["files"]
        if row["sample"] in SAMPLES and int(row["ordinal"]) == 1
    ]
    if len(rows) != 3 or {x["sample"] for x in rows} != set(SAMPLES):
        raise ValueError("expected exactly one original ordinal-1 source per class")
    nmt = len(grid["master_mt_edges_GeV"]) - 1
    identities = []
    summary = {}
    for row in rows:
        sample = row["sample"]
        prefix = shard_prefix(root, sample, 1)
        meta, (y, m, counts, anchor, avar, anc, edges) = validate_full_shard(
            prefix, row, source_root, grid_sha,
            grid["n_candidates"], nmt, expected_models=list(OLD_KAPPAS)
        )
        if meta["source_manifest_sha256"] != sha256(provenance):
            raise ValueError(f"{sample}: original source provenance changed")
        skip = source_root / "run3_association_complete_skip_manifest.json"
        if meta["skip_manifest_sha256"] != sha256(skip):
            raise ValueError(f"{sample}: skip manifest changed")
        source_norm = source_root / f"run3_{sample}_raw_weight_summary.json"
        if meta["normalization_summary_sha256"] != sha256(source_norm):
            raise ValueError(f"{sample}: source normalization changed")
        if not np.array_equal(edges, grid["master_mt_edges_GeV"]):
            raise ValueError(f"{sample}: master mT edges differ")
        if np.any(anc > meta["events_processed"]) or np.any(
                counts > meta["events_processed"]):
            raise ValueError(f"{sample}: fake event accounting exceeds source count")
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(m))
                and np.all(np.isfinite(anchor)) and np.all(np.isfinite(avar))):
            raise ValueError(f"{sample}: nonfinite master-bin moments")
        if np.any(avar < 0) or np.any(anc < 0):
            raise ValueError(f"{sample}: invalid public-anchor moments")

        # An event-level second moment is PSD. Test each individual
        # covariance only when nonzero; tolerate finite FP noise.
        for model_idx in range(len(OLD_KAPPAS)):
            for cand_idx in range(grid["n_candidates"]):
                mat = m[model_idx, cand_idx]
                if not np.any(mat):
                    continue
                eig = np.linalg.eigvalsh(mat)
                scale = max(float(np.max(np.abs(np.diag(mat)))), 1.0)
                if np.min(eig) < -1e-8 * scale:
                    raise ValueError(f"{sample}: non-PSD covariance "
                                     f"model {model_idx}, candidate {cand_idx}")

        chosen = [
            c for c in grid["fakesim_candidate_rows"]
            if c["mass_GeV"] == 800
        ]
        diagnostics = []
        for c in chosen:
            ix = int(c["index"])
            for model_idx, name in enumerate(OLD_KAPPAS):
                yt = float(np.sum(y[model_idx,ix]))
                v = float(np.sum(m[model_idx,ix]))
                if not np.isfinite(v) or v < -1e-10:
                    raise ValueError(f"{sample}: negative integrated second moment")
                neff = yt * yt / v if v > 0 else 0.0
                if yt != 0 or v != 0:
                    diagnostics.append({
                        "channel": c["channel"],
                        "efficiency_target": c["efficiency_target"],
                        "benchmark": name, "uncalibrated_sumw": yt,
                        "sumw2_source_event": v, "neff": neff,
                    })
        identity = {
            key: meta[key] for key in (
                "current_model_sha256", "legacy_source_sha256",
                "benchmark_configurations", "benchmarks",
                "grid_sha256", "source_manifest_sha256",
                "skip_manifest_sha256",
            )
        }
        identities.append(identity)
        summary[sample] = {
            "status": meta["status"],
            "raw_events": meta["events_processed"],
            "association_skips": len(meta["observed_skips"]),
            "hypotheses_scored": meta["diagnostics"].get("scored_hypotheses", 0),
            "scored_outside_old_hbb_window": meta["diagnostics"].get(
                "scored_outside_old_mass_window", 0
            ),
            "nonzero_selected_candidate_model_pairs": int(
                np.count_nonzero(counts)
            ),
            "anchor_sumw_29fb": dict(zip(OLD_KAPPAS, anchor.tolist())),
            "anchor_sumw2_29fb": dict(zip(OLD_KAPPAS, avar.tolist())),
            "anchor_source_neff_29fb": {
                name: float(anchor[i]**2/avar[i]) if avar[i] > 0 else 0.0
                for i, name in enumerate(OLD_KAPPAS)
            },
            "mass800_nonzero_candidates": diagnostics,
        }
    if not all(x == identities[0] for x in identities[1:]):
        raise ValueError("different current BDT, fake benchmark, or source hashes")
    return {
        "schema_version": 1,
        "status": "RUN3_FAKESIM_THREE_FULL_SHARDS_NUMERICAL_PASS",
        "physics_validated": False,
        "n_shards_checked": 3,
        "raw_events_verified": 30000,
        "calibration_kappa_frozen": False,
        "full_campaign_ready_for_397_remaining": True,
        "samples": summary,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path(
        "/data/aalida/bbll_ml/fakes"
    ))
    p.add_argument("--manifest", type=Path, default=Path(
        "/data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json"
    ))
    p.add_argument("--grid", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    result = validate_three(a.output_root, a.source_root, a.manifest, a.grid)
    print("3-SOURCE RUN3 FAKESIM NUMERICAL VALIDATION")
    print("=" * 75)
    for sample in SAMPLES:
        d = result["samples"][sample]
        print(f"{sample}: {d['raw_events']} events; "
              f"{d['hypotheses_scored']} scored; "
              f"{d['nonzero_selected_candidate_model_pairs']} nonzero "
              "candidate/benchmark pairs")
        for name in OLD_KAPPAS:
            print(f"  {name}: anchor_29fb={d['anchor_sumw_29fb'][name]:.8g} "
                  f"anchor_neff={d['anchor_source_neff_29fb'][name]:.4g}")
        for row in d["mass800_nonzero_candidates"][:8]:
            print("  m800 {} eff={} {} yield={:.6g} Neff={:.3g}".format(
                row["channel"], row["efficiency_target"], row["benchmark"],
                row["uncalibrated_sumw"], row["neff"]))
    print("RESULT:", result["status"])
    print("NOTE: source-level anchor entries are not calibrated yields; "
          "full 4M normalization/closure still required.")
    if a.report is not None:
        if a.report.exists():
            raise FileExistsError(f"report already exists: {a.report}")
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print("Saved:", a.report)


if __name__ == "__main__":
    main()
