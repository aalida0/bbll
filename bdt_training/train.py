"""Train final resolved/boosted BDTs from Stage-1 selected ROOT files.

One model bank per era/topology; each bank contains five held-out-fold models.
Uses raw absolute sidecar weights for training. Luminosity scaling cancels under
per-family / per-signal-cell balancing and is applied later to physics yields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import sklearn
import uproot
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from bbll.config import load_config, load_stage_config
from .environment import verify_runtime

ANALYSIS = load_config()
TRAINING = load_stage_config("bdt_training")
FAMILIES = TRAINING["background_families"]
N_FOLDS = int(TRAINING["folds"])
SIGNAL_MASSES = [int(x) for x in ANALYSIS["samples"]["signal_masses_GeV"]]


def balance(y, absw, mass, decay, family):
    """Exactly match the original per-fold weight normalization."""
    w = np.zeros(len(y), dtype=np.float64)
    sig, bkg = y == 1, y == 0
    if not np.any(sig) or not np.any(bkg):
        raise ValueError("signal and background are both required")

    ids = np.unique(family[bkg])
    if not np.array_equal(ids, np.arange(len(FAMILIES))):
        raise ValueError(f"expected {len(FAMILIES)} background families, got {ids}")
    for fid in ids:
        m = bkg & (family == fid)
        total = np.sum(absw[m], dtype=np.float64)
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"background family {fid}: nonpositive total weight")
        w[m] = absw[m] / total
    w[bkg] /= np.sum(w[bkg], dtype=np.float64)

    cells = sorted(set(zip(mass[sig].tolist(), decay[sig].tolist())))
    required = {(m, d) for m in SIGNAL_MASSES for d in (0, 1, 2)}
    if set(cells) != required:
        raise ValueError(f"signal mass×decay coverage incomplete: missing={sorted(required - set(cells))}")
    for signal_mass, decay_id in cells:
        m = sig & (mass == signal_mass) & (decay == decay_id)
        total = np.sum(absw[m], dtype=np.float64)
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"signal cell {signal_mass}/{decay_id}: nonpositive weight")
        w[m] = absw[m] / total
    w[sig] /= np.sum(w[sig], dtype=np.float64)

    w *= len(w) / 2.0
    return w


def sample_paths(selected_root: Path, era: str):
    """Exact 26-sample inventory and original trainer's lexicographic order."""
    folder = selected_root / era
    inventory = ANALYSIS["samples"]
    stems = (
        [f"X{m}_{era}" for m in inventory["signal_masses_GeV"]]
        + [f"{x}_{era}" for x in inventory["prompt_backgrounds"]]
    )
    expected = {f"{stem}.root" for stem in stems}
    found = {p.name for p in folder.glob("*.root")}
    if found != expected:
        raise ValueError(f"{folder}: missing={sorted(expected-found)} extra={sorted(found-expected)}")
    manifest_path = folder / "selection_campaign.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing validated preselection manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "PASS" or manifest.get("era") != era
            or manifest.get("n_samples") != len(stems)):
        raise ValueError(f"{manifest_path}: invalid campaign manifest")
    reference = ANALYSIS["reference_selected_events"][era]
    if manifest.get("category2_counts") != reference:
        raise ValueError(f"{manifest_path}: reference selection totals mismatch")
    return [folder / name for name in sorted(expected)]


def read_pool(selected_root: Path, era: str, topology: str):
    """Use the same training events/order as the validated topology-specific reducer."""
    if topology not in ("resolved", "boosted") or era not in ANALYSIS["eras"]:
        raise ValueError("unknown era/topology")
    cols = list(ANALYSIS["features"][topology])
    if len(cols) != (20 if topology == "resolved" else 25):
        raise ValueError("unexpected frozen feature count")
    if "mt_hh_mH125" in cols or "n_tags_inside_largeR" in cols:
        raise ValueError("fit variable or b-tag multiplicity leaked into BDT")
    branches = cols + [
        "category3", "event_uid", "is_signal", "signal_mass_GeV",
        "signal_decay_id", "train_weight_base",
    ]
    bulk = int(TRAINING["sampling"]["bulk_divisor"])
    rare = int(TRAINING["sampling"]["other_divisor"])
    if min(bulk, rare) < 1 or N_FOLDS != 5:
        raise ValueError("unsupported training subsample or fold definition")

    groups = {
        sample: family_id
        for family_id, names in enumerate(FAMILIES.values())
        for sample in names
    }
    if set(groups) != set(ANALYSIS["samples"]["prompt_backgrounds"]):
        raise ValueError("training background groups do not match the global sample inventory")

    xs, ys, ws, folds, masses, decays, families = [], [], [], [], [], [], []
    for file in sample_paths(selected_root, era):
        sample = file.stem[: -len(f"_{era}")]
        is_sig = sample.startswith("X")
        fid = -1 if is_sig else groups[sample]
        divisor = rare if fid == list(FAMILIES).index("other") else bulk
        for a in uproot.iterate(f"{file}:events", branches, step_size="500 MB", library="np"):
            uid = a["event_uid"].astype(np.uint64)
            cat = a["category3"]
            chosen = (cat == 0 if topology == "resolved" else cat >= 1)
            # Match old reducer -> deterministic subsampling; no train/OOF leakage.
            keep = chosen & (((uid // N_FOLDS) % divisor) == 0)
            if not np.any(keep):
                continue
            labels = a["is_signal"][keep].astype(np.int8)
            if np.any(labels != int(is_sig)):
                raise ValueError(f"{file.name}: signal labels disagree with file role")
            absw = a["train_weight_base"][keep].astype(np.float64)
            if np.any(~np.isfinite(absw)) or np.any(absw < 0):
                raise ValueError(f"{file.name}: invalid absolute training weight")
            xs.append(np.column_stack([a[c][keep] for c in cols]).astype(np.float32))
            ys.append(labels)
            ws.append(absw)
            folds.append((uid[keep] % N_FOLDS).astype(np.int8))
            masses.append(a["signal_mass_GeV"][keep].astype(np.int32))
            decays.append(a["signal_decay_id"][keep].astype(np.int8))
            families.append(np.full(len(labels), fid, dtype=np.int8))
    if not xs:
        raise ValueError("empty training pool")
    return (
        np.concatenate(xs), np.concatenate(ys), np.concatenate(ws),
        np.concatenate(folds), np.concatenate(masses),
        np.concatenate(decays), np.concatenate(families),
    )


def run_training(selected_root: Path, output_root: Path, era: str,
                 topology: str, dry_run: bool = False):
    variant = TRAINING["model_variants"][topology]
    target = output_root / "models" / era / topology / variant
    if target.exists() and not dry_run:
        raise FileExistsError(f"model bank already exists: {target}")
    x, y, absw, fold, mass, decay, family = read_pool(selected_root, era, topology)
    global_weights = balance(y, absw, mass, decay, family)
    counts = [int(np.count_nonzero(fold == i)) for i in range(N_FOLDS)]
    print(f"{era}/{topology}: N={len(y):,}; signal={np.count_nonzero(y):,}; "
          f"background={len(y)-np.count_nonzero(y):,}; fold_counts={counts}", flush=True)
    print(f"global balanced weights: S={np.sum(global_weights[y == 1]):.6g} "
          f"B={np.sum(global_weights[y == 0]):.6g}", flush=True)
    # Even in --dry-run, validate every fold has all three prompt families,
    # all 16 mass x 3 decay cells, and positive weight sums.
    for i in range(N_FOLDS):
        train, valid = fold != i, fold == i
        balance(y[train], absw[train], mass[train], decay[train], family[train])
        balance(y[valid], absw[valid], mass[valid], decay[valid], family[valid])
    if dry_run:
        print("PASS: complete training pool and five-fold contract", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{variant}-", dir=target.parent))
    try:
        metrics = []
        for i in range(N_FOLDS):
            train, valid = fold != i, fold == i
            w_train = balance(y[train], absw[train], mass[train], decay[train], family[train])
            w_valid = balance(y[valid], absw[valid], mass[valid], decay[valid], family[valid])
            model = HistGradientBoostingClassifier(
                **TRAINING["hyperparameters"],
                random_state=int(TRAINING["seed"]) + i,
            )
            model.fit(x[train], y[train], sample_weight=w_train)
            auc = float(roc_auc_score(
                y[valid], model.predict_proba(x[valid])[:, 1],
                sample_weight=w_valid,
            ))
            with (tmp / f"fold{i}.pkl").open("wb") as f:
                pickle.dump(model, f)
            row = {
                "fold": i, "n_train": int(np.count_nonzero(train)),
                "n_valid": int(np.count_nonzero(valid)), "weighted_auc": auc,
                "n_iter": int(model.n_iter_),
            }
            metrics.append(row)
            print(f"{era}/{topology} fold{i}: AUC={auc:.6f} "
                  f"N_train={row['n_train']:,}, N_valid={row['n_valid']:,} "
                  f"iterations={row['n_iter']}", flush=True)
        features = list(ANALYSIS["features"][topology])
        (tmp / "features.txt").write_text("\n".join(features) + "\n")
        values = [row["weighted_auc"] for row in metrics]
        payload = {
            "era": era, "topology": topology, "variant": variant,
            "fold_rule": "event_uid % 5", "n_folds": N_FOLDS,
            "training_events": len(y), "fold_counts": counts,
            "signal_events": int(np.count_nonzero(y)), "background_events": int(len(y)-np.count_nonzero(y)),
            "features": features,
            "features_sha256": hashlib.sha256(
                ("\n".join(features) + "\n").encode()).hexdigest(),
            "hyperparameters": TRAINING["hyperparameters"],
            "bulk_divisor": int(TRAINING["sampling"]["bulk_divisor"]),
            "other_divisor": int(TRAINING["sampling"]["other_divisor"]),
            "seed": int(TRAINING["seed"]),
            "weighted_auc_mean": float(np.mean(values)),
            "weighted_auc_std": float(np.std(values)),
            "fold_metrics": metrics, "python": sys.version.split()[0],
            "numpy": np.__version__, "sklearn": sklearn.__version__,
            "physics_weights": "raw sidecar abs weights; no lumi rescaling in training",
        }
        (tmp / "training_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        if target.exists():
            raise FileExistsError(target)
        tmp.rename(target)
        print(f"PASS: models saved to {target}", flush=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("era", choices=("run2", "run3"))
    p.add_argument("topology", choices=("resolved", "boosted"))
    p.add_argument("--selected-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    verify_runtime()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    run_training(args.selected_root, args.output_root,
                 args.era, args.topology, args.dry_run)


if __name__ == "__main__":
    main()
