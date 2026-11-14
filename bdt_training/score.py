"""Write full-event, held-out-fold BDT scores without changing physics weights.

Each selected event is scored ONLY by the model trained without its fold.
The output is a compact ROOT tree per original sample and topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import uproot
from sklearn.ensemble import HistGradientBoostingClassifier

from bbll.config import load_config, load_stage_config
from .environment import verify_runtime
from .train import sample_paths

ANALYSIS = load_config()
SETTINGS = load_stage_config("bdt_training")
N_FOLDS = int(SETTINGS["folds"])
KEEP = (
    "event_uid", "source_entry", "category2", "category3",
    "physics_weight", "mt_hh_mH125", "is_signal",
    "signal_mass_GeV", "signal_decay_id", "n_tags_inside_largeR",
)


def model_bank(root: Path, era: str, topology: str):
    variant = SETTINGS["model_variants"][topology]
    folder = root / "models" / era / topology / variant
    features = list(ANALYSIS["features"][topology])
    expected_hash = hashlib.sha256(("\n".join(features) + "\n").encode()).hexdigest()
    metrics = json.loads((folder / "training_metrics.json").read_text())
    if (metrics.get("era") != era or metrics.get("topology") != topology
            or metrics.get("features") != features
            or metrics.get("features_sha256") != expected_hash
            or metrics.get("fold_rule") != "event_uid % 5"
            or metrics.get("n_folds") != N_FOLDS
            or len(metrics.get("fold_metrics", [])) != N_FOLDS):
        raise ValueError(f"{folder}: trained model-bank metadata mismatch")
    if (metrics.get("numpy") != "2.0.2"
            or metrics.get("sklearn") != "1.6.1"
            or not str(metrics.get("python", "")).startswith("3.9.")):
        raise ValueError(f"{folder}: training environment metadata mismatch")
    models, file_hashes = [], []
    for fold in range(N_FOLDS):
        path = folder / f"fold{fold}.pkl"
        file_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with path.open("rb") as f:
            model = pickle.load(f)  # Only load models produced by our own training.
        if (not isinstance(model, HistGradientBoostingClassifier)
                or model.n_features_in_ != len(features)
                or not np.array_equal(model.classes_, [0, 1])
                or model.n_iter_ != metrics["fold_metrics"][fold]["n_iter"]):
            raise ValueError(f"{path}: invalid trained model")
        models.append(model)
    return features, models, file_hashes


def score_chunk(x, fold, models):
    """Assign the held-out model for each event; never ensemble in-fold models."""
    if len(models) != N_FOLDS or np.any((fold < 0) | (fold >= N_FOLDS)):
        raise ValueError("invalid fold/model count")
    score = np.empty(len(fold), dtype=np.float32)
    for i in range(N_FOLDS):
        mask = fold == i
        if np.any(mask):
            score[mask] = models[i].predict_proba(x[mask])[:, 1].astype(np.float32)
    if not np.all(np.isfinite(score)) or np.any((score < 0) | (score > 1)):
        raise ValueError("invalid out-of-fold score")
    return score


def score_sample(source: Path, output: Path, era: str, topology: str,
                 features, models, model_hashes, resume=False):
    metadata = output.with_suffix(".oof.json")
    with uproot.open(source) as f:
        n_source = int(f["events"].num_entries)
    if output.exists() or metadata.exists():
        if not (resume and output.is_file() and metadata.is_file()):
            raise FileExistsError(f"{output}: existing or incomplete result")
        report = json.loads(metadata.read_text())
        with uproot.open(output) as f:
            n_written = int(f["events"].num_entries)
        if (report.get("input") != str(source)
                or report.get("model_sha256") != model_hashes
                or report.get("source_events") != n_source
                or report.get("events") != n_written
                or report.get("era") != era
                or report.get("topology") != topology):
            raise ValueError(f"{output}: invalid resume metadata or ROOT count")
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".root.tmp")
    if tmp.exists():
        raise FileExistsError(f"{tmp}: unfinished previous output")
    count, signed_sumw = 0, 0.0
    cols = list(dict.fromkeys(features + list(KEEP)))
    try:
        with uproot.recreate(tmp) as target:
            dest = None
            for a in uproot.iterate(f"{source}:events", cols,
                                    step_size=SETTINGS["scoring"]["chunk_size"],
                                    library="np"):
                tags = a["category3"]
                chosen = (tags == 0) if topology == "resolved" else (tags >= 1)
                if not np.any(chosen):
                    continue
                uid = a["event_uid"][chosen].astype(np.uint64)
                fold = (uid % N_FOLDS).astype(np.int8)
                x = np.column_stack([a[k][chosen] for k in features]).astype(np.float32)
                score = score_chunk(x, fold, models)

                payload = {k: a[k][chosen] for k in KEEP}
                payload["ml_score"] = score
                payload["oof_fold"] = fold
                weight = payload["physics_weight"]
                if (not np.all(np.isfinite(weight))
                        or not np.all(np.isfinite(payload["mt_hh_mH125"]))):
                    raise ValueError(f"{source}: invalid signed weight or fit observable")
                if (np.any(payload["category2"] != (topology == "boosted"))
                        or (topology == "resolved" and np.any(payload["category3"] != 0))
                        or (topology == "boosted" and np.any(payload["category3"] < 1))):
                    raise ValueError(f"{source}: inconsistent category mapping")

                if dest is None:
                    dest = target.mktree("events", {k: v.dtype for k, v in payload.items()})
                dest.extend(payload)
                count += len(score)
                signed_sumw += float(np.sum(weight, dtype=np.float64))
            if dest is None:
                raise ValueError(f"{source}: no events in {topology}")
        with uproot.open(tmp) as f:
            if f["events"].num_entries != count:
                raise ValueError(f"{tmp}: entry count mismatch")
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": 1, "input": str(source), "output": str(output),
        "era": era, "topology": topology, "source_events": n_source,
        "events": count, "sumw_raw": signed_sumw,
        "n_models": N_FOLDS, "fold_rule": "event_uid % 5",
        "model_sha256": model_hashes,
        "weights": "raw signed physics_weight; luminosity adjustment downstream",
    }
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def score_campaign(selected_root: Path, output_root: Path,
                   era: str, topology: str, resume=False, sample=None):
    if era not in ANALYSIS["eras"] or topology not in SETTINGS["model_variants"]:
        raise ValueError("unknown era/topology")
    features, models, hashes = model_bank(output_root, era, topology)
    paths = sample_paths(selected_root, era)
    if sample:
        paths = [p for p in paths if p.stem == sample]
        if not paths:
            raise ValueError(f"unknown sample {sample} for {era}")
    dest = output_root / SETTINGS["scoring"]["output_subdir"] / era / topology
    total = 0
    reports = []
    for path in paths:
        report = score_sample(path, dest / path.name, era, topology,
                              features, models, hashes, resume=resume)
        total += report["events"]
        reports.append({"sample": path.stem, "events": report["events"]})
        print(f"{era}/{topology}/{path.stem}: {report['events']:,} scored", flush=True)
    if sample is None:
        expected = ANALYSIS["reference_selected_events"][era][topology]
        if total != expected:
            raise ValueError(f"OOF event count {total} != frozen {expected}")
        (dest / "oof_campaign.json").write_text(json.dumps({
            "status": "PASS", "era": era, "topology": topology,
            "events": total, "samples": reports,
            "reference_events": expected,
        }, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {era}/{topology} OOF scores: {total:,} events", flush=True)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("era", choices=("run2", "run3"))
    p.add_argument("topology", choices=("resolved", "boosted"))
    p.add_argument("--selected-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--sample", help="Score just one sample for a pilot")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    verify_runtime()
    score_campaign(a.selected_root, a.output_root,
                   a.era, a.topology, a.resume, a.sample)


if __name__ == "__main__":
    main()
