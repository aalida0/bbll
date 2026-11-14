"""Verify full-event out-of-fold scores against frozen selected ROOT data."""

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from bbll.config import load_config, load_stage_config
from .environment import verify_runtime
from .score import KEEP, N_FOLDS, model_bank, score_chunk
from .train import sample_paths


def verify_sample(source, scored, era, topology, hashes, features=(),
                  models=None, spot_events=100):
    meta = scored.with_suffix(".oof.json")
    if not scored.is_file() or not meta.is_file():
        raise FileNotFoundError(f"missing OOF ROOT/metadata: {scored}")
    record = json.loads(meta.read_text())
    if (record.get("era") != era or record.get("topology") != topology
            or record.get("input") != str(source)
            or record.get("model_sha256") != hashes
            or record.get("fold_rule") != "event_uid % 5"
            or record.get("n_models") != N_FOLDS):
        raise ValueError(f"{meta}: OOF metadata/model hash mismatch")

    checked, raw_sum, spotted = 0, 0., 0
    columns = tuple(dict.fromkeys((*KEEP, *features))) if models else KEEP
    fields = (*KEEP, "ml_score", "oof_fold")
    with uproot.open(source) as inp, uproot.open(scored) as outf:
        original, output = inp["events"], outf["events"]
        if record.get("source_events") != int(original.num_entries):
            raise ValueError(f"{source}: source count mismatch")
        for batch in original.iterate(columns, step_size="100 MB", library="np"):
            cat = batch["category3"]
            mask = (cat == 0) if topology == "resolved" else (cat >= 1)
            n = int(np.count_nonzero(mask))
            if n == 0:
                continue
            other = output.arrays(fields, entry_start=checked,
                                  entry_stop=checked + n, library="np")
            if len(other["ml_score"]) != n:
                raise ValueError(f"{scored}: incomplete score output")
            for field in KEEP:
                if not np.array_equal(batch[field][mask], other[field],
                                      equal_nan=True):
                    raise ValueError(f"{scored}: source mismatch in {field} at row {checked}")
            folds = (batch["event_uid"][mask].astype(np.uint64) % N_FOLDS)
            if not np.array_equal(other["oof_fold"], folds):
                raise ValueError(f"{scored}: incorrect held-out-fold assignment")
            scores = other["ml_score"]
            if (not np.all(np.isfinite(scores))
                    or np.any(scores < 0) or np.any(scores > 1)):
                raise ValueError(f"{scored}: invalid scores")
            if models is not None and spotted < spot_events:
                m = min(n, spot_events - spotted)
                xx = np.column_stack([batch[k][mask][:m] for k in features]).astype(np.float32)
                expected = score_chunk(xx, folds[:m], models)
                if not np.array_equal(scores[:m], expected):
                    raise ValueError(f"{scored}: scores do not reproduce held-out models")
                spotted += m
            raw_sum += float(np.sum(other["physics_weight"], dtype=np.float64))
            checked += n
        if checked != output.num_entries or checked != record.get("events"):
            raise ValueError(f"{scored}: row-count mismatch")
    if not np.isclose(raw_sum, record.get("sumw_raw"), rtol=1e-11, atol=1e-9):
        raise ValueError(f"{scored}: signed weight sum mismatch")
    return {"sample": source.stem, "events": checked, "spot_checked": spotted}


def validate_campaign(selected_root, output_root, era, topology,
                      sample=None, spot_events=100):
    features, models, hashes = model_bank(output_root, era, topology)
    paths = sample_paths(selected_root, era)
    if sample is not None:
        paths = [p for p in paths if p.stem == sample]
        if not paths:
            raise ValueError(f"unknown sample {sample} for {era}")
    dest = (output_root / load_stage_config("bdt_training")["scoring"]["output_subdir"]
            / era / topology)
    rows = []
    for path in paths:
        row = verify_sample(path, dest / path.name, era, topology, hashes,
                            features, models, spot_events)
        rows.append(row)
        print(f"{era}/{topology}/{path.stem}: PASS {row['events']:,}", flush=True)
    total = sum(row["events"] for row in rows)
    if sample is None:
        expected = load_config()["reference_selected_events"][era][topology]
        manifest = dest / "oof_campaign.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        saved = json.loads(manifest.read_text())
        if (total != expected or saved.get("status") != "PASS"
                or saved.get("events") != expected or saved.get("era") != era
                or saved.get("topology") != topology
                or len(saved.get("samples", [])) != len(paths)):
            raise ValueError(f"{era}/{topology}: campaign closure failure")
        (dest / "oof_validation.json").write_text(json.dumps({
            "status": "PASS", "era": era, "topology": topology,
            "events": total, "reference_events": expected,
            "checked": "event identity, categories, raw signed weights, mT, folds, scores",
            "samples": rows,
        }, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {era}/{topology}: {total:,} verified events", flush=True)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("era", choices=("run2", "run3"))
    p.add_argument("topology", choices=("resolved", "boosted"))
    p.add_argument("--selected-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--sample")
    p.add_argument("--spot-events", type=int, default=100)
    args = p.parse_args()
    if args.spot_events < 0:
        p.error("--spot-events must be >= 0")
    verify_runtime()
    validate_campaign(args.selected_root, args.output_root, args.era,
                      args.topology, args.sample, args.spot_events)


if __name__ == "__main__":
    main()
