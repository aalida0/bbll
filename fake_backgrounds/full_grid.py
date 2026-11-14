"""Frozen prompt-only candidate grid for full raw-Delphes FakeSim production.

This freezes ALL valid prompt-only signal-efficiency candidates, not just the
exhaustive-scan winners, so fake-aware SR selection can be reoptimized.
Master mT binning is shared with the current SR scan. Input SHA256 checks
prevent silently mixing score grids from different analysis configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bbll.config import load_config, load_stage_config
from sr_optimization.scan import CHANNEL_TO_TOPOLOGY, validate_existing_screen


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def create_grid(root):
    root = Path(root)
    config = load_stage_config("sr_optimization")
    masses = [int(x) for x in load_config()["samples"]["signal_masses_GeV"]]
    if not masses or len(masses) != len(set(masses)):
        raise ValueError("empty or repeated mass grid")
    edges = np.asarray(config["master_mt_edges_GeV"], dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 3 or
            not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0)):
        raise ValueError("invalid master mT edges")
    candidates = []
    hashes = {}
    by_channel = {ch: 0 for ch in CHANNEL_TO_TOPOLOGY}
    for mass in masses:
        path = root / "sr_optimization" / "run3" / f"m{mass}" / "prompt_candidate_screen.json"
        doc = validate_existing_screen(path, "run3", mass)
        hashes[str(mass)] = sha256(path)
        for channel in CHANNEL_TO_TOPOLOGY:
            for local_index, item in enumerate(doc["candidates"][channel]):
                # The frozen optimizer already validates signed MC statistics.
                # Full FakeSim reuses every valid candidate, irrespective of
                # whether adding fake backgrounds makes it statistically viable.
                score = float(item["score_threshold"])
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"invalid score in {mass}/{channel}/{local_index}")
                if int(item["n_mt_bins"]) < config["minimum_mt_bins"]:
                    raise ValueError("candidate below frozen mT shape requirement")
                candidates.append({
                    "index": len(candidates), "mass_GeV": mass,
                    "channel": channel, "local_index": local_index,
                    "efficiency_target": float(item["efficiency_target"]),
                    "score_threshold": score,
                })
                by_channel[channel] += 1
    if not candidates or not all(by_channel.values()):
        raise ValueError("empty or incomplete full FakeSim threshold inventory")
    return {
        "schema_version": 1,
        "status": "FROZEN_RUN3_ALL_MASS_FAKESIM_CANDIDATE_GRID",
        "era": "run3",
        "source": "validated_prompt_only_candidate_screen",
        "masses_GeV": masses,
        "channels": list(CHANNEL_TO_TOPOLOGY),
        "n_candidates": len(candidates),
        "candidate_counts_by_channel": by_channel,
        "master_mt_edges_GeV": edges.tolist(),
        "source_screen_sha256": hashes,
        "sr_config_sha256": sha256(Path(__file__).resolve().parents[1] /
                                    "sr_optimization" / "config.yaml"),
        "fakesim_candidate_rows": candidates,
        "physics_validated": False,
    }


def validate_grid(path, root, check_source=True):
    path = Path(path)
    doc = json.loads(path.read_text())
    rows = doc.get("fakesim_candidate_rows")
    if (doc.get("schema_version") != 1 or
            doc.get("status") != "FROZEN_RUN3_ALL_MASS_FAKESIM_CANDIDATE_GRID" or
            not isinstance(rows, list) or len(rows) != doc.get("n_candidates")):
        raise ValueError("invalid full-source FakeSim score-grid schema")
    for idx, row in enumerate(rows):
        if (row.get("index") != idx or row.get("channel") not in CHANNEL_TO_TOPOLOGY or
                not math.isfinite(float(row["score_threshold"]))):
            raise ValueError("invalid score-grid row")
    if check_source:
        expected = create_grid(root)
        if doc != expected:
            raise ValueError(
                "FakeSim grid disagrees with current Run3 SR screen or config; "
                "do not reuse stale candidate grid"
            )
    return doc


def pack_candidates(doc):
    by_channel = {}
    for channel in CHANNEL_TO_TOPOLOGY:
        subset = [row for row in doc["fakesim_candidate_rows"]
                  if row["channel"] == channel]
        by_channel[channel] = {
            "index": np.asarray([row["index"] for row in subset], dtype=np.int32),
            "score": np.asarray([row["score_threshold"] for row in subset],
                                dtype=np.float64),
        }
    return by_channel, np.asarray(doc["master_mt_edges_GeV"], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Validate exact existing grid rather than overwriting")
    args = parser.parse_args()
    out = (args.output if args.output is not None else
           args.output_root / "fake_backgrounds" / "run3_candidate_grid.json")
    if out.exists():
        if not args.resume:
            raise FileExistsError(out)
        grid = validate_grid(out, args.output_root)
        print(f"PASS: validated existing grid with {grid['n_candidates']} candidates")
        return
    if args.resume:
        raise FileNotFoundError(out)
    doc = create_grid(args.output_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"PASS: froze {doc['n_candidates']} Run3 FakeSim candidate thresholds")
    print(f"file: {out}")


if __name__ == "__main__":
    main()
