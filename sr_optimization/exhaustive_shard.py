"""Evaluate one deterministic shard of the exhaustive prompt-only pyhf scan.

A shard records all assigned CLs combinations, successful or failed.
A nonzero exit on failures prevents a misleading complete Condor campaign.
"""

import argparse
import json
import math
from pathlib import Path

from .exhaustive_inputs import (
    assigned_to_shard, load_inputs, shard_path,
)
from .prompt_likelihood import expected_mu95


def run_shard(root, era, mass, index, shards, mu_max=1e5):
    screen, _, _, identity = load_inputs(root, era, mass)
    if not (type(index) is int and type(shards) is int
            and shards >= 1 and 0 <= index < shards):
        raise ValueError("invalid shard index/count")
    if not (math.isfinite(mu_max) and 0 < mu_max <= 1e10):
        raise ValueError("invalid mu_max")

    path = shard_path(root, era, mass, index, shards)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {path}")
    rows, failures = [], []
    for task in assigned_to_shard(screen, index, shards):
        chosen = [
            screen["candidates"][ch][item]
            for ch, item in task["candidate_indices"].items()
        ]
        try:
            limit, solver = expected_mu95(chosen, mu_max)
            if not (math.isfinite(limit) and limit > 0):
                raise ValueError("nonfinite/negative median CLs limit")
            rows.append({
                **task, "expected_mu95": float(limit), "solver": solver
            })
        except Exception as exc:
            failures.append({
                **task, "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:1000],
            })
    expected = len(range(index, identity["total_combinations"], shards))
    if len(rows) + len(failures) != expected:
        raise RuntimeError("shard task accounting failed")
    result = {
        **identity, "shard_index": index, "shard_count": shards,
        "mu_max": float(mu_max),
        "status": ("COMPLETE_EXHAUSTIVE_SHARD"
                   if not failures else "EXHAUSTIVE_SHARD_FIT_FAILURES"),
        "n_assigned": expected, "n_success": len(rows),
        "n_failed": len(failures), "results": rows, "failures": failures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.exists():
        raise FileExistsError(temp)
    try:
        temp.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{era}/m{mass} shard {index}/{shards}: "
          f"{len(rows)}/{expected} fits; failures={len(failures)}", flush=True)
    print(f"Saved {path}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--era", choices=("run2", "run3"), required=True)
    p.add_argument("--mass", type=int, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--mu-max", type=float, default=1e5)
    args = p.parse_args()
    result = run_shard(args.output_root, args.era, args.mass,
                       args.shard_index, args.shards, args.mu_max)
    if result["n_failed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
