"""Frozen prompt-only exhaustive-scan inputs and deterministic fit inventory.

Do not modify source scored ROOT trees, BDT models, initial SR candidate JSON,
or earlier top-three CLs results. This is exclusively a follow-up optimizer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

from .aggregate import validate_comparison
from .prompt_likelihood import ARCHITECTURES, validate_template
from .scan import validate_existing_screen


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(root, era, mass):
    root = Path(root)
    folder = root / "sr_optimization" / era / f"m{mass}"
    screen_path = folder / "prompt_candidate_screen.json"
    shortlist_path = folder / "prompt_pyhf_top3.json"
    # Validate both frozen inputs, including the existing short-three closure.
    baseline_info = validate_comparison(root, era, mass, 3)
    screen = validate_existing_screen(screen_path, era, mass)
    baseline = json.loads(shortlist_path.read_text())
    counts = {}
    for channel in ("resolved", "boosted", "boosted_1tag", "boosted_2tag"):
        rows = screen["candidates"][channel]
        ranked = baseline["singles"][channel]
        counts[channel] = len(rows)
        if not rows:
            raise ValueError(f"{era}/m{mass}: no valid candidates in {channel}")
        if len(ranked) != len(rows):
            raise ValueError(f"{era}/m{mass}/{channel}: single-channel list incomplete")
        indices = [int(r["index"]) for r in ranked]
        if sorted(indices) != list(range(len(rows))):
            raise ValueError(f"{era}/m{mass}/{channel}: invalid ranked candidate indices")
        for record in ranked:
            cand = rows[int(record["index"])]
            validate_template(cand)
            if (record["channel"] != channel
                    or not math.isclose(float(record["score_threshold"]),
                                        float(cand["score_threshold"]), abs_tol=1e-12)
                    or not math.isclose(float(record["efficiency_target"]),
                                        float(cand["efficiency_target"]), abs_tol=1e-12)
                    or not (math.isfinite(float(record["expected_mu95"]))
                            and float(record["expected_mu95"]) > 0)):
                raise ValueError(f"{era}/m{mass}/{channel}: stale single-channel candidate")
    totals = {
        name: math.prod(counts[ch] for ch in channels)
        for name, channels in ARCHITECTURES.items()
    }
    if not all(n > 0 for n in totals.values()):
        raise ValueError("empty exhaustive architecture")
    config_path = Path(__file__).with_name("config.yaml")
    identity = {
        "schema_version": 1,
        "era": era, "mass_GeV": mass,
        "screen_sha256": sha256(screen_path),
        "shortlist_sha256": sha256(shortlist_path),
        "optimizer_config_sha256": sha256(config_path),
        "algorithm": "prompt_only_exhaustive_grid_v1",
        "architecture_order": list(ARCHITECTURES),
        "architecture_sizes": totals,
        "total_combinations": sum(totals.values()),
    }
    return screen, baseline, baseline_info, identity


def iter_combinations(screen):
    """Enumerate both independent likelihood architectures without overlap."""
    offset = 0
    for name, channels in ARCHITECTURES.items():
        iterables = [range(len(screen["candidates"][ch])) for ch in channels]
        for local_index, indices in enumerate(itertools.product(*iterables)):
            yield {
                "index": offset + local_index,
                "architecture": name,
                "candidate_indices": dict(zip(channels, indices)),
            }
        offset += math.prod(len(x) for x in iterables)


def assigned_to_shard(screen, shard_index, shards):
    if shards <= 0 or not 0 <= shard_index < shards:
        raise ValueError("invalid shard assignment")
    return (
        task for task in iter_combinations(screen)
        if task["index"] % shards == shard_index
    )


def shard_path(root, era, mass, shard_index, shards):
    return (Path(root) / "sr_optimization" / era / f"m{mass}"
            / "prompt_exhaustive"
            / f"shard_{shard_index:03d}_of_{shards:03d}.json")


def merged_path(root, era, mass):
    return (Path(root) / "sr_optimization" / era / f"m{mass}"
            / "prompt_pyhf_exhaustive.json")
