"""Exhaustive prompt-only CLs sharding, merge and BDT-score extraction gates."""

import json
import math
from pathlib import Path

import pytest

import sr_optimization.exhaustive_shard as shard
import sr_optimization.exhaustive_merge as merge
import sr_optimization.exhaustive_submit as submit
import sr_optimization.exhaustive_campaign as campaign

from sr_optimization.exhaustive_inputs import (
    iter_combinations, assigned_to_shard, shard_path, merged_path,
)
from sr_optimization.prompt_likelihood import ARCHITECTURES


def fake_fixture(root):
    """Two candidates per channel: 4 + 8 = 12 exact combinations."""
    screen = {"candidates": {}}
    for channel in ("resolved", "boosted", "boosted_1tag", "boosted_2tag"):
        screen["candidates"][channel] = [
            {
                "channel": channel,
                "efficiency_target": .95 - idx * .1,
                "score_threshold": .25 + idx * .2,
                "n_mt_bins": 2,
                "edges_GeV": [0, 500, 10000],
                "min_background_neff": 11.0,
                "signal_yield": 6.0 + idx,
                "background_yield": 30.0,
            }
            for idx in range(2)
        ]
    shortlist = {
        "architectures": {
            name: {"best": {"expected_mu95": 500.0}}
            for name in ARCHITECTURES
        }
    }
    file = root / "input_shortlist.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(shortlist))
    identity = {
        "schema_version": 1,
        "era": "run3",
        "mass_GeV": 600,
        "screen_sha256": "s" * 64,
        "shortlist_sha256": "t" * 64,
        "optimizer_config_sha256": "c" * 64,
        "algorithm": "prompt_only_exhaustive_grid_v1",
        "architecture_order": list(ARCHITECTURES),
        "architecture_sizes": {"two_channel": 4, "three_channel": 8},
        "total_combinations": 12,
    }
    return screen, shortlist, {"file": str(file)}, identity


def test_exhaustive_cartesian_product_covers_both_architectures_exactly(tmp_path):
    screen, _, _, _ = fake_fixture(tmp_path)
    tasks = list(iter_combinations(screen))
    assert len(tasks) == 12
    assert [x["index"] for x in tasks] == list(range(12))
    assert {x["architecture"] for x in tasks[:4]} == {"two_channel"}
    assert {x["architecture"] for x in tasks[4:]} == {"three_channel"}
    for nshards in (1, 2, 3, 5, 24):
        chunks = [
            list(assigned_to_shard(screen, i, nshards))
            for i in range(nshards)
        ]
        assert sorted(x["index"] for c in chunks for x in c) == list(range(12))


def test_exhaustive_shard_merge_finds_global_best_and_rejects_corruption(
        tmp_path, monkeypatch):
    inputs = fake_fixture(tmp_path)
    monkeypatch.setattr(shard, "load_inputs", lambda *args: inputs)
    monkeypatch.setattr(merge, "load_inputs", lambda *args: inputs)

    def expected_limit(candidates, mu_max=1e5):
        return 200.0 / sum(r["signal_yield"] for r in candidates), "fixture"
    monkeypatch.setattr(shard, "expected_mu95", expected_limit)

    for i in range(3):
        doc = shard.run_shard(tmp_path, "run3", 600, i, 3)
        assert doc["n_assigned"] == 4
        assert doc["n_failed"] == 0

    with pytest.raises(FileExistsError, match="overwrite"):
        shard.run_shard(tmp_path, "run3", 600, 0, 3)

    p = shard_path(tmp_path, "run3", 600, 1, 3)
    saved = json.loads(p.read_text())
    saved["screen_sha256"] = "stale"
    p.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="stale/wrong"):
        merge.merge_mass(tmp_path, "run3", 600, 3)

    saved["screen_sha256"] = inputs[3]["screen_sha256"]
    p.write_text(json.dumps(saved))
    merged = merge.merge_mass(tmp_path, "run3", 600, 3)
    assert merged["status"] == "COMPLETE_PROMPT_ONLY_EXHAUSTIVE"
    assert merged["n_combined_tested"] == 12
    assert merged["n_combined_failed"] == 0
    for name in ARCHITECTURES:
        assert merged["architectures"][name]["best_candidates"]
        assert merged["architectures"][name]["shortlist_to_exhaustive_ratio"] >= 1

    assert merged_path(tmp_path, "run3", 600).exists()
    with pytest.raises(FileExistsError):
        merge.merge_mass(tmp_path, "run3", 600, 3)


def test_partial_and_failed_shards_are_never_accepted(tmp_path, monkeypatch):
    inputs = fake_fixture(tmp_path)
    monkeypatch.setattr(shard, "load_inputs", lambda *args: inputs)
    monkeypatch.setattr(merge, "load_inputs", lambda *args: inputs)
    monkeypatch.setattr(shard, "expected_mu95",
                        lambda rows, mu_max: (4.0, "fixture"))
    shard.run_shard(tmp_path, "run3", 600, 0, 3)

    with pytest.raises(FileNotFoundError):
        merge.merge_mass(tmp_path, "run3", 600, 3)

    p = shard_path(tmp_path, "run3", 600, 0, 3)
    saved = json.loads(p.read_text())
    saved["n_success"] -= 1
    p.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="accounting"):
        merge.validate_shard(tmp_path, "run3", 600, 0, 3, inputs[3])
    saved["n_success"] += 1
    saved["status"] = "EXHAUSTIVE_SHARD_FIT_FAILURES"
    p.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="incomplete or failed"):
        merge.validate_shard(tmp_path, "run3", 600, 0, 3, inputs[3])


def test_deterministic_shard_plan_and_file_transfer_contract(tmp_path):
    assert submit.calculate_shards(1452) == 30
    assert submit.calculate_shards(1452, fits_per_shard=10) == 32
    assert submit.calculate_shards(1) == 1
    with pytest.raises(ValueError):
        submit.calculate_shards(100, fits_per_shard=0)

    repo = tmp_path / "repo"
    (repo / "sr_optimization").mkdir(parents=True)
    worker = repo / "sr_optimization" / "exhaustive_worker.sh"
    worker.write_text("#!/bin/bash\nexec echo testing\n")
    py = repo / "python"
    py.touch()
    data = tmp_path / "data"
    data.mkdir()
    spec = submit.build_submit(
        repo, py, data, repo / "condor" / "logs",
        [("run3", 600, 0, 3), ("run2", 3000, 1, 2)],
    )
    assert "should_transfer_files = YES" in spec
    assert "request_memory = 3GB" in spec
    assert "getenv = False" in spec
    assert "transfer_input_files = " in spec
    assert "run3 600 0 3" in spec
    assert "run2 3000 1 2" in spec
    assert "set -e" not in worker.read_text()
    assert "set -u" not in worker.read_text()
