"""Safe two-era prompt-only CLs submission and aggregation contract."""

import hashlib
import json

import pytest

from sr_optimization.aggregate import MASSES, aggregate, validate_comparison
from sr_optimization.submit import build_submit, pending


def seed_comparison(root, era, mass, shortlist=3, status="COMPLETE_PROMPT_ONLY_DIAGNOSTIC"):
    folder = root / "sr_optimization" / era / f"m{mass}"
    folder.mkdir(parents=True, exist_ok=True)
    screen = folder / "prompt_candidate_screen.json"
    screen.write_text('{"validated":"screen"}\n')
    two, three = 10.0, 8.0
    result = {
        "schema_version": 1, "status": status,
        "era": era, "mass_GeV": mass,
        "source_sha256": hashlib.sha256(screen.read_bytes()).hexdigest(),
        "shortlist_per_channel": shortlist,
        "fake_backgrounds_included": False,
        "systematic_uncertainties_included": False,
        "signal_MCstat_included": False,
        "fit_failures": [],
        "shortlist_exhaustive": False,
        "two_over_three_expected_mu95_ratio": two / three,
        "architectures": {
            "two_channel": {
                "status": "SHORTLIST_COMPLETE", "n_attempted": 9,
                "n_success": 9,
                "best": {"expected_mu95": two,
                         "candidates": {
                             "resolved": {}, "boosted": {}
                         }},
            },
            "three_channel": {
                "status": "SHORTLIST_COMPLETE", "n_attempted": 27,
                "n_success": 27,
                "best": {"expected_mu95": three,
                         "candidates": {
                             "resolved": {}, "boosted_1tag": {},
                             "boosted_2tag": {},
                         }},
            },
        },
    }
    path = folder / f"prompt_pyhf_top{shortlist}.json"
    path.write_text(json.dumps(result))
    return path


def test_completed_closure_and_reject_incomplete_results(tmp_path):
    path = seed_comparison(tmp_path, "run3", 600)
    report = validate_comparison(tmp_path, "run3", 600, 3)
    assert report["ratio_two_over_three"] == pytest.approx(1.25)
    assert report["percent_limit_reduction"] == pytest.approx(20)
    saved = json.loads(path.read_text())
    saved["status"] = "INCOMPLETE_PROMPT_ONLY_DIAGNOSTIC"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="incomplete"):
        validate_comparison(tmp_path, "run3", 600, 3)


def test_submits_only_missing_mass_points_and_transfers_launcher(tmp_path):
    root, repo, py = tmp_path / "data", tmp_path / "repo", tmp_path / "python"
    (repo / "sr_optimization").mkdir(parents=True)
    (repo / "sr_optimization" / "worker.sh").write_text("#!/bin/bash\n")
    py.touch()
    for mass in MASSES:
        folder = root / "sr_optimization" / "run3" / f"m{mass}"
        folder.mkdir(parents=True)
        (folder / "prompt_candidate_screen.json").write_text("{}")
    # Replace one fake screen with a matching completed-fit fixture.
    seed_comparison(root, "run3", 600)
    missing, reused = pending(root, ("run3",), 3)
    assert reused == [("run3", 600)]
    assert len(missing) == len(MASSES) - 1
    assert ("run3", 600) not in missing
    logdir = repo / "condor" / "logs"
    sub = build_submit(repo, py, root, logdir, missing, 3)
    assert "should_transfer_files = YES" in sub
    assert "transfer_input_files = " in sub
    assert "getenv = False" in sub
    assert "request_memory = 3GB" in sub
    assert "FileSystemDomain" not in sub
    assert "run3 700" in sub
    assert "run3 600" not in sub
    worker = (repo / "sr_optimization" / "worker.sh").read_text()
    assert "set -e" not in worker and "set -u" not in worker


def test_aggregate_requires_every_mass_and_preserves_ratios(tmp_path):
    seed_comparison(tmp_path, "run2", 600)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        aggregate(tmp_path, ("run2",), 3)
    for mass in MASSES:
        if mass != 600:
            seed_comparison(tmp_path, "run2", mass)
    result = aggregate(tmp_path, ("run2",), 3)
    assert len(result["results"]) == len(MASSES)
    assert all(x["ratio_two_over_three"] == pytest.approx(1.25)
               for x in result["results"])
    assert result["fake_backgrounds_included"] is False
    with pytest.raises(FileExistsError, match="existing summary"):
        aggregate(tmp_path, ("run2",), 3)
