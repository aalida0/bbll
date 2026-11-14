"""Tests for prompt-only pyhf diagnostic, not publication inference."""

import json

import numpy as np
import pytest

pyhf = pytest.importorskip("pyhf")

from sr_optimization.prompt_likelihood import (
    workspace_for, validate_template, expected_mu95,
)
from sr_optimization.compare_architectures import compare


def toy(channel, scale=1.0, efficiency=0.8):
    return {
        "channel": channel, "mass_GeV": 600,
        "efficiency_target": efficiency, "score_threshold": 0.5,
        "edges_GeV": [0.0, 800.0, 1600.0],
        "signal_bins": [4.0 * scale, 2.0 * scale],
        "signal_sumw2_bins": [1.0, 1.0],
        "background_bins": [25.0, 30.0],
        "background_sumw2_bins": [5.0, 6.0],
        "background_yield": 55.0,
        "signal_yield": 6.0 * scale,
        "n_mt_bins": 2, "min_background_neff": 125.0,
    }


def test_pyhf_prompt_workspace_excludes_overlap():
    with pytest.raises(ValueError, match="overlap"):
        workspace_for([toy("boosted"), toy("boosted_1tag")])
    with pytest.raises(ValueError, match="duplicate"):
        workspace_for([toy("resolved"), toy("resolved")])
    ws = workspace_for([toy("resolved"), toy("boosted")])
    assert ws.model().config.poi_name == "mu"


def test_pyhf_background_statistical_safeguards():
    row = toy("resolved")
    assert len(validate_template(row)[0]) == 2
    row["background_bins"] = [-25.0, 30.0]
    with pytest.raises(ValueError, match="statistically unsafe"):
        validate_template(row)
    row = toy("resolved")
    row["background_sumw2_bins"] = [5000.0, 6000.0]
    with pytest.raises(ValueError, match="statistically unsafe"):
        validate_template(row)


def test_pyhf_asimov_expected_limit_is_finite():
    limit, solver = expected_mu95([toy("resolved")], mu_max=1000)
    assert np.isfinite(limit) and 0 < limit < 1000
    assert solver in {"pyhf_upper_limit", "median_cls_bisection"}
    better, _ = expected_mu95([toy("resolved", scale=2.0)], mu_max=1000)
    assert better < limit


def test_comparison_reports_architectures_and_does_not_double_count(
        tmp_path, monkeypatch):
    import sr_optimization.compare_architectures as comp
    from sr_optimization.scan import CFG

    src = (tmp_path / "sr_optimization" / "run3" / "m600"
           / "prompt_candidate_screen.json")
    src.parent.mkdir(parents=True)
    all_channels = ("resolved", "boosted", "boosted_1tag", "boosted_2tag")
    src.write_text(json.dumps({
        "schema_version": 1, "status": "PROMPT_ONLY_SCREEN_NOT_FINAL",
        "era": "run3", "mass_GeV": 600,
        "fake_backgrounds_included": False,
        "systematic_uncertainties_included": False,
        "expected_limits_computed": False,
        "minimum_background_neff_per_bin":
            CFG["minimum_background_neff_per_bin"],
        "minimum_mt_bins": CFG["minimum_mt_bins"],
        "candidates": {
            channel: [
                toy(channel, scale=1 + index / 100, efficiency=eff)
                for index, eff in enumerate(CFG["signal_efficiency_grids"][channel])
            ] for channel in all_channels
        },
        "rejected": {channel: {} for channel in all_channels},
    }))
    calls = []
    def fake_expected(rows, mu_max):
        group = tuple(row["channel"] for row in rows)
        calls.append(group)
        return 12. / sum(row["signal_yield"] for row in rows), "mock"
    monkeypatch.setattr(comp, "expected_mu95", fake_expected)
    result = compare(tmp_path, "run3", 600, shortlist=2)
    assert result["status"] == "COMPLETE_PROMPT_ONLY_DIAGNOSTIC"
    assert result["architectures"]["two_channel"]["n_attempted"] == 4
    assert result["architectures"]["three_channel"]["n_attempted"] == 8
    assert all(not (
        "boosted" in group and
        ("boosted_1tag" in group or "boosted_2tag" in group))
        for group in calls)
    assert result["two_over_three_expected_mu95_ratio"] is not None
    with pytest.raises(FileExistsError):
        compare(tmp_path, "run3", 600, shortlist=2)


def test_single_bin_cut_and_count_is_opt_in_and_keeps_neff_guard():
    # A single boosted template is valid only for the matched control.
    row = toy("boosted_2tag")
    row.update({
        "edges_GeV": [0.0, 1600.0],
        "signal_bins": [6.0],
        "signal_sumw2_bins": [2.0],
        "background_bins": [55.0],
        "background_sumw2_bins": [11.0],
        "n_mt_bins": 1, "min_background_neff": 275.0,
    })
    with pytest.raises(ValueError, match="statistically unsafe"):
        validate_template(row)
    validate_template(row, min_mt_bins=1)
    limit, solver = expected_mu95([row], mu_max=1000, min_mt_bins=1)
    assert 0 < limit < 1000
    assert solver in {"pyhf_upper_limit", "median_cls_bisection"}
    # Even with 1 bin enabled, an insufficient signed-background N_eff
    # is a hard error, never a justification to accept a cut-and-count bin.
    row["background_sumw2_bins"] = [1000.0]
    with pytest.raises(ValueError, match="statistically unsafe"):
        validate_template(row, min_mt_bins=1)
