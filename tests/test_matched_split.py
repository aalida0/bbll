"""Regression gates for same-cut/same-bin inclusive-vs-split control."""

import numpy as np
import pytest

from sr_optimization.matched_split import (
    boosted_histograms, matched_templates, merge_adjacent_bins,
)


def input_events(tag1=(40, 40, 40), tag2=(5, 5, 40)):
    mt_centers = (250., 750., 1500.)
    data = {key: [] for key in (
        "category3", "is_signal", "signal_mass_GeV", "ml_score",
        "mt_hh_mH125", "physics_weight",
    )}

    def append(tag, signal, mass, mt, weight):
        data["category3"].append(tag)
        data["is_signal"].append(signal)
        data["signal_mass_GeV"].append(mass)
        data["ml_score"].append(0.9 if signal else 0.7)
        data["mt_hh_mH125"].append(mt)
        data["physics_weight"].append(weight)

    for tag, counts in ((1, tag1), (2, tag2)):
        for mt, count in zip(mt_centers, counts):
            for _ in range(count):
                append(tag, False, 0, mt, 1.0)
            for _ in range(10):
                append(tag, True, 600, mt, 1.0)

    return {
        name: np.asarray(values)
        for name, values in data.items()
    }


def test_inclusive_is_exact_signed_sum_of_two_exclusive_tags():
    data = input_events(tag1=(40, 40, 40), tag2=(40, 40, 40))
    edges = [0., 500., 1000., 2000.]
    original = boosted_histograms(data, "run2", 600, 0.5, edges)
    for field in ("s", "s2", "b", "b2"):
        np.testing.assert_allclose(
            original["boosted"][field],
            original["boosted_1tag"][field] + original["boosted_2tag"][field]
        )
    assert original["boosted"]["n_background"] == 240
    out = matched_templates(data, "run2", 600, 0.5, edges)
    assert out["status"] == "MATCHED_TEMPLATES_VALID"
    assert out["coarsening_steps"] == []
    np.testing.assert_allclose(out["edges"], edges)


def test_common_binning_coarsens_both_tags_together():
    data = input_events()
    edges = [0., 500., 1000., 2000.]
    original = boosted_histograms(data, "run2", 600, 0.5, edges)
    baseline = {
        "signal_bins": original["boosted"]["s"].tolist(),
        "background_bins": original["boosted"]["b"].tolist(),
        "background_sumw2_bins": original["boosted"]["b2"].tolist(),
    }
    merged = matched_templates(
        data, "run2", 600, 0.5, edges, reference_boosted=baseline
    )
    assert merged["status"] == "MATCHED_TEMPLATES_VALID"
    assert len(merged["edges"]) == 3
    assert merged["coarsening_steps"] == [500.]
    for ch in ("boosted_1tag", "boosted_2tag"):
        row = merged["rows"][ch]
        assert np.all(row["b"] > 0)
        assert np.all(row["b"]**2 / row["b2"] >= 10)

    analytical = merge_adjacent_bins(original, 1)
    for ch in original:
        np.testing.assert_allclose(analytical[ch]["b"], merged["rows"][ch]["b"])

    baseline["background_bins"][0] += 1
    with pytest.raises(ValueError, match="template mismatch"):
        matched_templates(data, "run2", 600, 0.5, edges,
                          reference_boosted=baseline)


def test_refuses_matched_comparison_if_split_cannot_retain_two_bins():
    data = input_events(tag1=(40, 40, 40), tag2=(3, 3, 3))
    out = matched_templates(data, "run2", 600, 0.5,
                            [0., 500., 1000., 2000.],
                            min_mt_bins=2)
    assert out["status"] == "NOT_COMPARABLE_UNDER_CONFIGURED_GUARDS"
    assert len(out["edges"]) - 1 == 2
    assert any("boosted_2tag" in s for s in out["reasons"])



def test_matched_control_recomputes_both_architectures_with_identical_bins(
        tmp_path, monkeypatch):
    import json
    import sr_optimization.matched_split_control as control

    era, mass, shortlist = "run3", 600, 3
    folder = tmp_path / "sr_optimization" / era / f"m{mass}"
    folder.mkdir(parents=True)
    screen_path = folder / "prompt_candidate_screen.json"
    screen_path.write_text("{}")
    comparison = folder / "prompt_pyhf_top3.json"
    comparison.write_text(json.dumps({
        "architectures": {"two_channel": {"best": {"candidates": {
            "resolved": {"index": 0, "score_threshold": 0.5},
            "boosted": {"index": 0, "score_threshold": 0.5},
        }}}}
    }))
    data = input_events(tag1=(40, 40, 40), tag2=(5, 5, 40))
    edges = [0., 500., 1000., 2000.]
    boosted = boosted_histograms(data, era, mass, 0.5, edges)["boosted"]
    b_row = {
        "channel": "boosted", "score_threshold": 0.5,
        "edges_GeV": edges,
        "signal_bins": boosted["s"].tolist(),
        "signal_sumw2_bins": boosted["s2"].tolist(),
        "background_bins": boosted["b"].tolist(),
        "background_sumw2_bins": boosted["b2"].tolist(),
        "signal_yield": float(np.sum(boosted["s"])),
        "background_yield": float(np.sum(boosted["b"])),
        "n_mt_bins": 3, "min_background_neff": 45.0,
    }
    resolved = dict(b_row, channel="resolved")

    monkeypatch.setattr(control, "validate_existing_screen",
                        lambda src, era, mass: {
                            "candidates": {
                                "resolved": [resolved],
                                "boosted": [b_row],
                            },
                        })
    monkeypatch.setattr(control, "validate_comparison",
                        lambda root, era, mass, shortlist: {
                            "file": str(comparison),
                            "sha256": "unit_fixture",
                            "two_channel_mu95": 10.0,
                        })
    monkeypatch.setattr(control, "load_scores",
                        lambda root, era, topology: data)
    calls = []

    def fake_limit(rows, mu_max, min_mt_bins=None):
        channels = tuple(r["channel"] for r in rows)
        calls.append((channels, [r["edges_GeV"] for r in rows]))
        return (10.0 if "boosted" in channels else 8.0), "mock"

    monkeypatch.setattr(control, "expected_mu95", fake_limit)
    result = control.compare_matched(tmp_path, era, mass)
    assert result["status"] == "PASS_MATCHED_SPLIT_CONTROL"
    assert result["common_boosted_mt_edges_GeV"] == [0., 1000., 2000.]
    assert result["coarsened_boundaries_GeV"] == [500.]
    assert result["percent_limit_reduction"] == pytest.approx(20.0)
    assert [c[0] for c in calls] == [
        ("resolved", "boosted"),
        ("resolved", "boosted"),
        ("resolved", "boosted_1tag", "boosted_2tag"),
    ]
    assert calls[-2][1][1] == calls[-1][1][1] == calls[-1][1][2]
    with pytest.raises(FileExistsError):
        control.compare_matched(tmp_path, era, mass)


def test_one_bin_yaml_setting_and_previous_two_bin_guard():
    from sr_optimization.matched_split import (
        configured_min_bins, invalid_template_reasons,
    )
    from bbll.config import load_stage_config

    cfg = load_stage_config("sr_optimization")
    assert cfg["minimum_mt_bins"] == 2  # frozen Stage-3 screen remains untouched
    assert cfg["matched_split_control"]["minimum_mt_bins"] == 2
    assert configured_min_bins() == 2

    data = input_events(tag1=(40, 40, 40), tag2=(3, 3, 5))
    edges = [0., 500., 1000., 2000.]
    two_bin = matched_templates(
        data, "run2", 600, .5, edges, min_mt_bins=2
    )
    assert two_bin["status"] == "NOT_COMPARABLE_UNDER_CONFIGURED_GUARDS"
    assert len(two_bin["edges"]) - 1 == 2

    one_bin = matched_templates(data, "run2", 600, .5, edges,
                                min_mt_bins=1)
    assert one_bin["status"] == "MATCHED_TEMPLATES_VALID"
    assert len(one_bin["edges"]) - 1 == 1
    assert one_bin["min_mt_bins"] == 1
    assert not invalid_template_reasons(
        one_bin["rows"], one_bin["edges"], min_bins=1
    )
    for ch in ("boosted", "boosted_1tag", "boosted_2tag"):
        r = one_bin["rows"][ch]
        assert r["b"][0] > 0
        assert r["b"][0]**2 / r["b2"][0] >= 10


def test_one_bin_neff_is_never_waived():
    data = input_events(tag1=(40, 40, 40), tag2=(3, 3, 3))
    out = matched_templates(
        data, "run2", 600, .5, [0., 500., 1000., 2000.],
        min_mt_bins=1
    )
    assert out["status"] == "NOT_COMPARABLE_UNDER_CONFIGURED_GUARDS"
    assert len(out["edges"]) - 1 == 1
    assert "boosted_2tag:low_background_neff" in out["reasons"]


def test_min_mt_bins_parameter_validation():
    from sr_optimization.matched_split import select_min_bins
    for invalid in (0, 3, 1.0, True, "1"):
        with pytest.raises(ValueError, match="1 or 2"):
            select_min_bins(invalid)
