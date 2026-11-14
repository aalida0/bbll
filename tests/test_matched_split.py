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
                            [0., 500., 1000., 2000.])
    assert out["status"] == "NOT_COMPARABLE_UNDER_FROZEN_GUARDS"
    assert len(out["edges"]) - 1 == 2
    assert any("boosted_2tag" in s for s in out["reasons"])
