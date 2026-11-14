"""Unit tests for prompt-only SR candidate screening and MC-stat safeguards."""

import numpy as np
import pytest

from bbll.config import load_stage_config
from sr_optimization.scan import (
    adaptive_edges,
    balanced_signal_for_cut,
    candidate,
    channel_mask,
    histogram,
    neff,
    positive_weighted_quantile,
)


def test_sr_configuration_covers_two_and_three_categories():
    c = load_stage_config("sr_optimization")
    assert c["minimum_background_neff_per_bin"] == 10
    assert c["minimum_mt_bins"] == 2
    assert set(c["signal_efficiency_grids"]) == {
        "resolved", "boosted", "boosted_1tag", "boosted_2tag",
    }
    assert 0.10 in c["signal_efficiency_grids"]["boosted_1tag"]
    assert c["master_mt_edges_GeV"][0] == 0
    assert c["master_mt_edges_GeV"][-1] == 10000


def test_weighted_quantile_and_signed_background_neff():
    np.testing.assert_allclose(neff([10, -2, 0], [20, 2, 0]), [5, 2, 0])
    assert positive_weighted_quantile(
        np.array([0.2, 0.5, 0.9]), .5, np.ones(3),
    ) == pytest.approx(.35)
    with pytest.raises(ValueError):
        positive_weighted_quantile(np.array([1.]), .5, np.array([-1.]))


def test_category_masks_exclusive_and_combined():
    data = {"category3": np.array([0, 1, 2, 2, 1], dtype=np.int8)}
    resolved = channel_mask(data, "resolved")
    boosted = channel_mask(data, "boosted")
    one = channel_mask(data, "boosted_1tag")
    two = channel_mask(data, "boosted_2tag")
    np.testing.assert_array_equal(resolved | boosted, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(one | two, boosted)
    assert not np.any(one & two)
    assert not np.any(resolved & boosted)


def test_adaptive_binning_requires_positive_yields_and_neff():
    mt = np.concatenate([
        np.full(30, 250.), np.full(30, 700.), np.full(30, 1300.),
    ])
    weights = np.ones(len(mt))
    edges = adaptive_edges(mt, weights, threshold=10)
    assert edges is not None
    assert len(edges) >= 3
    sw, sw2 = histogram(mt, weights, edges)
    assert np.all(sw > 0)
    assert np.all(neff(sw, sw2) >= 10)
    sparse = adaptive_edges([200., 300.], [1., -1.], threshold=10)
    assert sparse is None
    # Underflow and overflow are clipped to the first and final bins.
    sums, _ = histogram([-100., 20000.], [1., 2.], [0., 1000., 10000.])
    np.testing.assert_array_equal(sums, [1, 2])


def test_decay_balanced_threshold_and_diagnostic_candidate():
    # Need three signal decays; 30 events each and 120 backgrounds per mT band.
    signal_n = 90
    bkg_n = 360
    masses = np.concatenate([np.full(signal_n, 600), np.zeros(bkg_n)]).astype(np.int16)
    decays = np.concatenate([np.repeat([1, 2, 3], 30), np.full(bkg_n, -1)])
    is_signal = np.array([True] * signal_n + [False] * bkg_n)
    x = np.linspace(.1, .9, signal_n)
    bkg_score = np.tile(np.linspace(.1, .9, 120), 3)
    score = np.concatenate([x, bkg_score])
    mt = np.concatenate([
        np.tile(np.array([300., 800., 1600.]), 30),
        np.repeat([300., 800., 1600.], 120),
    ])
    w = np.ones(len(score))
    data = {
        "ml_score": score, "mt_hh_mH125": mt,
        "physics_weight": w,
        "is_signal": is_signal,
        "signal_mass_GeV": masses,
        "signal_decay_id": decays,
        "category3": np.zeros(len(score), dtype=np.int8),
    }
    sigw, sig = balanced_signal_for_cut(data, 600, "resolved")
    assert np.isclose(sigw[sig].sum(), 1)
    assert all(np.isclose(sigw[sig & (decays == d)].sum(), 1/3)
               for d in (1, 2, 3))
    row, status = candidate(data, "run2", "resolved", 600, .8)
    assert status == "ok"
    assert row["n_mt_bins"] >= 2
    assert row["min_background_neff"] >= 10
    assert row["signal_yield"] > 0
    assert row["background_yield"] > 0
    assert row["score_threshold"] < 1



def test_all_masses_loads_oof_once_and_resumes(tmp_path, monkeypatch):
    import json
    import sr_optimization.scan as scanner

    load_calls = []
    screen_calls = []
    def fake_load(root, era, topology):
        load_calls.append((era, topology))
        return {"dummy": topology}

    def fake_screen(root, era, mass, output, by_topology=None):
        assert by_topology == {
            "resolved": {"dummy": "resolved"},
            "boosted": {"dummy": "boosted"},
        }
        screen_calls.append(mass)
        valid = {
            channel: [
                {
                    "channel": channel,
                    "mass_GeV": mass,
                    "n_mt_bins": 2,
                    "min_background_neff": 12.0,
                    "efficiency_target": eff,
                }
                for eff in scanner.CFG["signal_efficiency_grids"][channel]
            ]
            for channel in scanner.CHANNEL_TO_TOPOLOGY
        }
        doc = {
            "schema_version": 1,
            "status": "PROMPT_ONLY_SCREEN_NOT_FINAL",
            "era": era, "mass_GeV": mass,
            "fake_backgrounds_included": False,
            "systematic_uncertainties_included": False,
            "expected_limits_computed": False,
            "minimum_background_neff_per_bin":
                scanner.CFG["minimum_background_neff_per_bin"],
            "minimum_mt_bins": scanner.CFG["minimum_mt_bins"],
            "candidates": valid,
            "rejected": {name: {} for name in scanner.CHANNEL_TO_TOPOLOGY},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(doc))
        return doc

    monkeypatch.setattr(scanner, "load_scores", fake_load)
    monkeypatch.setattr(scanner, "screen", fake_screen)
    summary = scanner.screen_all(tmp_path, "run3", masses=[600, 700])
    assert summary["n_masses"] == 2
    assert screen_calls == [600, 700]
    assert load_calls == [("run3", "resolved"), ("run3", "boosted")]
    assert (tmp_path / "sr_optimization" / "run3" /
            "prompt_screen_campaign.json").exists()

    with pytest.raises(FileExistsError, match="resume"):
        scanner.screen_all(tmp_path, "run3", masses=[600, 700])

    load_calls.clear()
    screen_calls.clear()
    reused = scanner.screen_all(
        tmp_path, "run3", resume=True, masses=[600, 700]
    )
    assert reused["masses"] == summary["masses"]
    assert not load_calls and not screen_calls

    broken = (tmp_path / "sr_optimization" / "run3" / "m600" /
              "prompt_candidate_screen.json")
    saved = json.loads(broken.read_text())
    saved["candidates"]["resolved"].pop()
    broken.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="incomplete threshold grid"):
        scanner.screen_all(tmp_path, "run3", resume=True, masses=[600, 700])
