"""Fake-lepton MC-closure numerical gates (no real-data claims)."""

import numpy as np
import pytest

from fake_backgrounds.fakes import (
    calibrate_transfer, weighted_pt_bins, predict_os_fake,
    compare_independent_closure,
)


def test_transfer_factor_signed_yields_neff_and_statistical_variance():
    edges = [10, 20, 30]
    os_pt = np.array([15]*20 + [25]*40, dtype=float)
    ss_pt = np.array([15]*40 + [25]*20, dtype=float)
    os_w = np.ones(len(os_pt))
    ss_w = np.ones(len(ss_pt))
    result = calibrate_transfer(os_pt, os_w, ss_pt, ss_w,
                                edges=edges, minimum_neff=10)
    assert result["transfer_factors"] == pytest.approx([.5, 2])
    assert result["mc_os_neff"] == pytest.approx([20, 40])
    assert result["mc_ss_neff"] == pytest.approx([40, 20])
    assert result["transfer_variances"] == pytest.approx([
        20/40**2 + 20**2*40/40**4,
        40/20**2 + 40**2*20/20**4,
    ])
    p, p2 = weighted_pt_bins([10, 20, 30], [1, -2, 3], edges)
    assert p.tolist() == [1, 1]
    assert p2.tolist() == [1, 13]


def test_transfer_refuses_bad_pt_range_or_low_mcstat():
    with pytest.raises(ValueError, match="outside"):
        calibrate_transfer([9.9]*20, [1]*20, [15]*20,
                           [1]*20, edges=[10, 20, 30])
    with pytest.raises(ValueError, match="nonpositive"):
        calibrate_transfer([15]*20, [1]*20,
                           [15]*20, [1]*20, edges=[10, 20, 30])
    with pytest.raises(ValueError, match="effective statistics"):
        calibrate_transfer(
            [15]*10 + [25]*10, [1]*20,
            [15]*10 + [25]*9, [1]*19,
            edges=[10, 20, 30], minimum_neff=10
        )
    with pytest.raises(ValueError, match="nonpositive"):
        calibrate_transfer(
            [15]*20 + [25]*20, [1]*20 + [-1]*20,
            [15]*20 + [25]*20, [1]*40,
            edges=[10, 20, 30]
        )


def test_simulated_ss_prompt_subtraction_and_correlated_mt_template():
    transfer = {
        "transfer_factors": [.5, 2.0],
        "transfer_variances": [.01, .04]
    }
    fake_ss = np.array([[20., 10.], [5., 15.]])
    prompt_ss = np.array([[5., 2.], [1., 3.]])
    pseudo_ss = fake_ss + prompt_ss
    result = predict_os_fake(
        transfer,
        pseudo_ss,
        prompt_ss,
        pseudo_ss,
        prompt_ss,
    )
    prediction = np.array(result["fake_mt_bins"])
    np.testing.assert_allclose(prediction, [20, 35])
    cov = np.array(result["fake_mt_covariance"])
    np.testing.assert_allclose(
        cov[0, 1],
        .01*20*10 + .04*5*15,
    )
    assert cov[1, 0] == pytest.approx(cov[0, 1])
    assert np.all(np.linalg.eigvalsh(cov) >= 0)

    closure = compare_independent_closure(
        result, truth_os_sumw=[20, 35], truth_os_sumw2=[10, 10]
    )
    assert closure["closure_ratio_integral"] == pytest.approx(1.0)
    assert closure["closure_pull_per_mt_bin"] == pytest.approx([0, 0])
    assert closure["status"].endswith("NOT_NOMINAL")


def test_negative_ss_subtraction_is_never_clipped():
    tf = {"transfer_factors": [1.], "transfer_variances": [0.]}
    with pytest.raises(ValueError, match="negative SS"):
        predict_os_fake(tf, [[1., 2.]], [[0., 3.]],
                        [[1., 2.]], [[0., 3.]])
    with pytest.raises(ValueError, match="number of lepton-pT"):
        predict_os_fake(tf, [[1.], [1.]], [[0.], [0.]],
                        [[1.], [1.]], [[0.], [0.]])


def test_schema_audit_detects_os_only_selected_output(tmp_path):
    uproot = pytest.importorskip("uproot")
    from fake_backgrounds.audit import audit_file

    path = tmp_path / "selected.root"
    with uproot.recreate(path) as fout:
        fout["events"] = {
            "event_uid": np.array([1, 2], dtype="u8"),
            "category3": np.array([1, 2], dtype="i1"),
            "mt_hh_mH125": np.array([100, 300], dtype="f8"),
            "physics_weight": np.array([2, -1], dtype="f8"),
        }
    result = audit_file(path)
    assert result["status"] == "BLOCKED_MISSING_CANONICAL_CHARGE_TRUTH_FIELDS"
    assert "l1_charge" in result["missing_required_columns"]
    assert "is_prompt_lepton2" in result["missing_required_columns"]
    assert result["is_selected_or_oof_tree"] is True
    assert result["fake_sr_template_produced"] is False
