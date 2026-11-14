import numpy as np
import pytest

from bbll.selection import (
    BRANCHES, RESOLVED, BOOSTED, CAT2, CAT3, categories, reduce_sample
)


def test_features_frozen_and_no_fit_variable_or_tag_input():
    assert len(RESOLVED) == 20
    assert len(BOOSTED) == 25
    assert "mt_hh_mH125" not in RESOLVED + BOOSTED
    assert "n_tags_inside_largeR" not in RESOLVED + BOOSTED


def test_exclusive_boosted_first_and_exact_two_three_category_closure():
    # 0: both candidates, one tag => boosted 1tag wins
    # 1: both candidates, two tags => boosted 2tag wins
    # 2: no boosted tags, two resolved tags => resolved
    # 3: trimmed largeR invalid, resolved succeeds => resolved
    # 4: common preselection fails; 5: no Higgs candidate
    # 6: >=3 tags => boosted 2tag
    a = {
        "pass_common_preselection": np.array([1, 1, 1, 1, 0, 1, 1]),
        "largeR_candidate_valid": np.array([1, 1, 1, 1, 1, 0, 1]),
        "largeR_trim_valid": np.array([1, 1, 1, 0, 1, 0, 1]),
        "n_tags_inside_largeR": np.array([1, 2, 0, 2, 1, 0, 3]),
        "resolved_candidate_valid": np.array([1, 1, 1, 1, 1, 0, 1]),
        "resolved_b1_tag": np.ones(7, dtype=np.int8),
        "resolved_b2_tag": np.ones(7, dtype=np.int8),
    }
    three, two = categories(a)
    assert three.tolist() == [1, 2, 0, 0, -1, -1, 2]
    assert two.tolist() == [1, 1, 0, 0, -1, -1, 1]
    assert np.array_equal(three >= 0, two >= 0)
    assert np.count_nonzero(three == CAT3["resolved"]) == 2
    assert np.count_nonzero(two == CAT2["boosted"]) == (
        np.count_nonzero(three == CAT3["boosted_1tag"])
        + np.count_nonzero(three == CAT3["boosted_2tag"])
    )


def test_root_reduction_retains_signed_weights_and_dual_categories(tmp_path):
    uproot = pytest.importorskip("uproot")

    n = 7
    data = {name: np.arange(n, dtype=np.float32) for name in BRANCHES}
    data.update({
        "event_uid": np.arange(n, dtype=np.uint64),
        "sample_class": np.zeros(n, dtype=np.int8),
        "is_signal": np.ones(n, dtype=np.bool_),
        "signal_mass_GeV": np.full(n, 600, dtype=np.int16),
        "signal_decay_id": np.zeros(n, dtype=np.int8),
        "n_tags_inside_largeR": np.array([1, 2, 0, 2, 1, 0, 3], dtype=np.int8),
        "pass_common_preselection": np.array([1, 1, 1, 1, 0, 1, 1], dtype=np.int8),
        "largeR_candidate_valid": np.array([1, 1, 1, 1, 1, 0, 1], dtype=np.int8),
        "largeR_trim_valid": np.array([1, 1, 1, 0, 1, 0, 1], dtype=np.int8),
        "resolved_candidate_valid": np.array([1, 1, 1, 1, 1, 0, 1], dtype=np.int8),
        "resolved_b1_tag": np.ones(n, dtype=np.int8),
        "resolved_b2_tag": np.ones(n, dtype=np.int8),
        "min_dR_lepton_bjet_valid": np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.int8),
    })
    weights = np.array([1., -2., 3., 4., 5., 6., 7.], dtype=np.float64)
    source = tmp_path / "X600_run3.root"
    sidecar = tmp_path / "X600_run3.weights.root"
    output = tmp_path / "reduced" / "X600_run3.root"

    with uproot.recreate(source) as f:
        f.mktree(
            "Analysis_NOSYS", {key: value.dtype for key, value in data.items()}
        ).extend(data)
    with uproot.recreate(sidecar) as f:
        f.mktree("Weights", {"weight_nominal": weights.dtype}).extend(
            {"weight_nominal": weights}
        )

    report = reduce_sample(source, sidecar, output, chunk_size=3)
    assert report["counts"]["resolved"] == 2
    assert report["counts"]["boosted_1tag"] == 1
    assert report["counts"]["boosted_2tag"] == 2
    assert report["selected"] == 5
    with uproot.open(output) as f:
        a = f["events"].arrays(library="np")
    assert a["category3"].tolist() == [1, 2, 0, 0, 2]
    assert a["category2"].tolist() == [1, 1, 0, 0, 1]
    assert a["physics_weight"].tolist() == [1., -2., 3., 4., 7.]
    assert a["train_weight_base"].tolist() == [1., 2., 3., 4., 7.]
    assert np.isnan(a["min_dR_lepton_bjet"][3])
    assert a["event_uid"].tolist() == [0, 1, 2, 3, 6]
