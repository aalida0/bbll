from pathlib import Path

import numpy as np
import pytest

from preselection.campaign import run_campaign
from preselection.selection import BRANCHES


def _write_sample(root: Path, era: str, kind: str, stem: str):
    uproot = pytest.importorskip("uproot")
    folder = root / kind / era
    folder.mkdir(parents=True, exist_ok=True)
    n = 5
    a = {key: np.arange(n, dtype=np.float32) for key in BRANCHES}
    a.update({
        "event_uid": np.arange(n, dtype=np.uint64),
        "sample_class": np.zeros(n, dtype=np.int8),
        "is_signal": np.full(n, kind == "signal", dtype=np.bool_),
        "signal_mass_GeV": np.full(n, 600 if kind == "signal" else 0, dtype=np.int16),
        "signal_decay_id": np.zeros(n, dtype=np.int8),
        "n_tags_inside_largeR": np.array([1, 2, 0, 0, 3], dtype=np.int8),
        "pass_common_preselection": np.ones(n, dtype=np.int8),
        "largeR_candidate_valid": np.array([1, 1, 0, 0, 1], dtype=np.int8),
        "largeR_trim_valid": np.array([1, 1, 0, 0, 1], dtype=np.int8),
        "resolved_candidate_valid": np.ones(n, dtype=np.int8),
        "resolved_b1_tag": np.ones(n, dtype=np.int8),
        "resolved_b2_tag": np.ones(n, dtype=np.int8),
        "min_dR_lepton_bjet_valid": np.array([1, 1, 0, 1, 1], dtype=np.int8),
    })
    weights = np.array([1., -2., 3., 4., -5.], dtype=np.float64)
    with uproot.recreate(folder / f"{stem}.root") as f:
        f.mktree("Analysis_NOSYS", {k: v.dtype for k, v in a.items()}).extend(a)
    with uproot.recreate(folder / f"{stem}.weights.root") as f:
        f.mktree("Weights", {"weight_nominal": weights.dtype}).extend(
            {"weight_nominal": weights}
        )


def test_campaign_resume_and_old_selection_reference(tmp_path):
    uproot = pytest.importorskip("uproot")
    era = "run3"
    inventory = [("signal", "X600_run3"), ("background", "ttbar_dil_run3")]
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "selected"
    reference = tmp_path / "legacy"
    for kind, stem in inventory:
        _write_sample(inputs, era, kind, stem)

    first = run_campaign(
        input_root=inputs, output_root=outputs, era=era, expected=inventory
    )
    assert first["n_samples"] == 2
    assert first["counts"]["input"] == 10
    assert first["category2_counts"] == {"resolved": 4, "boosted": 6}
    assert first["category3_counts"] == {
        "resolved": 4, "boosted_1tag": 2, "boosted_2tag": 4
    }

    # Construct an exact frozen-reduction witness for the optional validation.
    for _, stem in inventory:
        with uproot.open(outputs / era / f"{stem}.root") as f:
            a = f["events"].arrays(library="np")
        for topology, mask in (
            ("resolved", a["category3"] == 0),
            ("boosted", a["category3"] >= 1),
        ):
            folder = reference / era / topology
            folder.mkdir(parents=True, exist_ok=True)
            selected = {k: v[mask] for k, v in a.items()}
            with uproot.recreate(folder / f"{stem}.root") as f:
                f.mktree("events", {k: v.dtype for k, v in selected.items()}).extend(
                    selected
                )

    second = run_campaign(
        input_root=inputs, output_root=outputs, era=era,
        expected=inventory, resume=True, reference_dir=reference
    )
    assert second["reference_checked"] is True
    assert second["category2_counts"] == first["category2_counts"]


def test_campaign_fails_missing_inputs_before_writing(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_campaign(
            input_root=tmp_path / "none",
            output_root=tmp_path / "outputs",
            era="run2",
            expected=[("signal", "X600_run2")],
        )
    assert not (tmp_path / "outputs").exists()
