"""Test Stage-2 BDT contract without any real production ntuples."""

import copy

import numpy as np
import pytest

from bbll.config import load_config, load_stage_config
from bdt_training.train import balance, sample_paths, TRAINING, SIGNAL_MASSES
from bdt_training.submit import build_submit


def test_frozen_training_contract():
    ana = load_config()
    cfg = load_stage_config("bdt_training")
    assert cfg["model_variants"] == {"resolved": "conservative", "boosted": "safer"}
    assert cfg["folds"] == 5
    assert cfg["seed"] == 12345
    assert cfg["sampling"] == {"bulk_divisor": 10, "other_divisor": 1}
    assert len(ana["features"]["resolved"]) == 20
    assert len(ana["features"]["boosted"]) == 25
    assert "mt_hh_mH125" not in ana["features"]["resolved"] + ana["features"]["boosted"]
    assert "n_tags_inside_largeR" not in ana["features"]["resolved"] + ana["features"]["boosted"]
    assert len(SIGNAL_MASSES) == 16
    assert cfg["hyperparameters"]["max_iter"] == 1200


def test_balancing_equalizes_families_cells_and_sb():
    # Six background events covering all 3 families, 48 mass/decay signal cells.
    y = np.asarray([0] * 6 + [1] * 48, dtype=np.int8)
    mass = np.asarray([0] * 6 + [m for m in SIGNAL_MASSES for _ in range(3)], dtype=np.int32)
    decay = np.asarray([-1] * 6 + [d for _ in SIGNAL_MASSES for d in range(3)], dtype=np.int8)
    family = np.asarray([0, 0, 1, 1, 2, 2] + [-1] * 48, dtype=np.int8)
    weights = np.arange(1, len(y) + 1, dtype=np.float64)
    balanced = balance(y, weights, mass, decay, family)
    assert np.isclose(balanced[y == 0].sum(), len(y) / 2)
    assert np.isclose(balanced[y == 1].sum(), len(y) / 2)
    assert np.allclose([balanced[(y == 0) & (family == k)].sum() for k in range(3)], len(y)/6)
    assert np.allclose(balanced[y == 1], len(y)/(2*48))

    missing_cell = (y == 1) & (mass == SIGNAL_MASSES[0]) & (decay == 0)
    with pytest.raises(ValueError, match="coverage"):
        balance(y[~missing_cell], weights[~missing_cell],
                mass[~missing_cell], decay[~missing_cell], family[~missing_cell])


def test_selected_file_inventory_and_reference_manifest(tmp_path):
    folder = tmp_path / "selected" / "run3"
    folder.mkdir(parents=True)
    expected = [f"X{m}_run3" for m in SIGNAL_MASSES] + [
        f"{x}_run3" for x in load_config()["samples"]["prompt_backgrounds"]
    ]
    for stem in expected:
        (folder / f"{stem}.root").touch()
    import json
    counts = load_config()["reference_selected_events"]["run3"]
    (folder / "selection_campaign.json").write_text(json.dumps({
        "status": "PASS", "era": "run3", "n_samples": len(expected),
        "category2_counts": counts,
    }))
    paths = sample_paths(tmp_path / "selected", "run3")
    assert [p.name for p in paths] == sorted(p.name for p in paths)
    assert len(paths) == 26
    (folder / "fake_leak_run3.root").touch()
    with pytest.raises(ValueError, match="extra"):
        sample_paths(tmp_path / "selected", "run3")


def test_condor_has_four_jobs_and_no_site_specific_path(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    py = tmp_path / "python"
    py.touch()
    inp = tmp_path / "selected"
    inp.mkdir()
    spec = build_submit(root, py, inp, tmp_path / "results", tmp_path / "logs")
    assert "run2 resolved" in spec and "run3 boosted" in spec
    assert "request_memory = 16GB" in spec
    assert "request_cpus = 1" in spec
    assert "should_transfer_files = NO" in spec
    assert "/home/aalida/" not in spec
    assert "AF.uchicago" not in spec
