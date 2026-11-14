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
    assert "getenv = False" in spec
    assert "/home/aalida/" not in spec
    assert "AF.uchicago" not in spec



def test_category_filtered_root_training_pool_with_all_48_signal_cells(tmp_path):
    """Verify the new selected-ROOT reader is equivalent to topology reduction."""
    import json
    import uproot
    from bdt_training.train import read_pool

    ana = load_config()
    folder = tmp_path / "selected" / "run3"
    folder.mkdir(parents=True)
    stems = (
        [f"X{m}_run3" for m in SIGNAL_MASSES]
        + [f"{p}_run3" for p in ana["samples"]["prompt_backgrounds"]]
    )
    all_features = list(dict.fromkeys(
        ana["features"]["resolved"] + ana["features"]["boosted"]
    ))
    for stem in stems:
        mass = int(stem.split("_")[0][1:]) if stem.startswith("X") else 0
        sig = mass > 0
        uid, tags, decays = [], [], []
        # Every fold, every signal decay, with resolved + two boosted categories.
        for fold in range(5):
            for decay in (0, 1, 2) if sig else (0,):
                for cat in (0, 1, 2):
                    uid.append(50 * len(uid) + fold)  # training sample passes 1/10
                    tags.append(cat)
                    decays.append(decay if sig else -1)
        # A resolved event excluded by deterministic 1/10 bulk subsampling.
        # The rare 'other' background group keeps it, as in the frozen trainer.
        uid.append(5005)
        tags.append(0)
        decays.append(-1 if not sig else 0)
        n = len(uid)
        a = {f: np.ones(n, dtype=np.float32) for f in all_features}
        a.update({
            "event_uid": np.asarray(uid, dtype=np.uint64),
            "category3": np.asarray(tags, dtype=np.int8),
            "is_signal": np.full(n, sig, dtype=np.bool_),
            "signal_mass_GeV": np.full(n, mass, dtype=np.int16),
            "signal_decay_id": np.asarray(decays, dtype=np.int8),
            "train_weight_base": np.ones(n, dtype=np.float64),
        })
        with uproot.recreate(folder / f"{stem}.root") as f:
            f.mktree("events", {key: value.dtype for key, value in a.items()}).extend(a)
    (folder / "selection_campaign.json").write_text(json.dumps({
        "status": "PASS", "era": "run3", "n_samples": len(stems),
        "category2_counts": ana["reference_selected_events"]["run3"],
    }))
    for topology, expected_sig in (("resolved", 240), ("boosted", 480)):
        x, y, absw, fold, mass, decay, family = read_pool(
            tmp_path / "selected", "run3", topology
        )
        assert x.shape[1] == (20 if topology == "resolved" else 25)
        assert np.count_nonzero(y) == expected_sig
        assert set(fold.tolist()) == set(range(5))
        assert set(mass[y == 1].tolist()) == set(SIGNAL_MASSES)
        for k in range(5):
            chosen = fold == k
            assert np.all(np.isfinite(balance(
                y[chosen], absw[chosen], mass[chosen],
                decay[chosen], family[chosen],
            )))



def test_submit_preserves_virtualenv_python_symlink(tmp_path, monkeypatch):
    import sys
    from bdt_training.submit import main

    repo = tmp_path / "repo"
    selected = tmp_path / "selected"
    out = tmp_path / "results"
    python = tmp_path / "venv" / "bin" / "python"
    repo.mkdir()
    selected.mkdir()
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)

    monkeypatch.setattr(sys, "argv", [
        "submit.py", "--repo", str(repo), "--python", str(python),
        "--selected-root", str(selected), "--output-root", str(out),
    ])
    main()
    text = (out / "logs" / "bdt_training.sub").read_text()
    assert f"executable = {python}" in text


def test_reference_environment_rejects_mixed_python_packages():
    from bdt_training.environment import verify_runtime

    clean = verify_runtime(
        python="3.9", numpy_version="2.0.2", sklearn_version="1.6.1",
        paths=["/usr/lib/python3.9", "/home/example/bbll/.venv39/site-packages"],
    )
    assert clean["python"] == "3.9"

    with pytest.raises(RuntimeError, match="Python 3.13"):
        verify_runtime(
            python="3.13", numpy_version="2.0.2", sklearn_version="1.6.1",
            paths=["/clean"],
        )
    with pytest.raises(RuntimeError, match="Key4hep"):
        verify_runtime(
            python="3.9", numpy_version="2.0.2", sklearn_version="1.6.1",
            paths=["/cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/lib/python3.13"],
        )
    with pytest.raises(RuntimeError, match="NumPy"):
        verify_runtime(
            python="3.9", numpy_version="2.2.0", sklearn_version="1.6.1",
            paths=["/clean"],
        )
