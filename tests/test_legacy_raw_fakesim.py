"""Proof that raw-Delphes FakeSim can reuse old reconstruction and current BDTs.

Fixtures deliberately include an Hbb-mass-window-failing signal region event.
The accepted fake prediction must retain it and preserve source-event sumw2.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fake_backgrounds import legacy_raw as pilot
from fake_backgrounds import legacy_inventory as inventory


def test_inventory_builds_from_raw_delphes_provenance_not_ntuple(tmp_path):
    folder = tmp_path / "legacy"
    folder.mkdir()
    raw = tmp_path / "test.delphes.root"
    raw.touch()
    manifest = folder / "btag_provenance.json"
    manifest.write_text(json.dumps({
        "files": [{
            "sample": "Wenu_b", "ordinal": 1,
            "raw_file": str(raw), "sample_id": "sample_W",
            "source_id": "source_0"
        }]
    }))
    command, paths, dest = inventory.build_pilot_command(
        manifest, "Wenu_b", 1, "run3", 800,
        folder, tmp_path/"results", events=100
    )
    assert "--raw-root" in command
    assert command[command.index("--raw-root") + 1] == str(raw)
    assert "--sample-id" in command
    assert command[command.index("--sample-id") + 1] == "sample_W"
    assert "--source-id" in command
    assert command[command.index("--source-id") + 1] == "source_0"
    assert not any("ntuple" in x.lower() for x in command)
    assert dest.name.endswith("m800_n100.json")
    assert str(paths["association"]).endswith(".assoc.npz")


def test_scoring_accepts_nan_missing_geometry_not_infinity():
    class Model:
        def predict_proba(self, arr):
            assert arr.dtype == np.float32
            assert np.isnan(arr[0, 0])
            return np.array([[0.2, 0.8]])

    models = {"resolved": (["geometry"], [Model()]*5)}
    hypothesis = {
        "hbb_topology": {"analysis_subcategory_name": "resolved"},
        "observables": {"geometry": float("nan")},
    }
    assert pilot.score_hypothesis(hypothesis, 7, models) == ("resolved", .8)
    hypothesis["observables"]["geometry"] = float("inf")
    with pytest.raises(ValueError, match="infinite"):
        pilot.score_hypothesis(hypothesis, 7, models)


def test_pure_delphes_fake_score_and_signed_source_event_second_moment(
        tmp_path, monkeypatch):
    uproot = pytest.importorskip("uproot")
    raw = tmp_path / "small.delphes.root"
    with uproot.recreate(raw) as file:
        file["Delphes"] = {"dummy": np.array([1], dtype=np.int32)}
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps({
        "norm_factor": 0.5, "lumi_fb": 308.0, "n_entries": 1
    }))
    assoc, larger = tmp_path/"assoc.npz", tmp_path/"larger.npz"
    assoc.touch()
    larger.touch()
    root = tmp_path/"run"
    model = {"alpha": {"mu": .5, "sigma": .1},
             "charge_weights": {-1: .5, 1: .5}}

    def make_hyp(event, jet, flavour, alpha, charge):
        return {
            "hbb_topology": {
                "analysis_subcategory_name": "resolved",
                "pass_hbb_mass_window": False,
            },
            "observables": {"feature": .2, "mt_hh_mH125": 500.0},
        }

    fake = {
        "raw": SimpleNamespace(
            LOGICAL={"x": "dummy"},
            load_reco_events=lambda *a, **kw: ([
                {"entry": 0, "event_weight": 1.0}
            ], None)
        ),
        "hyp": SimpleNamespace(
            one_prompt_lepton_source=lambda event: {"lepton": 1},
            eligible_source_jets=lambda event, nom: [{"pt": 80}],
            build_fake_hypothesis=make_hyp,
            truncated_gaussian_quadrature=lambda *a: [(.5, 1.0)]
        ),
        "bench": SimpleNamespace(
            benchmark_names=lambda: ["fake"],
            get_benchmark=lambda name: model,
            fake_probability=lambda bench, jet, flavour: (
                .1 if flavour == "electron" else 0.0
            )
        ),
        "tags": SimpleNamespace(event_uid=lambda *a: 2),
        "assoc": SimpleNamespace(load_association_cache=lambda *a, **k: {}),
        "larger": SimpleNamespace(load_cache=lambda *a, **k: {}),
        "provenance": {"fake.py": "fixture"}, "root": str(tmp_path),
    }
    monkeypatch.setattr(pilot, "legacy_modules", lambda path: fake)
    monkeypatch.setattr(pilot, "frozen_winners", lambda *a: (
        {ch: {"threshold": .7, "edges": np.array([0., 1000., 2000.]),
              "efficiency_target": .8} for ch in pilot.CHANNELS},
        "frozen_sha"
    ))

    class Classifier:
        def predict_proba(self, arr):
            return np.array([[.1, .9]])

    monkeypatch.setattr(pilot, "current_models", lambda *a: (
        {"resolved": (["feature"], [Classifier()]*5),
         "boosted": (["feature"], [Classifier()]*5)},
        {"resolved": ["mock"]*5, "boosted": ["mock"]*5}
    ))
    out = tmp_path/"pilot.json"
    row = pilot.run_pilot(
        raw_root=raw, legacy_repo=tmp_path, model_root=root,
        output_root=root, norm_summary=norm, association_cache=assoc,
        larger_cache=larger, era="run3", mass=800, sample="Wenu_b",
        sample_id="sample_W", source_id="source_0",
        output=out, events=1, alpha_nodes=1
    )
    assert row["status"] == "RAW_DELPHES_FAKESIM_PILOT_NOT_NOMINAL"
    assert row["hbb_mass_window_required"] is False
    assert row["diagnostics"]["outside_hbb_window_scored"] == 2
    # One raw event, 2 fake-charge hypotheses each weighted:
    # 0.5 norm * 0.1 fake probability * 0.5 charge = 0.025.
    # Correct event-level total is 0.05; MC second moment = (0.05)^2.
    h = row["templates"]["fake"]["resolved"]
    assert h["sumw"][0] == pytest.approx(.05)
    assert h["second_moment"][0][0] == pytest.approx(.0025)
    assert h["neff"][0] == pytest.approx(1.0)
    assert row["templates"]["fake"]["boosted"]["sumw"] == [0., 0.]
    assert row["historical_fake_kappa_applied"] is False
    assert out.exists()
    with pytest.raises(FileExistsError):
        pilot.run_pilot(
            raw_root=raw, legacy_repo=tmp_path, model_root=root,
            output_root=root, norm_summary=norm, association_cache=assoc,
            larger_cache=larger, era="run3", mass=800, sample="Wenu_b",
            sample_id="sample_W", source_id="source_0",
            output=out, events=1
        )


def test_pilot_rejects_missing_exact_caches_and_wrong_era(tmp_path):
    raw = tmp_path/"raw.delphes.root"
    raw.touch()
    with pytest.raises(FileNotFoundError, match="exact association cache"):
        pilot.run_pilot(
            raw_root=raw, legacy_repo=tmp_path,
            model_root=tmp_path, output_root=tmp_path,
            norm_summary=tmp_path/"none", association_cache=tmp_path/"none",
            larger_cache=tmp_path/"none", era="run3", mass=800,
            sample="Wenu_b", sample_id="a", source_id="b",
            output=tmp_path/"result.json"
        )
    with pytest.raises(ValueError, match="unsupported"):
        pilot.run_pilot(
            raw_root=raw, legacy_repo=tmp_path,
            model_root=tmp_path, output_root=tmp_path,
            norm_summary=tmp_path/"none", association_cache=tmp_path/"none",
            larger_cache=tmp_path/"none", era="run4", mass=800,
            sample="Wenu_b", sample_id="a", source_id="b",
            output=tmp_path/"result.json"
        )
