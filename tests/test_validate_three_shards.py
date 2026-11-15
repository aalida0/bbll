"""Check full-source three-shard numerical validation with controlled data."""

import json

import numpy as np
import pytest

from fake_backgrounds import validate_three_shards as gate


def _fixture(tmp_path, monkeypatch, *, indefinite=False):
    source_root = tmp_path/"input"
    source_root.mkdir()
    root = tmp_path/"results"
    root.mkdir()
    rows = [{"sample": sample, "ordinal": 1, "sample_id": sample,
             "source_id": "source:"+sample}
            for sample in gate.SAMPLES]
    manifest = tmp_path/"provenance.json"
    manifest.write_text(json.dumps({"files": rows}))
    grid = {"n_candidates": 1,
            "master_mt_edges_GeV": [0,1000,2000],
            "fakesim_candidate_rows": [{
                "index": 0, "mass_GeV": 800, "channel": "resolved",
                "efficiency_target": .8
            }]}
    (root/"fake_backgrounds").mkdir()
    gridpath = root/"fake_backgrounds/run3_candidate_grid.json"
    gridpath.write_text(json.dumps(grid))
    skip = source_root/"run3_association_complete_skip_manifest.json"
    skip.write_text("{}")
    for row in rows:
        (source_root/f"run3_{row['sample']}_raw_weight_summary.json").write_text(
            json.dumps({"n_entries": 10000})
        )
    def checked_grid(path, parent, check_source=True):
        assert path == gridpath
        return grid
    monkeypatch.setattr(gate,"validate_grid",checked_grid)
    def fake_loader(prefix, row, srcroot, gridhash, nc, nm, expected_models):
        sample = row["sample"]
        norm = srcroot/f"run3_{sample}_raw_weight_summary.json"
        mock = {
            "status": "COMPLETE_RUN3_RAW_FAKESIM_SHARD",
            "source_manifest_sha256": gate.sha256(manifest),
            "skip_manifest_sha256": gate.sha256(skip),
            "normalization_summary_sha256": gate.sha256(norm),
            "events_processed": 10000,
            "observed_skips": [],
            "diagnostics": {
                "scored_hypotheses": 120,
                "scored_outside_old_mass_window": 60,
            },
            "current_model_sha256": {"resolved": ["modelsha"]},
            "legacy_source_sha256": {"src": "legacysha"},
            "benchmark_configurations": ["fixed"] * 4,
            "benchmarks": list(gate.OLD_KAPPAS),
            "grid_sha256": gate.sha256(gridpath),
        }
        y = np.ones((4,1,2))*2
        cov = np.zeros((4,1,2,2))
        cov[:,:,0,0] = 1
        cov[:,:,1,1] = 1
        cov[:,:,0,1] = cov[:,:,1,0] = .2
        if indefinite and sample == "Wenu_b":
            cov[0,0,0,1] = cov[0,0,1,0] = 5
        counts = np.ones((4,1),dtype=int)*50
        a = np.ones(4)
        av = np.ones(4)*.2
        an = np.ones(4,dtype=int)*10
        edges = np.array([0,1000,2000])
        return mock, (y, cov, counts, a, av, an, edges)

    monkeypatch.setattr(gate,"validate_full_shard",fake_loader)
    return root,source_root,manifest,gridpath


def test_three_shard_numerical_gate_reports_anchor_and_signed_moments(
        tmp_path,monkeypatch):
    paths=_fixture(tmp_path,monkeypatch)
    result=gate.validate_three(*paths[:3],grid_path=paths[3])
    assert result["status"]=="RUN3_FAKESIM_THREE_FULL_SHARDS_NUMERICAL_PASS"
    assert result["raw_events_verified"]==30000
    assert set(result["samples"])==set(gate.SAMPLES)
    x=result["samples"]["Wenu_b"]
    assert x["nonzero_selected_candidate_model_pairs"]==4
    assert x["mass800_nonzero_candidates"][0]["neff"]==pytest.approx(
        16/2.4
    )
    assert x["anchor_sumw_29fb"]["dror2016_global"]==1
    assert result["calibration_kappa_frozen"] is False


def test_non_psd_event_level_covariance_fails_closed(tmp_path,monkeypatch):
    paths=_fixture(tmp_path,monkeypatch,indefinite=True)
    with pytest.raises(ValueError,match="non-PSD covariance"):
        gate.validate_three(*paths[:3],grid_path=paths[3])
