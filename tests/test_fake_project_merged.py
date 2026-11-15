"""Full 4M FakeSim projected-template regression checks."""

import json

import numpy as np
import pytest

from fake_backgrounds import project_merged as pr


def test_grouped_fake_covariance_includes_cross_master_bin_event_terms():
    edges = [0.,100.,200.,300.,400.]
    chosen = [0.,200.,400.]
    g = pr.grouping_matrix(edges,chosen)
    np.testing.assert_array_equal(g,[[1,1,0,0],[0,0,1,1]])
    # Two source events contribute to different master bins within the same
    # merged group. Positive off-diagonal source-event contributions must
    # survive G C G^T.
    x = np.array([1.,2.,3.,4.])
    event1 = np.array([1.,2.,0.,0.])
    event2 = np.array([0.,0.,3.,4.])
    cov = np.outer(event1,event1)+np.outer(event2,event2)
    y, m = pr.project(x,cov,g)
    assert y.tolist() == [3.,7.]
    assert m[0,0] == pytest.approx(9.)   # not diagonal 5
    assert m[1,1] == pytest.approx(49.)  # not diagonal 25
    assert m[0,1] == 0


def test_nonmaster_edges_and_negative_fake_variance_are_rejected():
    with pytest.raises(ValueError,match="not a frozen master edge"):
        pr.grouping_matrix([0,100,200,300],[0,125,300])
    with pytest.raises(ValueError,match="span full frozen master"):
        pr.grouping_matrix([0,100,200,300],[100,200,300])
    g = pr.grouping_matrix([0,100,200,300],[0,200,300])
    with pytest.raises(ValueError,match="negative fake variance"):
        pr.project([1,1,1],np.diag([-3.,0.,1.]),g)


def test_fake_candidate_mapping_requires_exact_optimized_bdt_cut():
    grid = {
        "master_mt_edges_GeV":[0.,100.,200.,300.],
        "fakesim_candidate_rows":[{
            "index": 0, "mass_GeV": 800,
            "channel": "resolved", "local_index": 2,
            "score_threshold": .997, "efficiency_target": .80,
        }]
    }
    arrays = {
        "calibrated_sumw":np.ones((4,1,3)),
        "calibrated_second_moment":np.tile(
            np.eye(3),(4,1,1,1)
        ),
    }
    row = {
        "score_threshold":.997,"efficiency_target":.8,
        "edges_GeV":[0.,200.,300.],
        "background_bins":[10,20],
        "background_sumw2_bins":[1,4],
        "signal_bins":[2,1]
    }
    result=pr.project_candidate(grid,arrays,800,"resolved",2,row)
    assert result["grid_index"]==0
    assert result["benchmarks"]["dror2016_global"]["sumw"]==[2.,1.]
    assert result["benchmarks"]["dror2016_global"]["neff"]==[2.,1.]
    row["score_threshold"] = .998
    with pytest.raises(ValueError,match="not identical"):
        pr.project_candidate(grid,arrays,800,"resolved",2,row)


def test_mass_report_preserves_all_candidates_and_rejects_stale_inventory(
        tmp_path,monkeypatch):
    grid = {
        "n_candidates": 2,
        "master_mt_edges_GeV": [0,100,200],
        "fakesim_candidate_rows": [
            {"index":0,"mass_GeV":800,"channel":"resolved",
             "local_index":0,"score_threshold":.6,"efficiency_target":.8},
            {"index":1,"mass_GeV":800,"channel":"boosted",
             "local_index":0,"score_threshold":.7,"efficiency_target":.9},
        ],
    }
    arrays = {
        "calibrated_sumw":np.ones((4,2,2)),
        "calibrated_second_moment":np.tile(
            np.eye(2),(4,2,1,1)
        )
    }
    meta = {
        "npz_sha256":"abc123", "candidate_grid_sha256":"abc",
        "current_luminosity_fb":308.
    }
    monkeypatch.setattr(pr,"load_merged",lambda root:(grid,meta,arrays))
    prompt = {
        "candidates": {
            "resolved":[{
                "score_threshold":.6,"efficiency_target":.8,
                "edges_GeV":[0,100,200],"background_bins":[10,20],
                "background_sumw2_bins":[1,4],"signal_bins":[2,3],
            }],
            "boosted":[{
                "score_threshold":.7,"efficiency_target":.9,
                "edges_GeV":[0,100,200],"background_bins":[10,20],
                "background_sumw2_bins":[1,4],"signal_bins":[2,3],
            }],
            "boosted_1tag":[], "boosted_2tag":[],
        }
    }
    monkeypatch.setattr(pr,"validate_existing_screen",lambda *args:prompt)
    result=pr.mass_report(tmp_path,800)
    assert result["status"]=="RUN3_FAKESIM_MASS_PROJECTED_DIAGNOSTIC_NOT_FINAL"
    assert len(result["candidates"])==2
    assert all(not x["benchmarks"]["dror2016_global"]["negative_bins"]
               for x in result["candidates"])
    grid["fakesim_candidate_rows"].pop()
    with pytest.raises(ValueError,match="incomplete mass coverage"):
        pr.mass_report(tmp_path,800)
