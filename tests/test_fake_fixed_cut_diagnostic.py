"""Run-3 fixed-cut fake-aware preview likelihood nuisance regression tests."""

import numpy as np
import pytest

from fake_backgrounds import fixed_cut_diagnostic as diag
from fake_backgrounds.full_merge import OLD_KAPPAS


def candidate(fvals=None,corr=0.0):
    fvals = [2.,3.] if fvals is None else fvals
    d = np.diag([1.,4.])
    d[0,1]=corr
    d[1,0]=corr
    fake={
        name: {
            "sumw": list(fvals if name=="dror2016_global"
                          else [float(x)*scale for x in fvals]),
            "event_level_covariance": d.tolist()
        }
        for name,scale in zip(OLD_KAPPAS,(1.,2.,.5,1.5))
    }
    return {
        "channel":"resolved",
        "signal_bins":[10.,12.],
        "prompt_background_bins":[100.,200.],
        "prompt_background_sumw2_bins":[25.,40.],
        "mt_edges_GeV":[0,100,200],
        "candidate_index":0,
        "score_threshold":.97,
        "benchmarks":fake,
    }


def test_fixed_cut_nuisances_and_statistical_gate():
    row=diag.build_channel(candidate())
    assert row["background"]==[102,203]
    assert row["background_sumw2"]==[26,44]
    assert row["fake_norm_lo"]==[101.,201.5]
    assert row["fake_norm_hi"]==[103.,204.5]
    assert row["fake_model_lo"]==[101.,201.5]
    assert row["fake_model_hi"]==[104.,206.]
    assert row["minimum_total_neff"]>=10
    assert row["max_abs_mc_correlation"]==0
    assert row["fake_total"]==5.


def test_fixed_cut_rejects_sparse_or_correlated_background():
    inp=candidate()
    inp["prompt_background_bins"]=[1.,1.]
    inp["prompt_background_sumw2_bins"]=[25.,40.]
    with pytest.raises(ValueError,match="Neff or positivity"):
        diag.build_channel(inp)
    with pytest.raises(ValueError,match="correlations exceed"):
        diag.build_channel(candidate(corr=6.0))
    inp=candidate()
    inp["benchmarks"]["dror2016_bonly"]["sumw"]=[-1000.,1.]
    with pytest.raises(ValueError,match="nonpositive fake histosys endpoint"):
        diag.build_channel(inp)


def test_pyhf_template_contains_two_correlated_fake_nuisances():
    pyhf=pytest.importorskip("pyhf")
    first=diag.build_channel(candidate())
    second=diag.build_channel(candidate())
    second["channel"]="boosted_1tag"
    workspace=diag.workspace_for([first,second])
    assert set(workspace.channels)=={"resolved","boosted_1tag"}
    spec=workspace
    for channel in spec["channels"]:
        nuis={x["name"]:x for x in channel["samples"][1]["modifiers"]}
        assert "fake_norm" in nuis
        assert "fake_transfer_model" in nuis
        assert nuis["fake_norm"]["type"]=="histosys"
        assert nuis["fake_transfer_model"]["type"]=="histosys"
    model=workspace.model()
    data=workspace.data(model,include_auxdata=True)
    assert len(data)>0
    second["channel"]="boosted"
    with pytest.raises(ValueError,match="inclusive and split"):
        diag.workspace_for([second,first,{"channel":"boosted_2tag"}])


def test_fixed_winner_preview_marks_unusable_bins_not_clip(monkeypatch,tmp_path):
    row=candidate()
    with pytest.raises(ValueError,match="nonpositive fake histosys endpoint"):
        bad=candidate()
        bad["benchmarks"]["dror2016_bonly"]["sumw"]=[-1000.,1.]
        diag.build_channel(bad)
    # Deliberately sparse combined background -> no fit possible.
    row["prompt_background_bins"]=[1.,1.]
    row["prompt_background_sumw2_bins"]=[25.,40.]
    with pytest.raises(ValueError,match="Neff or positivity"):
        diag.build_channel(row)
