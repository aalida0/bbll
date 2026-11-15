"""Full raw-FakeSim production gates: all candidates, signed event covariance,
public e-mu anchor BEFORE SR cuts and no old Higgs mass veto.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fake_backgrounds import full_grid, full_shard, full_submit


def _screen(mass, channels):
    return {
        "candidates": {
            name: [
                {"score_threshold": .8, "efficiency_target": .8,
                 "n_mt_bins": 2}
            ] for name in channels
        }
    }


def test_grid_includes_all_masses_and_channels_and_rejects_stale(tmp_path, monkeypatch):
    masses = [400, 800]
    channels = ("resolved", "boosted", "boosted_1tag", "boosted_2tag")
    monkeypatch.setattr(full_grid, "load_config", lambda: {
        "samples": {"signal_masses_GeV": masses}
    })
    monkeypatch.setattr(full_grid, "load_stage_config", lambda stage: {
        "master_mt_edges_GeV": [0,1000,2000],
        "minimum_mt_bins": 2,
    })
    monkeypatch.setattr(full_grid, "validate_existing_screen",
                        lambda path, era, mass: _screen(mass,channels))
    for mass in masses:
        folder = tmp_path / "sr_optimization" / "run3" / f"m{mass}"
        folder.mkdir(parents=True)
        (folder/"prompt_candidate_screen.json").write_text(
            json.dumps(_screen(mass,channels))
        )
    doc = full_grid.create_grid(tmp_path)
    assert doc["n_candidates"] == 8
    by_ch,edges = full_grid.pack_candidates(doc)
    assert len(by_ch["boosted"]["index"]) == 2
    assert edges.tolist() == [0,1000,2000]
    out = tmp_path/"grid.json"
    out.write_text(json.dumps(doc))
    assert full_grid.validate_grid(out,tmp_path) == doc
    folder=tmp_path/"sr_optimization/run3/m800/prompt_candidate_screen.json"
    folder.write_text(folder.read_text()+"\n")
    with pytest.raises(ValueError,match="disagrees"):
        full_grid.validate_grid(out,tmp_path)


def test_event_level_correlation_preserved_across_master_bins():
    y=np.zeros((1,2,3))
    M=np.zeros((1,2,3,3))
    n=np.zeros((1,2),dtype=int)
    events = {
        (0,0):{0:.5, 1:.25},
        (0,1):{2:1.0},
    }
    full_shard.accumulate_source_event(events,y,M,n)
    np.testing.assert_allclose(y[0,0],[.5,.25,0])
    assert M[0,0,0,1] == pytest.approx(.125)
    assert M[0,0,1,0] == pytest.approx(.125)
    assert M[0,0,0,0] == pytest.approx(.25)
    assert n.tolist() == [[1,1]]
    full_shard.accumulate_source_event({(0,0):{0:-.1}},y,M,n)
    assert y[0,0,0] == pytest.approx(.4)
    assert M[0,0,0,0] == pytest.approx(.26)
    assert n[0,0] == 2


def test_candidate_channels_count_boosted_in_both_architectures():
    bank = {
        name: {
            "index": np.array([i],dtype=np.int32),
            "score": np.array([cut]),
        } for i,(name,cut) in enumerate((
            ("resolved",.8), ("boosted",.7),
            ("boosted_1tag",.9), ("boosted_2tag",.6)
        ))
    }
    assert full_shard.chosen_indices("resolved",.85,bank).tolist()==[0]
    assert full_shard.chosen_indices("boosted_1tag",.85,bank).tolist()==[1]
    assert full_shard.chosen_indices("boosted_2tag",.85,bank).tolist()==[1,3]
    assert full_shard.chosen_indices("none",.99,bank).size==0


def test_raw_full_producer_short_smoke_preserves_100_event_units(
        tmp_path, monkeypatch):
    raw = tmp_path / "src.delphes.root"
    raw.write_bytes(b"delphes-stand-in")
    src_root = tmp_path / "source_fakes"
    sample = "Wenu_b"
    row = {
        "sample": sample,"ordinal":1,
        "sample_id":"sample_700777","source_id":"source:0001",
        "raw_file":str(raw),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files":[row]}))
    from fake_backgrounds.full_source_audit import raw_paths

    paths=raw_paths(row,src_root)
    for p in paths.values():
        if p == raw:
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(b"cache")
    (src_root/"run3_Wenu_b_raw_weight_summary.json").write_text(
        json.dumps({"norm_factor":1.0,"lumi_fb":308.0,"n_entries":10000})
    )
    (src_root/"run3_association_complete_skip_manifest.json").write_text(
        json.dumps({"skip_events":[],"n_skip_events":0})
    )
    score_grid = {
        "master_mt_edges_GeV":[0,1000,2000],
        "n_candidates":2,
        "fakesim_candidate_rows":[
            {"index":0, "mass_GeV":800,"channel":"resolved",
             "score_threshold":.8},
            {"index":1, "mass_GeV":800,"channel":"boosted",
             "score_threshold":.8},
        ],
    }
    grid_path=tmp_path/"grid.json"
    grid_path.write_text(json.dumps(score_grid))
    monkeypatch.setattr(full_shard,"validate_grid",lambda *a,**k:score_grid)
    monkeypatch.setattr(full_shard,"current_models",lambda *a,**k:(
        {"resolved":("feature",[object()]*5)},
        {"resolved":["sha"]*5,"boosted":["sha"]*5},
    ))
    monkeypatch.setattr(full_shard,"score_hypothesis",
                        lambda hypothesis,uid,scores:("resolved",.95))

    fake=dict(
        raw=SimpleNamespace(
            LOGICAL={"fake":"fake"},
            load_reco_events=lambda raw,entry_start,entry_stop,**kw:(
                [
                    {"entry":i,"event_weight":1.0}
                    for i in range(entry_start,entry_stop)
                ], None
            ),
        ),
        hyp=SimpleNamespace(
            one_prompt_lepton_source=lambda event:{"signal_lepton":1},
            eligible_source_jets=lambda *args:[{"pt":50}],
            build_fake_hypothesis=lambda *args: {
                "hbb_topology":{"pass_hbb_mass_window":False},
                "observables":{"mt_hh_mH125":500.0},
            },
        ),
        bench=SimpleNamespace(
            fake_probability=lambda cfg,jet,flavour: (
                .1 if flavour=="electron" else 0.0
            ),
        ),
        tags=SimpleNamespace(event_uid=lambda *args:1),
        assoc=SimpleNamespace(load_association_cache=lambda *a,**kw:{}),
        larger=SimpleNamespace(load_cache=lambda *a,**kw:{}),
        provenance={"fake":"source_SHA"},
    )
    monkeypatch.setattr(full_shard,"legacy_modules",lambda path:fake)
    monkeypatch.setattr(full_shard,"validate_raw_schema",lambda *a:10000)
    def fake_models(old,n_alpha):
        assert n_alpha==8
        return (
            ["dror2016_global", "dror2016_bonly",
             "curtin2015_mu_A__dror_alpha",
             "curtin2015_mu_B__dror_alpha"],
            [{"alpha":{"mu":.5,"sigma":.1}}]*4,
            np.array([.5]),
            np.ones((4,1)),
            [{-1:.5,1:.5}]*4,
        )
    monkeypatch.setattr(full_shard,"load_benchmarks",fake_models)
    module_gates=SimpleNamespace(
        load_association_skip_manifest=lambda path: {},
        expected_skip_entries=lambda *args: {},
        should_skip_association_incomplete=lambda *a:False,
    )
    module_old=SimpleNamespace(
        anchor_proxy_pass=lambda *a: True,
    )
    monkeypatch.setattr(full_shard.importlib,"import_module",
                        lambda name: module_gates if name.endswith(
                            "fakesim_source_gates") else module_old)
    prefix=tmp_path/"out"/"ord001"
    result=full_shard.run_source_file(
        manifest=manifest,source_root=src_root,legacy_repo=tmp_path,
        model_root=tmp_path,grid_path=grid_path,output_root=tmp_path,
        sample=sample,ordinal=1,output_prefix=prefix,
        max_events=100,n_alpha=8,
    )
    assert result["status"] == "PARTIAL_RUN3_RAW_FAKESIM_DIAGNOSTIC"
    assert result["diagnostics"]["raw_events"] == 100
    assert result["diagnostics"]["scored_outside_old_mass_window"] == 200
    with np.load(str(prefix)+".npz") as npz:
        sw = npz["template_sumw"]
        M = npz["template_second_moment"]
        anchor = npz["anchor_sumw_29fb"]
        n = npz["template_nonzero_events"]
    assert sw[0,0,0] == pytest.approx(10.0)
    assert M[0,0,0,0] == pytest.approx(1.0)
    assert n[0,0] == 100
    assert anchor[0] == pytest.approx(10.0*29/308)
    assert np.all(sw[:,1] == 0)
    assert result["anchor_applied_before_sr_selection"] is True
    with pytest.raises(FileExistsError):
        full_shard.run_source_file(
            manifest=manifest,source_root=src_root,legacy_repo=tmp_path,
            model_root=tmp_path,grid_path=grid_path,output_root=tmp_path,
            sample=sample,ordinal=1,output_prefix=prefix,max_events=100,
        )


def test_submit_batch_uses_output_root_and_refuses_existing_queue(
        tmp_path):
    root=tmp_path/"repo"
    (root/"fake_backgrounds").mkdir(parents=True)
    (root/"fake_backgrounds"/"full_worker.sh").write_text("echo OK\n")
    sample=[{"sample":"Wenu_b","ordinal":1}]
    output=tmp_path/"outputs"
    sub,queue=full_submit.render_submit(root,output,sample)
    assert str(output/"fake_backgrounds"/"full_shards") in queue.read_text()
    text=sub.read_text()
    assert "request_memory = 6000MB" in text
    assert "should_transfer_files = YES" in text
    assert "when_to_transfer_output = ON_EXIT" in text
    assert "transfer_input_files = " + str(root/"fake_backgrounds"/"full_worker.sh") in text
    assert "arguments = full_worker.sh $(sample) $(ordinal) $(prefix)" in text
    assert "should_transfer_files = NO" not in text
    assert "queue sample,ordinal,prefix from" in text
    with pytest.raises(FileExistsError):
        full_submit.render_submit(root,output,sample)


def test_balanced_source_interleaving_preserves_all_jobs_and_source_coverage():
    from collections import Counter

    rows = (
        [{"sample": "Wenu_b", "ordinal": i} for i in range(1, 50)]
        + [{"sample": "Wmunu_b", "ordinal": i} for i in range(1, 50)]
        + [{"sample": "ttbar_semileptonic", "ordinal": i}
           for i in range(1, 300)]
    )
    mixed = full_submit.interleave_source_classes(rows)
    assert Counter((r["sample"], r["ordinal"]) for r in mixed) == Counter(
        (r["sample"], r["ordinal"]) for r in rows
    )
    assert Counter(r["sample"] for r in mixed[:32]) == {
        "Wenu_b": 11,
        "Wmunu_b": 11,
        "ttbar_semileptonic": 10,
    }
    assert [r["ordinal"] for r in mixed if r["sample"] == "Wenu_b"] == list(
        range(1, 50)
    )
    assert mixed[-1]["sample"] == "ttbar_semileptonic"


def test_balanced_source_interleaving_for_subset():
    rows = [{"sample": "ttbar_semileptonic", "ordinal": 12},
            {"sample": "ttbar_semileptonic", "ordinal": 11}]
    result = full_submit.interleave_source_classes(
        rows, sample_names=("ttbar_semileptonic",)
    )
    assert [r["ordinal"] for r in result] == [11, 12]


def test_condor_active_batch_guard_refuses_pending_or_unreadable_jobs(
        monkeypatch):
    from types import SimpleNamespace

    def simulate(stdout, code=0):
        def run(cmd, **kwargs):
            assert cmd[:3] == ["condor_q", "-json", "-constraint"]
            assert "bbll_run3_fakesim_raw_current" in cmd[3]
            return SimpleNamespace(
                returncode=code, stdout=stdout, stderr="condor offline"
            )
        monkeypatch.setattr(full_submit.subprocess, "run", run)

    simulate("[]")
    full_submit.ensure_no_active_fakesim_jobs()

    simulate('[{"ClusterId":4273766,"ProcId":0}]')
    with pytest.raises(RuntimeError, match="still queued/running"):
        full_submit.ensure_no_active_fakesim_jobs()

    simulate("garbled output")
    with pytest.raises(RuntimeError, match="unparseable condor_q"):
        full_submit.ensure_no_active_fakesim_jobs()

    simulate("[]", code=1)
    with pytest.raises(RuntimeError, match="cannot check active"):
        full_submit.ensure_no_active_fakesim_jobs()
