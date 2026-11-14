"""Full-campaign inventory gates for existing raw-Delphes FakeSim sources."""

import json

import numpy as np
import pytest

from fake_backgrounds.full_source_audit import (
    audit_run3, raw_paths, verify_skip_manifest,
)


def fake_source_fixture(tmp_path):
    uproot = pytest.importorskip("uproot")
    data = tmp_path / "fakes"
    manifest = tmp_path / "manifest.json"
    rows = []
    for sample in ("Wenu_b", "ttbar_semileptonic"):
        for ordinal in (1, 2) if sample == "ttbar_semileptonic" else (1,):
            filename = f"{sample}_file{ordinal}.delphes.root"
            raw = tmp_path / filename
            with uproot.recreate(raw) as root:
                root["Delphes"] = {"dummy": np.zeros(10000, dtype=np.int8)}
            row = {
                "sample": sample, "ordinal": ordinal,
                "sample_id": f"Run3_{sample}",
                "source_id": f"source:{sample}:{ordinal}",
                "raw_file": str(raw),
            }
            rows.append(row)
            for name, path in raw_paths(row, data).items():
                if name == "raw":
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"cached")
    for sample, entries in (("Wenu_b", 10000), ("ttbar_semileptonic", 20000)):
        out = data / f"run3_{sample}_raw_weight_summary.json"
        out.write_text(json.dumps({
            "norm_factor": 1.0, "lumi_fb": 312.0,
            "n_entries": entries, "effective_xsec_pb": 2.0,
        }))
    manifest.write_text(json.dumps({"files": rows}))
    skips = data / "run3_association_complete_skip_manifest.json"
    skips.write_text(json.dumps({
        "n_skip_events": 2,
        "skip_events": [
            {
                "sample": "ttbar_semileptonic",
                "source_id": f"source:ttbar_semileptonic:{ordinal}",
                "raw_file": str(tmp_path /
                                f"ttbar_semileptonic_file{ordinal}.delphes.root"),
                "local_entry": 3370,
            }
            for ordinal in (1, 2)
        ],
    }))
    return data, manifest, rows


def test_full_inventory_checks_every_source_and_ttbar_skip(tmp_path):
    data, manifest, rows = fake_source_fixture(tmp_path)
    counts = {"Wenu_b": 1, "ttbar_semileptonic": 2}
    doc = audit_run3(manifest, data, check_root_entries=True,
                     expected_files=counts)
    assert doc["status"] == "RAW_DELPHES_RUN3_SOURCE_INVENTORY_PASS"
    assert doc["n_source_files"] == 3
    assert doc["n_expected_entries"] == 30000
    assert doc["root_entries_verified"] is True
    assert doc["n_skipped_association_incomplete"] == 2
    assert doc["source_counts"] == counts
    assert verify_skip_manifest(data / "run3_association_complete_skip_manifest.json",
                                rows) == 2


def test_missing_cache_or_duplicate_manifest_blocks_campaign(tmp_path):
    data, manifest, rows = fake_source_fixture(tmp_path)
    expected = {"Wenu_b": 1, "ttbar_semileptonic": 2}
    cache = raw_paths(rows[0], data)["association"]
    cache.unlink()
    with pytest.raises(FileNotFoundError, match="association"):
        audit_run3(manifest, data, expected_files=expected)
    cache.write_bytes(b"cached")
    rows[1]["ordinal"] = 1
    rows[1]["sample"] = "Wenu_b"
    manifest.write_text(json.dumps({"files": rows}))
    with pytest.raises(ValueError, match="duplicate source sample/ordinal"):
        audit_run3(manifest, data, expected_files=expected)


def test_missing_or_invalid_normalization_blocks_campaign(tmp_path):
    data, manifest, rows = fake_source_fixture(tmp_path)
    expected = {"Wenu_b": 1, "ttbar_semileptonic": 2}
    summary = data / "run3_Wenu_b_raw_weight_summary.json"
    meta = json.loads(summary.read_text())
    meta["n_entries"] = 5
    summary.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="summary n_entries"):
        audit_run3(manifest, data, expected_files=expected)
    meta["n_entries"] = 10000
    summary.write_text(json.dumps(meta))
    skip = data / "run3_association_complete_skip_manifest.json"
    doc = json.loads(skip.read_text())
    doc["skip_events"][0]["raw_file"] = "wrong_source.root"
    skip.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="skip raw file identity"):
        audit_run3(manifest, data, expected_files=expected)
