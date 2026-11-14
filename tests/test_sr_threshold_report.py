"""Check BDT score report reads the winning candidate, not signal efficiency."""

import json

import pytest

import sr_optimization.threshold_report as report
from sr_optimization.prompt_likelihood import ARCHITECTURES


def test_score_report_exposes_mass_and_channel_specific_numeric_thresholds(
        tmp_path, monkeypatch):
    mass = 600
    folder = tmp_path / "sr_optimization" / "run3" / f"m{mass}"
    folder.mkdir(parents=True)
    payload = {
        "architectures": {
            arch: {
                "best": {
                    "expected_mu95": 10.0 if arch == "two_channel" else 8.0,
                    "candidates": {
                        channel: {
                            "score_threshold": 0.1234 + i*.11,
                            "efficiency_target": .9 - .1*i,
                        }
                        for i, channel in enumerate(channels)
                    },
                }
            }
            for arch, channels in ARCHITECTURES.items()
        }
    }
    path = folder / "prompt_pyhf_top3.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(report, "MASSES", (600,))
    monkeypatch.setattr(report, "validate_comparison",
                        lambda *args: {"file": str(path)})
    rows = report.chosen_rows(tmp_path, ("run3",), "top3")
    assert len(rows) == 5
    assert len([x for x in rows if x["architecture"] == "three_channel"]) == 3
    assert rows[0]["score_threshold"] == pytest.approx(.1234)
    assert rows[0]["score_threshold"] != rows[0]["signal_efficiency_target"]
    assert all(x["mass_GeV"] == mass for x in rows)
    assert all(x["optimization"] == "top3" for x in rows)


def test_score_report_rejects_unknown_sources(tmp_path):
    with pytest.raises(ValueError, match="source must"):
        report.chosen_rows(tmp_path, ("run3",), "unknown")
