"""Reproduce preselection for an era from the prepared ROOT campaign.

No site-specific paths, batch queue, or environment setup are required.
Optional old-reduction comparisons are a one-time audit, not needed by readers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from .selection import CONFIG, STAGE, RESOLVED, BOOSTED, reduce_sample


def samples(era: str):
    inventory = CONFIG["samples"]
    return (
        [("signal", f"X{mass}_{era}") for mass in inventory["signal_masses_GeV"]]
        + [("background", f"{name}_{era}") for name in inventory["prompt_backgrounds"]]
    )


def check_output(path: Path, report: dict) -> None:
    """Do not silently trust pre-existing results when resuming a campaign."""
    with uproot.open(path) as f:
        tree = f["events"]
        if int(tree.num_entries) != int(report["selected"]):
            raise ValueError(f"{path}: output count differs from report")
        counts = np.zeros(3, dtype=np.int64)
        last_entry = -1
        for arrays in tree.iterate(
            ["category3", "category2", "source_entry", "physics_weight",
             "train_weight_base"], step_size="100 MB", library="np"
        ):
            three = arrays["category3"]
            two = arrays["category2"]
            if not np.all((three >= 0) & (three <= 2)):
                raise ValueError(f"{path}: invalid category3")
            if not np.array_equal(two, np.where(three == 0, 0, 1)):
                raise ValueError(f"{path}: category2 does not close")
            counts += np.bincount(three, minlength=3)
            entry = arrays["source_entry"]
            if (len(entry) and (
                entry[0] <= last_entry or np.any(np.diff(entry) <= 0)
            )):
                raise ValueError(f"{path}: source entries not strictly ordered")
            if len(entry):
                last_entry = int(entry[-1])
            w = arrays["physics_weight"]
            if not np.all(np.isfinite(w)):
                raise ValueError(f"{path}: nonfinite signed weight")
            if not np.array_equal(arrays["train_weight_base"], np.abs(w)):
                raise ValueError(f"{path}: training weight is inconsistent")
        expected_counts = np.asarray(
            [report["counts"][k] for k in
             ("resolved", "boosted_1tag", "boosted_2tag")], dtype=np.int64
        )
        if not np.array_equal(counts, expected_counts):
            raise ValueError(f"{path}: per-category counts differ from report")
    if int(report["selected"]) != sum(
        int(report["counts"][k])
        for k in ("resolved", "boosted_1tag", "boosted_2tag")
    ):
        raise ValueError(f"{path}: category count closure failed")


def check_reference(selected: Path, reference_dir: Path, era: str, stem: str) -> None:
    """Require exact per-event closure with the earlier validated reduction."""
    with uproot.open(selected) as f:
        current = f["events"]
        cat = current["category3"].array(library="np")
        for topology, mask, features in (
            ("resolved", cat == 0, RESOLVED),
            ("boosted", cat >= 1, BOOSTED),
        ):
            old_path = reference_dir / era / topology / f"{stem}.root"
            with uproot.open(old_path) as ref_file:
                old = ref_file["events"]
                idx = np.flatnonzero(mask)
                if len(idx) != int(old.num_entries):
                    raise ValueError(
                        f"{stem}/{topology}: new count {len(idx)} != "
                        f"reference {old.num_entries}"
                    )
                # Event ordering, signed weights, reconstructed fit observable,
                # and *all* frozen classifier variables must agree exactly.
                for branch in ("event_uid", "physics_weight", "mt_hh_mH125",
                               "n_tags_inside_largeR", *features):
                    now = current[branch].array(library="np")[idx]
                    before = old[branch].array(library="np")
                    if now.dtype.kind in "fc" or before.dtype.kind in "fc":
                        same = np.array_equal(now, before, equal_nan=True)
                    else:
                        same = np.array_equal(now, before)
                    if not same:
                        raise ValueError(
                            f"{stem}/{topology}: mismatch in {branch}"
                        )


def run_campaign(
    *,
    input_root: Path,
    output_root: Path,
    era: str,
    resume: bool = False,
    reference_dir: Path | None = None,
    expected: list[tuple[str, str]] | None = None,
) -> dict:
    if era not in CONFIG["eras"]:
        raise ValueError("era must be run2 or run3")
    inventory = samples(era) if expected is None else expected
    if len(set(inventory)) != len(inventory):
        raise ValueError("duplicate sample in inventory")

    # Fail before production if any requested sample or sidecar is absent.
    files = []
    for kind, stem in inventory:
        source = input_root / kind / era / f"{stem}.root"
        weights = input_root / kind / era / f"{stem}{STAGE['input']['weight_suffix']}"
        if not source.is_file() or not weights.is_file():
            raise FileNotFoundError(f"missing {source} or {weights}")
        files.append((stem, source, weights))

    rows = []
    totals = {name: 0 for name in (
        "input", "common", "resolved", "boosted_1tag", "boosted_2tag"
    )}
    for stem, source, weights in files:
        output = output_root / era / f"{stem}.root"
        report_file = output.with_suffix(".selection.json")
        if output.exists() or report_file.exists():
            if not resume or not output.is_file() or not report_file.is_file():
                raise FileExistsError(
                    f"existing/incomplete output for {stem}; use --resume "
                    "only when both ROOT and selection JSON exist"
                )
            report = json.loads(report_file.read_text())
            if report["input"] != str(source) or report["weights"] != str(weights):
                raise ValueError(f"{stem}: input paths differ from saved report")
            check_output(output, report)
        else:
            report = reduce_sample(source, weights, output)
            check_output(output, report)

        if reference_dir is not None:
            check_reference(output, reference_dir, era, stem)

        for key in totals:
            totals[key] += int(report["counts"][key])
        rows.append({"sample": stem, "counts": report["counts"],
                     "selected": int(report["selected"])})
        print(f"{stem:25s} selected={int(report['selected']):>10,d}"
              + (" reference=PASS" if reference_dir is not None else ""),
              flush=True)

    resolved = totals["resolved"]
    boosted = totals["boosted_1tag"] + totals["boosted_2tag"]
    if resolved + boosted != sum(row["selected"] for row in rows):
        raise AssertionError("campaign-wide category closure failed")
    if expected is None:
        frozen = CONFIG["reference_selected_events"][era]
        if (resolved, boosted) != (frozen["resolved"], frozen["boosted"]):
            raise ValueError(
                f"{era}: full campaign differs from frozen reference counts: "
                f"new={(resolved, boosted)}, expected={frozen}"
            )
    result = {
        "status": "PASS",
        "era": era,
        "n_samples": len(rows),
        "counts": totals,
        "category2_counts": {"resolved": resolved, "boosted": boosted},
        "category3_counts": {
            name: totals[name]
            for name in ("resolved", "boosted_1tag", "boosted_2tag")
        },
        "reference_checked": reference_dir is not None,
        "samples": rows,
    }
    final = output_root / era / "selection_campaign.json"
    if final.exists() and not resume:
        raise FileExistsError(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"RESULT: PASS ({era})  resolved={resolved:,} boosted={boosted:,}")
    print(f"Manifest: {final}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--era", choices=(*CONFIG["eras"], "both"), required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reference-dir", type=Path)
    a = p.parse_args()
    for era in (CONFIG["eras"] if a.era == "both" else (a.era,)):
        run_campaign(input_root=a.input_root, output_root=a.output_root,
                     era=era, resume=a.resume, reference_dir=a.reference_dir)


if __name__ == "__main__":
    main()
