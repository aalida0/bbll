"""Audit the complete existing Run-3 raw-Delphes FakeSim production inventory.

A verified 100-event pilot does not imply that 400 raw source files and
their identity caches are present. Fail closed before submitting Condor.

This command is READ ONLY. It never overwrites historical FakeSim outputs,
rewrites ROOT, or uses old mass-window-vetoed templates.

Optional --check-root-entries opens every Delphes TTree to verify exactly
10,000 events per source file / 4,000,000 source events total.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .legacy_inventory import source_from_manifest

EXPECTED_RUN3_FILES = {
    "Wenu_b": 50,
    "Wmunu_b": 50,
    "ttbar_semileptonic": 300,
}
EXPECTED_EVENTS_PER_FILE = 10000


def raw_paths(row, source_root):
    name = Path(row["raw_file"]).name
    if not name.endswith(".delphes.root"):
        raise ValueError(f"unexpected raw source filename {name}")
    stem = name[: -len(".delphes.root")]
    sample = row["sample"]
    base = Path(source_root)
    return {
        "raw": Path(row["raw_file"]),
        "association": base / "association_cache" / "run3" / sample
                       / (stem + ".assoc.npz"),
        "largeR": base / "largeR_overlap_cache" / "run3" / sample
                  / (stem + ".largeR_overlap.npz"),
    }


def verify_skip_manifest(path, rows):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing source skip manifest: {path}")
    doc = json.loads(path.read_text())
    events = doc.get("skip_events")
    if (not isinstance(events, list)
            or doc.get("n_skip_events") != len(events)
            or len(events) != 2):
        raise ValueError("Run-3 incomplete-association skip manifest must contain 2 exact entries")
    by_source = {
        (str(row["sample"]), str(row["source_id"])): row for row in rows
    }
    matched = set()
    for event in events:
        if event.get("sample") != "ttbar_semileptonic":
            raise ValueError("unexpected association skip source sample")
        key = (str(event["sample"]), str(event["source_id"]))
        if key not in by_source:
            raise ValueError(f"skip source ID absent from campaign manifest: {key}")
        if str(event["raw_file"]) != str(by_source[key]["raw_file"]):
            raise ValueError(f"skip raw file identity mismatch: {key}")
        local_entry = int(event["local_entry"])
        if not 0 <= local_entry < EXPECTED_EVENTS_PER_FILE:
            raise ValueError("skip local entry outside raw file range")
        unique = (key, local_entry)
        if unique in matched:
            raise ValueError("duplicate skip event")
        matched.add(unique)
    return len(matched)


def audit_run3(manifest, source_root, check_root_entries=False,
               expected_files=EXPECTED_RUN3_FILES, check_skip=True):
    payload = json.loads(Path(manifest).read_text())
    all_files = payload.get("files")
    if not isinstance(all_files, list):
        raise ValueError("fake provenance manifest has no files array")
    rows = [item for item in all_files if item.get("sample") in expected_files]
    if len(rows) != sum(expected_files.values()):
        raise ValueError(
            f"FakeSim source inventory has {len(rows)} rows, expected "
            f"{sum(expected_files.values())}"
        )
    pairs = [(row["sample"], int(row["ordinal"])) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate source sample/ordinal")
    source_ids = [(row["sample"], str(row["source_id"])) for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate original source ID")
    expected_entries = EXPECTED_EVENTS_PER_FILE * sum(expected_files.values())
    found = Counter()
    result_rows = []
    for row in sorted(rows, key=lambda x: (x["sample"], int(x["ordinal"]))):
        name = row["sample"]
        ordn = int(row["ordinal"])
        found[name] += 1
        if ordn not in range(1, expected_files[name]+1):
            raise ValueError(f"{name}: unexpected ordinal {ordn}")
        if not row.get("sample_id") or not row.get("source_id"):
            raise ValueError(f"{name}/{ordn}: incomplete provenance IDs")
        paths = raw_paths(row, source_root)
        for label, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{name}/{ordn}: missing {label}: {path}")
        if check_root_entries:
            import uproot

            with uproot.open(paths["raw"]) as f:
                if "Delphes" not in f:
                    raise ValueError(f"{paths['raw']}: no Delphes tree")
                entries = int(f["Delphes"].num_entries)
            if entries != EXPECTED_EVENTS_PER_FILE:
                raise ValueError(
                    f"{name}/{ordn}: found {entries} ROOT entries, "
                    f"expected {EXPECTED_EVENTS_PER_FILE}"
                )
        result_rows.append({
            "sample": name, "ordinal": ordn,
            "sample_id": str(row["sample_id"]),
            "source_id": str(row["source_id"]),
            "paths": {key: str(path) for key, path in paths.items()},
            "root_entries_checked": bool(check_root_entries),
        })

    if dict(found) != dict(expected_files):
        raise ValueError(f"sample file counts mismatch: {dict(found)}")
    norm = {}
    for sample, count in expected_files.items():
        path = Path(source_root) / f"run3_{sample}_raw_weight_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"{sample}: missing normalization summary: {path}")
        meta = json.loads(path.read_text())
        required = ("norm_factor", "lumi_fb", "n_entries", "effective_xsec_pb")
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(f"{sample}: incomplete weight summary: {missing}")
        if int(meta["n_entries"]) != count * EXPECTED_EVENTS_PER_FILE:
            raise ValueError(f"{sample}: summary n_entries inconsistent with provenance")
        if any(not math.isfinite(float(meta[key])) or float(meta[key]) <= 0
               for key in ("norm_factor", "lumi_fb", "effective_xsec_pb")):
            raise ValueError(f"{sample}: invalid normalization values")
        norm[sample] = {
            key: meta[key] for key in required
        }

    skipped = verify_skip_manifest(
        Path(source_root) / "run3_association_complete_skip_manifest.json",
        rows
    ) if check_skip else None
    return {
        "status": "RAW_DELPHES_RUN3_SOURCE_INVENTORY_PASS",
        "physics_validated": False,
        "source_era": "run3",
        "n_source_files": len(rows),
        "n_expected_entries": expected_entries,
        "root_entries_verified": bool(check_root_entries),
        "n_skipped_association_incomplete": skipped,
        "source_counts": dict(found),
        "source_normalizations": norm,
        "rows": result_rows,
        "nominal_fake_templates_produced": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Original run3_btag_provenance.json"
    )
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/data/aalida/bbll_ml/fakes")
    )
    parser.add_argument(
        "--check-root-entries", action="store_true",
        help="Read all 400 raw Delphes TTree entry counts"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional new diagnostic manifest JSON; refuses overwrite"
    )
    args = parser.parse_args()
    result = audit_run3(args.manifest, args.source_root,
                        check_root_entries=args.check_root_entries)
    print("Full Run-3 raw-Delphes FakeSim inventory")
    print("="*70)
    for name, expected in EXPECTED_RUN3_FILES.items():
        count = result["source_counts"][name]
        print(f"{name:22s}: {count:3d}/{expected} sources")
    print(f"source files      : {result['n_source_files']}")
    print(f"expected events   : {result['n_expected_entries']:,}")
    print(f"ROOT entries read : {result['root_entries_verified']}")
    print(f"association skips: {result['n_skipped_association_incomplete']}")
    print(f"RESULT: {result['status']}")
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
        print(f"Saved verified source path manifest: {args.output}")


if __name__ == "__main__":
    main()
