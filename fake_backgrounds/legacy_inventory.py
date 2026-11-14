"""Generate a verified raw-Delphes FakeSim pilot command from legacy provenance.

No sample ntuples are expected. The btag provenance manifest already records
raw file, sample ID and source ID used for deterministic fake/BTAG reconstruction.
Use this helper to avoid manual guesses about those identifiers.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .legacy_raw import SOURCES


def source_from_manifest(path, sample, ordinal):
    payload = json.loads(Path(path).read_text())
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("source provenance manifest has no files array")
    rows = [
        row for row in files
        if row.get("sample") == sample
        and int(row.get("ordinal", -1)) == int(ordinal)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"cannot identify unique source {sample}/ordinal={ordinal}: "
            f"{len(rows)} records"
        )
    row = rows[0]
    for key in ("raw_file", "sample_id", "source_id"):
        if key not in row or not row[key]:
            raise ValueError(f"provenance row has no {key}")
    return row


def build_pilot_command(manifest, sample, ordinal, era, mass, legacy_repo,
                        output_root, model_root=None, source_root=None, events=100):
    if sample not in SOURCES or era not in ("run2", "run3"):
        raise ValueError("unsupported FakeSim source or era")
    row = source_from_manifest(manifest, sample, ordinal)
    raw = Path(row["raw_file"])
    if not raw.name.endswith(".delphes.root"):
        raise ValueError(f"unsupported raw Delphes filename: {raw.name}")
    stem = raw.name[:-len(".delphes.root")]
    data = Path(source_root or "/data/aalida/bbll_ml/fakes")
    assoc = data / "association_cache" / era / sample / f"{stem}.assoc.npz"
    larger = data / "largeR_overlap_cache" / era / sample / f"{stem}.largeR_overlap.npz"
    norm = data / f"{era}_{sample}_raw_weight_summary.json"
    skipped = data / f"{era}_association_complete_skip_manifest.json"
    out = Path(output_root) / "fake_backgrounds" / "raw_pilots" / era / (
        f"{sample}_ord{ordinal:03d}_m{mass}_n{events}.json"
    )
    paths = {
        "raw": raw, "association": assoc, "largeR": larger,
        "normalization": norm, "skip_manifest_optional": skipped,
    }
    command = [
        "python", "-m", "fake_backgrounds.legacy_raw",
        "--raw-root", str(raw),
        "--legacy-repo", str(legacy_repo),
        "--model-root", str(model_root or output_root),
        "--output-root", str(output_root),
        "--norm-summary", str(norm),
        "--association-cache", str(assoc),
        "--larger-cache", str(larger),
        "--era", era, "--mass", str(mass),
        "--sample", sample,
        "--sample-id", str(row["sample_id"]),
        "--source-id", str(row["source_id"]),
        "--events", str(events),
        "--output", str(out),
    ]
    if skipped.is_file():
        command += ["--skip-manifest", str(skipped)]
    return command, paths, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample", choices=SOURCES, required=True)
    parser.add_argument("--ordinal", type=int, default=1)
    parser.add_argument("--era", choices=("run2","run3"), default="run3")
    parser.add_argument("--mass", type=int, default=800)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--execute", action="store_true",
        help="After auditing, execute the pilot directly with this Python; "
             "refuse if required input files are missing or output exists",
    )
    args = parser.parse_args()
    cmd, paths, output = build_pilot_command(
        args.manifest, args.sample, args.ordinal, args.era, args.mass,
        args.legacy_repo, args.output_root, args.model_root,
        args.source_root, args.events
    )
    print("Raw Delphes FakeSim input audit:")
    for name, path in paths.items():
        tag = "FOUND" if path.is_file() else "MISSING"
        if name == "skip_manifest_optional":
            tag += " (optional pilot; mandatory when producing full ttbar)"
        print(f"  {tag:10s} {name:25s} {path}")
    print("\nPilot command (DO NOT run until required inputs above are present):")
    print(" ".join(shlex.quote(item) for item in cmd))
    print(f"\nOutput: {output}")
    print("No intermediate ntuple generation is required.")
    if args.execute:
        required = {
            key: path for key, path in paths.items()
            if key != "skip_manifest_optional"
        }
        missing = [f"{key}: {path}" for key, path in required.items()
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot execute FakeSim pilot; missing required inputs: "
                + "; ".join(missing)
            )
        if args.sample == "ttbar_semileptonic" and not paths[
                "skip_manifest_optional"].is_file():
            raise FileNotFoundError(
                "ttbar pilot requires historical exact-association skip manifest"
            )
        if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
            raise FileExistsError(
                f"Refusing to overwrite existing or incomplete pilot: {output}"
            )
        executable_command = [sys.executable] + cmd[1:]
        print("\nExecuting verified raw-Delphes FakeSim pilot:", flush=True)
        print(" ".join(shlex.quote(arg) for arg in executable_command),
              flush=True)
        subprocess.run(executable_command, check=True)


if __name__ == "__main__":
    main()
