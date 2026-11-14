"""Prepare/resume HTCondor full-Run3 FakeSim jobs; dry run by default.

The existing 400/400 raw Delphes inputs are read from their audited manifest.
Nothing touches historical FakeSim or prompt-only template namespaces.

First submit --one-per-sample --submit (3 jobs). Check all three full 10k-file
outputs and calibration contributions before submitting remaining 397 sources.

Jobs run 1 CPU, 6 GiB RAM, 4 GiB disk on AF shared /home and /data.
No intermediary ttbar/Wjets selected ntuples are generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from .full_grid import sha256, validate_grid
from .full_merge import shard_prefix, validate_full_shard, OLD_KAPPAS
from .full_source_audit import audit_run3, EXPECTED_RUN3_FILES


def _safe_condor_token(value):
    token = str(value)
    if not token or not re.fullmatch(r"[A-Za-z0-9_./:+-]+",token):
        raise ValueError(f"unsafe HTCondor token: {token!r}")
    return token


def plan(root, provenance, source_root, first_each=False,
         samples=None, max_jobs=None):
    root=Path(root)
    provenance=Path(provenance)
    source_root=Path(source_root)
    inventory=audit_run3(provenance,source_root,check_root_entries=False)
    grid_path=root/"fake_backgrounds"/"run3_candidate_grid.json"
    grid=validate_grid(grid_path,root,check_source=True)
    grid_hash=sha256(grid_path)
    manifest=json.loads(provenance.read_text())
    rows=[r for r in manifest["files"] if r["sample"] in EXPECTED_RUN3_FILES]
    rows.sort(key=lambda r:(r["sample"],int(r["ordinal"])))
    permitted=set(samples or EXPECTED_RUN3_FILES)
    if not permitted.issubset(EXPECTED_RUN3_FILES):
        raise ValueError("unknown requested sample")
    n_mt=len(grid["master_mt_edges_GeV"])-1
    incomplete=[]
    done=[]
    for row in rows:
        prefix=shard_prefix(root,row["sample"],row["ordinal"])
        jp=Path(str(prefix)+".json")
        npz=Path(str(prefix)+".npz")
        tmpj=Path(str(prefix)+".json.tmp")
        tmpn=Path(str(prefix)+".npz.tmp")
        running=Path(str(prefix)+".running")
        if running.exists():
            raise ValueError(
                f"{prefix}: existing in-flight/failed .running lock; "
                "inspect Condor status and source traceback before retrying"
            )
        if jp.is_file() and npz.is_file():
            validate_full_shard(
                prefix,row,source_root,grid_hash,grid["n_candidates"],n_mt,
                expected_models=list(OLD_KAPPAS)
            )
            done.append(row)
        elif any(path.exists() for path in (jp,npz,tmpj,tmpn)):
            raise ValueError(f"partial or interrupted FakeSim job: {prefix}")
        else:
            incomplete.append(row)
    selected=[row for row in incomplete if row["sample"] in permitted]
    if first_each:
        first={}
        for row in selected:
            first.setdefault(row["sample"],row)
        selected=[first[key] for key in sorted(first)]
    if max_jobs is not None:
        if max_jobs<1:
            raise ValueError("max_jobs must be >= 1")
        selected=selected[:max_jobs]
    return {
        "inventory":inventory,
        "grid":grid,
        "already_completed":len(done),
        "unfinished":len(incomplete),
        "selected":selected,
        "all_sources":rows,
    }


def render_submit(root, output_root, rows):
    root=Path(root)
    output_root=Path(output_root)
    batch_payload = "|".join("{}:{}".format(row["sample"], int(row["ordinal"])) for row in rows)
    batch_id=hashlib.sha256(batch_payload.encode()).hexdigest()[:12]
    work=root/"condor"/"work"/"run3_fakesim_current"/("batch_"+batch_id)
    logs=root/"condor"/"logs"/"run3_fakesim_current"
    worker=root/"fake_backgrounds"/"full_worker.sh"
    if not worker.is_file():
        raise FileNotFoundError(worker)
    work.mkdir(parents=True,exist_ok=True)
    logs.mkdir(parents=True,exist_ok=True)
    queue_file=work/"jobs.tsv"
    submit_file=work/"submit.sub"
    for path in (queue_file,submit_file):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing campaign plan: {path}")
    lines=[]
    for row in rows:
        prefix=shard_prefix(output_root,
                            row["sample"],row["ordinal"])
        fields=[
            _safe_condor_token(row["sample"]),
            _safe_condor_token(int(row["ordinal"])),
            _safe_condor_token(prefix),
        ]
        lines.append(" ".join(fields))
    if not lines:
        raise ValueError("no new FakeSim jobs to submit")
    queue_file.write_text("\n".join(lines)+"\n")
    spec=(
        "universe = vanilla\n"
        "executable = /bin/bash\n"
        "arguments = "+_safe_condor_token(worker)+" $(sample) $(ordinal) $(prefix)\n"
        "initialdir = "+_safe_condor_token(root)+"\n"
        "getenv = True\n"
        "should_transfer_files = NO\n"
        "request_cpus = 1\n"
        "request_memory = 6000MB\n"
        "request_disk = 4000MB\n"
        "output = "+_safe_condor_token(logs)+"/$(sample)_$(ordinal).out\n"
        "error = "+_safe_condor_token(logs)+"/$(sample)_$(ordinal).err\n"
        "log = "+_safe_condor_token(logs)+"/$(sample)_$(ordinal).log\n"
        "+JobBatchName = \"bbll_run3_fakesim_raw_current\"\n"
        "queue sample,ordinal,prefix from "+_safe_condor_token(queue_file)+"\n"
    )
    submit_file.write_text(spec)
    return submit_file,queue_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path("/home/aalida/bbll"))
    p.add_argument("--output-root",type=Path,
                   default=Path("/data/aalida/bbll_rerun_20261114"))
    p.add_argument("--manifest",type=Path,default=Path(
        "/data/aalida/bbll_ml/fakes/btag_provenance/run3_btag_provenance.json"
    ))
    p.add_argument("--source-root",type=Path,default=Path("/data/aalida/bbll_ml/fakes"))
    p.add_argument("--one-per-sample",action="store_true",
                   help="Run one complete 10k-event source file per source class first")
    p.add_argument("--samples",nargs="+",choices=tuple(EXPECTED_RUN3_FILES))
    p.add_argument("--max-jobs",type=int)
    p.add_argument("--submit",action="store_true")
    args=p.parse_args()
    params=plan(args.output_root,args.manifest,args.source_root,
                first_each=args.one_per_sample,
                samples=args.samples,max_jobs=args.max_jobs)
    selection=params["selected"]
    print("Run-3 current-model raw FakeSim full-source campaign")
    print("="*78)
    print(f"total sources           : {len(params['all_sources'])}")
    print(f"already complete        : {params['already_completed']}")
    print(f"remaining/unchecked     : {params['unfinished']}")
    print(f"jobs selected           : {len(selection)}")
    print(f"valid SR score candidates: {params['grid']['n_candidates']}")
    print("selected by source      : " + ", ".join(
        f"{name}={sum(r['sample']==name for r in selection)}"
        for name in EXPECTED_RUN3_FILES
    ))
    print(f"submission requested    : {args.submit}")
    if not selection:
        print("NO WORK: no eligible incomplete shards")
        return
    if not args.submit:
        print("DRY RUN. Pass --one-per-sample --submit for the 3-job production smoke.")
        return
    submit,queue=render_submit(args.repo,args.output_root,selection)
    print(f"HTCondor submit file    : {submit}")
    print(f"HTCondor selected jobs  : {queue}")
    completed=subprocess.run(
        ["condor_submit",str(submit)],cwd=args.repo,check=False
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__=="__main__":
    main()
