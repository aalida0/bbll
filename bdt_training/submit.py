"""Generate a site-neutral HTCondor submit file for the four BDT banks.

Shared filesystem is required for code, ROOT inputs, and output directories.
The submit file uses absolute paths supplied by the caller, not site defaults.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from bbll.config import load_stage_config


def build_submit(repo: Path, python: Path, selected: Path,
                 output: Path, logs: Path):
    cfg = load_stage_config("bdt_training")["condor"]
    for p in (repo, python, selected):
        if not p.exists():
            raise FileNotFoundError(p)
    for p in (repo, python, selected, output, logs):
        if any(c.isspace() for c in str(p)):
            raise ValueError(f"HTCondor path contains whitespace: {p}")
    logs.mkdir(parents=True, exist_ok=True)
    return f"""universe = vanilla
initialdir = {repo}
executable = {python}
arguments = -m bdt_training.train $(era) $(topology) --selected-root {selected} --output-root {output}
getenv = True
environment = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
should_transfer_files = NO
request_cpus = {int(cfg['cpus'])}
request_memory = {cfg['memory']}
request_disk = {cfg['disk']}
output = {logs}/bdt_$(era)_$(topology)_$(ClusterId).out
error = {logs}/bdt_$(era)_$(topology)_$(ClusterId).err
log = {logs}/bdt_$(ClusterId).log
notification = Never
queue era,topology from (
run2 resolved
run2 boosted
run3 resolved
run3 boosted
)
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    repo, interpreter, selected, out = (
        a.repo.resolve(), a.python.resolve(), a.selected_root.resolve(),
        a.output_root.resolve(),
    )
    logs = out / "logs"
    spec = build_submit(repo, interpreter, selected, out, logs)
    path = logs / "bdt_training.sub"
    path.write_text(spec)
    print(f"HTCondor submit file: {path}", flush=True)
    if a.submit:
        if shutil.which("condor_submit") is None:
            raise RuntimeError("condor_submit unavailable; run locally or submit from a Condor login node")
        subprocess.run(["condor_submit", str(path)], check=True)
    else:
        print(f"Review, then run: condor_submit {path}")


if __name__ == "__main__":
    main()
