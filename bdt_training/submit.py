"""Generate a portable HTCondor file-transfer submit description.

Only the small launcher is transferred to workers. The installed Python
environment and ROOT data are referenced using worker-accessible absolute
paths, following the older UChicago AF training setup.
"""

from __future__ import annotations

import argparse
import os
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
    launcher = repo / "bdt_training" / "worker.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"HTCondor launcher missing: {launcher}")
    logs.mkdir(parents=True, exist_ok=True)
    return f"""universe = vanilla
initialdir = {repo}
executable = /bin/bash
arguments = worker.sh $(era) $(topology) {python} {repo} {selected} {output}
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {launcher}
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
    p.add_argument(
        "--log-dir", type=Path,
        help="HTCondor-approved log directory (default: <repo>/condor/logs, not /data)",
    )
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    repo, interpreter, selected, out = (
        a.repo.resolve(), a.python.absolute(), a.selected_root.resolve(),
        a.output_root.resolve(),
    )
    # Condor event logs must not reside on Ceph. In AF, the input/model
    # files are on /data (Ceph), but the repository is on /home.
    logs = (a.log_dir if a.log_dir is not None else repo / "condor" / "logs").resolve()
    spec = build_submit(repo, interpreter, selected, out, logs)
    path = logs / "bdt_training.sub"
    path.write_text(spec)
    print(f"HTCondor submit file: {path}", flush=True)
    if a.submit:
        # Validate the actual Python used by worker jobs, not the login shell.
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONPATH", None)
        clean_env.pop("PYTHONHOME", None)
        subprocess.run(
            [str(interpreter), "-m", "bdt_training.environment"],
            cwd=str(repo), env=clean_env, check=True,
        )
        if shutil.which("condor_submit") is None:
            raise RuntimeError("condor_submit unavailable; run locally or submit from a Condor login node")
        subprocess.run(["condor_submit", str(path)], check=True)
    else:
        print(f"Review, then run: condor_submit {path}")


if __name__ == "__main__":
    main()
