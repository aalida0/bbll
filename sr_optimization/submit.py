"""Submit missing two-era prompt-only CLs mass fits through UChicago HTCondor.

Outputs and input screening JSON remain on shared /data; HTCondor event logs
live on /home. Only a small shell launcher is transferred to each worker.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from bbll.config import load_config

from .aggregate import ERAS, MASSES, comparison_path, validate_comparison


def pending(root, eras=ERAS, shortlist=3):
    """Never silently overwrite incomplete/stale existing fit output."""
    jobs = []
    reused = []
    for era in eras:
        for mass in MASSES:
            screen = (Path(root) / "sr_optimization" / era
                      / f"m{mass}" / "prompt_candidate_screen.json")
            if not screen.is_file():
                raise FileNotFoundError(f"Missing prerequisite SR candidate screen: {screen}")
            output = comparison_path(root, era, mass, shortlist)
            if output.exists():
                validate_comparison(root, era, mass, shortlist)
                reused.append((era, mass))
            else:
                jobs.append((era, mass))
    return jobs, reused


def build_submit(repo, python, root, logs, jobs, shortlist=3):
    repo, python, root, logs = map(Path, (repo, python, root, logs))
    if shortlist < 1:
        raise ValueError("shortlist must be >=1")
    for path in (repo, python, root):
        if not path.exists():
            raise FileNotFoundError(path)
    launcher = repo / "sr_optimization" / "worker.sh"
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    for path in (repo, python, root, logs, launcher):
        if any(char.isspace() for char in str(path)):
            raise ValueError(f"HTCondor path contains whitespace: {path}")
    if not jobs:
        raise ValueError("no missing fit jobs to submit")
    if len(jobs) != len(set(jobs)):
        raise ValueError("duplicate fit jobs")
    if any(era not in ERAS or mass not in MASSES for era, mass in jobs):
        raise ValueError("invalid era/mass requested")
    logs.mkdir(parents=True, exist_ok=True)
    queue = "\n".join(f"{era} {mass}" for era, mass in jobs)
    return f"""universe = vanilla
initialdir = {repo}
executable = /bin/bash
arguments = worker.sh $(era) $(mass) {shortlist} {python} {repo} {root}
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {launcher}
request_cpus = 1
request_memory = 3GB
request_disk = 1GB
output = {logs}/sr_pyhf_$(era)_m$(mass)_$(ClusterId).$(Process).out
error = {logs}/sr_pyhf_$(era)_m$(mass)_$(ClusterId).$(Process).err
log = {logs}/sr_pyhf_$(ClusterId).log
notification = Never
queue era,mass from (
{queue}
)
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", choices=("run2", "run3", "both"), default="both")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--shortlist", type=int, default=3)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--log-dir", type=Path,
                   help="Condor-approved log directory; defaults to repository /home logs")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    root, repo, interpreter = (
        a.output_root.resolve(), a.repo.resolve(), a.python.absolute()
    )
    logs = ((a.log_dir if a.log_dir else repo / "condor" / "logs")
            .resolve())
    eras = ERAS if a.era == "both" else (a.era,)
    jobs, reused = pending(root, eras, a.shortlist)
    print(f"Already completed and verified: {len(reused)}; missing: {len(jobs)}",
          flush=True)
    if not jobs:
        print("PASS: all requested mass fits already exist; nothing to submit",
              flush=True)
        return
    spec = build_submit(repo, interpreter, root, logs, jobs, a.shortlist)
    path = logs / f"sr_pyhf_top{a.shortlist}_{a.era}.sub"
    path.write_text(spec)
    print(f"HTCondor submit file: {path}", flush=True)

    if a.submit:
        clean = dict(os.environ)
        clean.pop("PYTHONPATH", None)
        clean.pop("PYTHONHOME", None)
        subprocess.run(
            [str(interpreter), "-m", "bdt_training.environment"],
            cwd=str(repo), env=clean, check=True,
        )
        subprocess.run(
            [str(interpreter), "-c",
             "import pyhf; assert pyhf.__version__ == '0.7.6'"],
            cwd=str(repo), env=clean, check=True,
        )
        if shutil.which("condor_submit") is None:
            raise RuntimeError("condor_submit unavailable")
        subprocess.run(["condor_submit", str(path)], check=True)
    else:
        print(f"Review then run: condor_submit {path}", flush=True)


if __name__ == "__main__":
    main()
