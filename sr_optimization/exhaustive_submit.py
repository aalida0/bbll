"""Submit missing exhaustive prompt-only CLs fits on UChicago AF via HTCondor.

Shards are strictly disjoint; existing verified shards are reused.
The saved top-three fits are inputs only, never overwritten.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .aggregate import ERAS, MASSES
from .exhaustive_inputs import load_inputs, merged_path, shard_path
from .exhaustive_merge import validate_shard


def calculate_shards(total, fits_per_shard=50, max_shards=32):
    if any(type(x) is not int or x < 1 for x in
           (total, fits_per_shard, max_shards)):
        raise ValueError("fit count and shard controls must be positive integers")
    return min(total, max_shards, math.ceil(total/fits_per_shard))


def pending(root, eras=ERAS, fits_per_shard=50, max_shards=32, mu_max=1e5):
    jobs, reused, plans = [], [], []
    for era in eras:
        for mass in MASSES:
            _, _, _, identity = load_inputs(root, era, mass)
            shards = calculate_shards(identity["total_combinations"],
                                      fits_per_shard, max_shards)
            plans.append((era, mass, shards, identity["total_combinations"]))
            combined = merged_path(root, era, mass)
            if combined.exists():
                # Completed outputs are only reusable if still valid for
                # current screened data and shortlisted reference.
                from .exhaustive_campaign import validate_mass
                completed = validate_mass(root, era, mass)
                if completed["mu_max"] != float(mu_max):
                    raise ValueError(
                        f"{combined}: completed mu_max does not match requested bound"
                    )
                reused.append((era, mass, "merged"))
                continue
            for index in range(shards):
                dest = shard_path(root, era, mass, index, shards)
                if dest.exists():
                    validate_shard(root, era, mass, index, shards,
                                   identity, mu_max)
                    reused.append((era, mass, index))
                else:
                    jobs.append((era, mass, index, shards))
    return jobs, reused, plans


def build_submit(repo, python, root, logs, jobs, mu_max=1e5):
    repo, python, root, logs = map(Path, (repo, python, root, logs))
    launcher = repo / "sr_optimization" / "exhaustive_worker.sh"
    for p in (repo, python, root, launcher):
        if not p.exists():
            raise FileNotFoundError(p)
    for p in (repo, python, root, logs, launcher):
        if any(c.isspace() for c in str(p)):
            raise ValueError(f"Condor path contains whitespace: {p}")
    if not jobs or len(set(jobs)) != len(jobs):
        raise ValueError("empty or duplicated shard queue")
    if not math.isfinite(mu_max) or mu_max <= 0:
        raise ValueError("invalid mu_max")
    logs.mkdir(parents=True, exist_ok=True)
    entries = "\n".join(
        f"{era} {mass} {index} {shards}"
        for era, mass, index, shards in jobs
    )
    return f"""universe = vanilla
initialdir = {repo}
executable = /bin/bash
arguments = exhaustive_worker.sh $(era) $(mass) $(shard) $(shards) {python} {repo} {root} {mu_max}
getenv = False
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {launcher}
request_cpus = 1
request_memory = 3GB
request_disk = 1GB
output = {logs}/exhaustive_$(era)_m$(mass)_s$(shard)_$(ClusterId).out
error = {logs}/exhaustive_$(era)_m$(mass)_s$(shard)_$(ClusterId).err
log = {logs}/exhaustive_$(ClusterId).log
notification = Never
queue era,mass,shard,shards from (
{entries}
)
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", choices=("run2", "run3", "both"), default="both")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--log-dir", type=Path)
    p.add_argument("--fits-per-shard", type=int, default=50)
    p.add_argument("--max-shards", type=int, default=32)
    p.add_argument("--mu-max", type=float, default=1e5)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    root, repo, py = args.output_root.resolve(), args.repo.resolve(), args.python.absolute()
    logs = (args.log_dir or (repo / "condor" / "logs")).resolve()
    eras = ERAS if args.era == "both" else (args.era,)
    jobs, reused, plans = pending(root, eras, args.fits_per_shard,
                                  args.max_shards, args.mu_max)
    print(f"Plans: {len(plans)} masses, "
          f"{sum(x[3] for x in plans)} exhaustive pyhf combinations", flush=True)
    print(f"Existing verified outputs: {len(reused)}; "
          f"missing shard jobs: {len(jobs)}", flush=True)
    for era, mass, nshards, total in plans:
        print(f"{era} m{mass}: {total} combinations / {nshards} shards",
              flush=True)
    if not jobs:
        print("No missing shards: merge completed mass points with "
              "python -m sr_optimization.exhaustive_campaign", flush=True)
        return
    spec = build_submit(repo, py, root, logs, jobs, args.mu_max)
    sub = logs / f"exhaustive_{args.era}_f{args.fits_per_shard}_m{args.max_shards}.sub"
    sub.write_text(spec)
    print(f"HTCondor submit description: {sub}", flush=True)
    if args.submit:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        subprocess.run([str(py), "-c",
                        "import pyhf; assert pyhf.__version__ == '0.7.6'"],
                       cwd=str(repo), env=env, check=True)
        if shutil.which("condor_submit") is None:
            raise RuntimeError("condor_submit is unavailable")
        subprocess.run(["condor_submit", str(sub)], check=True)
    else:
        print("Dry run only; add --submit to submit all missing shards", flush=True)


if __name__ == "__main__":
    main()
