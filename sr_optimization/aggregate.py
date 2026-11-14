"""Validate and summarize complete prompt-only two/three-channel CLs comparisons.

Read-only on mass-point fits. A campaign summary is written only after all
frozen mass points in the requested era(s) have completed without fit failures.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from bbll.config import load_config

from .prompt_likelihood import ARCHITECTURES


MASSES = tuple(int(m) for m in load_config()["samples"]["signal_masses_GeV"])
ERAS = ("run2", "run3")


def comparison_path(root, era, mass, shortlist):
    return Path(root) / "sr_optimization" / era / f"m{mass}" / f"prompt_pyhf_top{shortlist}.json"


def validate_comparison(root, era, mass, shortlist):
    """A completed result must match the current saved SR-screen SHA-256."""
    if era not in ERAS or mass not in MASSES or shortlist < 1:
        raise ValueError("Invalid era/mass/shortlist")
    path = comparison_path(root, era, mass, shortlist)
    if not path.is_file():
        raise FileNotFoundError(path)
    doc = json.loads(path.read_text())
    src = path.parent / "prompt_candidate_screen.json"
    if not src.is_file():
        raise FileNotFoundError(src)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if (doc.get("schema_version") != 1
            or doc.get("status") != "COMPLETE_PROMPT_ONLY_DIAGNOSTIC"
            or doc.get("era") != era
            or doc.get("mass_GeV") != mass
            or doc.get("source_sha256") != sha
            or doc.get("shortlist_per_channel") != shortlist
            or doc.get("fake_backgrounds_included") is not False
            or doc.get("systematic_uncertainties_included") is not False
            or doc.get("signal_MCstat_included") is not False
            or doc.get("fit_failures")):
        raise ValueError(f"{path}: invalid/incomplete or stale prompt-only CLs fit")

    arch = doc.get("architectures", {})
    if set(arch) != set(ARCHITECTURES):
        raise ValueError(f"{path}: missing/extra architectures")
    best = {}
    for name, channels in ARCHITECTURES.items():
        row = arch[name]
        fit = row.get("best")
        if (row.get("status") != "SHORTLIST_COMPLETE"
                or row.get("n_attempted", 0) <= 0
                or row.get("n_attempted") != row.get("n_success")
                or not isinstance(fit, dict)
                or set(fit.get("candidates", {})) != set(channels)):
            raise ValueError(f"{path}: {name} incomplete fit or wrong channels")
        value = float(fit["expected_mu95"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: {name} invalid expected mu95")
        best[name] = value
    ratio = best["two_channel"] / best["three_channel"]
    if (not math.isfinite(ratio)
            or not math.isclose(ratio, float(doc["two_over_three_expected_mu95_ratio"]),
                                rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError(f"{path}: malformed architecture ratio")
    return {
        "mass_GeV": mass,
        "two_channel_mu95": best["two_channel"],
        "three_channel_mu95": best["three_channel"],
        "ratio_two_over_three": ratio,
        "percent_limit_reduction": (1.0 - 1.0 / ratio) * 100.0,
        "shortlist_exhaustive": doc.get("shortlist_exhaustive") is True,
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def aggregate(root, eras=ERAS, shortlist=3):
    records = []
    missing = []
    for era in eras:
        if era not in ERAS:
            raise ValueError(f"unexpected era {era}")
        for mass in MASSES:
            path = comparison_path(root, era, mass, shortlist)
            if not path.exists():
                missing.append(str(path))
                continue
            item = validate_comparison(root, era, mass, shortlist)
            records.append({"era": era, **item})
    if missing:
        raise FileNotFoundError(
            f"CLs campaign incomplete: missing {len(missing)} files; "
            f"first missing: {missing[0]}"
        )
    result = {
        "schema_version": 1,
        "status": "COMPLETE_PROMPT_ONLY_DIAGNOSTIC",
        "description": "Two vs three exclusive-channel Asimov median expected CLs; "
                       "prompt-only, aggregate background MCstat shapesys",
        "eras": list(eras),
        "masses_GeV": list(MASSES),
        "shortlist_per_channel": shortlist,
        "final_physics_limits": False,
        "fake_backgrounds_included": False,
        "detector_theory_systematics_included": False,
        "signal_mcstat_included": False,
        "signal_normalization_in_fb_certified": False,
        "results": records,
    }
    out = Path(root) / "sr_optimization" / f"prompt_pyhf_top{shortlist}_campaign.json"
    if out.exists():
        raise FileExistsError(f"existing summary: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".json.tmp")
    if temp.exists():
        raise FileExistsError(temp)
    try:
        temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)

    print("era    mass   two_mu95    three_mu95   ratio   reduction%   exhaustive")
    for record in records:
        print(f"{record['era']:4s} {record['mass_GeV']:5d} "
              f"{record['two_channel_mu95']:11.5g} "
              f"{record['three_channel_mu95']:12.5g} "
              f"{record['ratio_two_over_three']:7.4f} "
              f"{record['percent_limit_reduction']:11.2f} "
              f"{record['shortlist_exhaustive']}")
    print(f"PASS: {len(records)} prompt-only CLs comparisons summarized: {out}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", choices=("run2", "run3", "both"), default="both")
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--shortlist", type=int, default=3)
    args = p.parse_args()
    eras = ERAS if args.era == "both" else (args.era,)
    aggregate(args.output_root, eras, args.shortlist)


if __name__ == "__main__":
    main()
