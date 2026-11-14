"""Print chosen BDT score thresholds from shortlisted or exhaustive CLs fits.

There are distinct thresholds per mass, era, channel, and architecture.
The saved threshold is ml_score >= score_threshold.
"""

import argparse
import csv
import json
from pathlib import Path

from .aggregate import ERAS, MASSES, validate_comparison
from .exhaustive_campaign import validate_mass
from .prompt_likelihood import ARCHITECTURES


def chosen_rows(root, eras=ERAS, source="top3"):
    if source not in ("top3", "exhaustive"):
        raise ValueError("source must be top3 or exhaustive")
    result = []
    for era in eras:
        for mass in MASSES:
            if source == "top3":
                info = validate_comparison(root, era, mass, 3)
                document = json.loads(Path(info["file"]).read_text())
                best = {
                    arch: document["architectures"][arch]["best"]["candidates"]
                    for arch in ARCHITECTURES
                }
                limits = {
                    arch: document["architectures"][arch]["best"]["expected_mu95"]
                    for arch in ARCHITECTURES
                }
            else:
                document = validate_mass(root, era, mass)
                best = {
                    arch: document["architectures"][arch]["best_candidates"]
                    for arch in ARCHITECTURES
                }
                limits = {
                    arch: document["architectures"][arch]["exhaustive_best_mu95"]
                    for arch in ARCHITECTURES
                }
            for architecture, channels in ARCHITECTURES.items():
                for channel in channels:
                    item = best[architecture][channel]
                    result.append({
                        "era": era, "mass_GeV": mass,
                        "architecture": architecture,
                        "channel": channel,
                        "score_threshold": float(item["score_threshold"]),
                        "signal_efficiency_target": float(item["efficiency_target"]),
                        "expected_mu95": float(limits[architecture]),
                        "optimization": source,
                    })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", choices=("run2", "run3", "both"), default="both")
    p.add_argument("--source", choices=("top3", "exhaustive"), default="top3")
    p.add_argument("--architecture",
                   choices=("two_channel", "three_channel", "both"),
                   default="both")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--csv", type=Path,
                   help="Optional output CSV; never overwritten")
    a = p.parse_args()
    eras = ERAS if a.era == "both" else (a.era,)
    rows = chosen_rows(a.output_root, eras, a.source)
    if a.architecture != "both":
        rows = [x for x in rows if x["architecture"] == a.architecture]

    for era in eras:
        for architecture, channels in ARCHITECTURES.items():
            if a.architecture not in ("both", architecture):
                continue
            print(f"\n{era.upper()} {architecture}: score thresholds "
                  f"({a.source} scan)")
            print(f"{'mass':>6} " + " ".join(f"{ch:>18}" for ch in channels))
            for mass in MASSES:
                chosen = {x["channel"]: x for x in rows if
                          x["era"] == era and x["mass_GeV"] == mass
                          and x["architecture"] == architecture}
                print(f"{mass:6d} " +
                      " ".join(f"{chosen[ch]['score_threshold']:18.9f}"
                               for ch in channels))
    if a.csv is not None:
        if a.csv.exists():
            raise FileExistsError(a.csv)
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with a.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "era", "mass_GeV", "architecture", "channel",
                "score_threshold", "signal_efficiency_target",
                "expected_mu95", "optimization"
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {a.csv}")


if __name__ == "__main__":
    main()
