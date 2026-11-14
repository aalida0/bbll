"""Run a controlled matched-threshold, matched-binning prompt-only split test.

Start with the best two-channel shortlist fit, freeze its resolved and boosted
BDT score cuts, and split the same boosted events into exclusive 1-tag and
2-tag samples. A single shared mT binning, coarsened only as necessary to
satisfy all frozen signed-N_eff guards, is applied to BOTH architectures.
This is not a final publication result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .aggregate import validate_comparison
from .matched_split import matched_templates
from .prompt_likelihood import expected_mu95, validate_template
from .scan import load_scores, validate_existing_screen


def row_from_histogram(channel, hist, edges, baseline):
    row = dict(baseline)
    row["channel"] = channel
    row["edges_GeV"] = list(map(float, edges))
    row["n_mt_bins"] = len(edges) - 1
    row["signal_bins"] = np.maximum(hist["s"], 0).tolist()
    row["signal_sumw2_bins"] = hist["s2"].tolist()
    row["background_bins"] = hist["b"].tolist()
    row["background_sumw2_bins"] = hist["b2"].tolist()
    row["signal_yield"] = float(np.sum(row["signal_bins"]))
    row["background_yield"] = float(np.sum(hist["b"]))
    row["min_background_neff"] = float(
        np.min(hist["b"] * hist["b"] / hist["b2"])
    )
    validate_template(row)
    return row


def compare_matched(root, era, mass, shortlist=3, mu_max=1e5):
    root = Path(root)
    check = validate_comparison(root, era, mass, shortlist)
    comp = json.loads(Path(check["file"]).read_text())
    src = root / "sr_optimization" / era / f"m{mass}" / "prompt_candidate_screen.json"
    screen = validate_existing_screen(src, era, mass)
    selected = comp["architectures"]["two_channel"]["best"]["candidates"]
    chosen = {}
    for channel in ("resolved", "boosted"):
        index = int(selected[channel]["index"])
        rows = screen["candidates"][channel]
        if not (0 <= index < len(rows)):
            raise ValueError("selected BDT candidate index is out of bounds")
        chosen[channel] = rows[index]
        if not np.isclose(
            chosen[channel]["score_threshold"],
            selected[channel]["score_threshold"],
            rtol=0, atol=1e-12
        ):
            raise ValueError(f"{channel}: candidate threshold reference mismatch")
    baseline_mu95, baseline_solver = expected_mu95(
        [chosen["resolved"], chosen["boosted"]], mu_max
    )
    if not np.isclose(baseline_mu95, check["two_channel_mu95"],
                      rtol=5e-5, atol=1e-6):
        raise ValueError("baseline two-channel CLs reproduction failed")

    scored = load_scores(root, era, "boosted")
    outcome = matched_templates(
        scored, era, mass, chosen["boosted"]["score_threshold"],
        chosen["boosted"]["edges_GeV"], reference_boosted=chosen["boosted"]
    )

    result = {
        "schema_version": 1, "era": era, "mass_GeV": mass,
        "status": outcome["status"],
        "model": "PROMPT_ONLY_CONTROLLED_MATCHED_THRESHOLD_AND_MT_BINNING",
        "baseline_anchor": "best two-channel shortlisted fit",
        "resolved_threshold": chosen["resolved"]["score_threshold"],
        "boosted_threshold_common_to_both_tags":
            chosen["boosted"]["score_threshold"],
        "original_boosted_mt_bins": outcome["original_nbins"],
        "common_boosted_mt_edges_GeV": outcome["edges"].tolist(),
        "coarsened_boundaries_GeV": outcome["coarsening_steps"],
        "guard_failures": outcome["reasons"],
        "baseline_two_channel_expected_mu95": check["two_channel_mu95"],
        "fakes_included": False, "detector_theory_systematics_included": False,
        "signal_MCstat_included": False,
        "screen_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "comparison_sha256": check["sha256"],
        "shortlist_per_channel": shortlist,
        "signal_normalization_in_fb_certified": False,
        "interpretation": (
            "Only boosted-category partition changes; resolved template, "
            "BDT score thresholds, and all boosted mT bin edges are shared. "
            "The background MCstat nuisance structure is channel-specific."
        ),
    }
    if outcome["status"] == "MATCHED_TEMPLATES_VALID":
        edges = outcome["edges"]
        boosted = row_from_histogram(
            "boosted", outcome["rows"]["boosted"], edges, chosen["boosted"]
        )
        one = row_from_histogram(
            "boosted_1tag", outcome["rows"]["boosted_1tag"], edges, chosen["boosted"]
        )
        two = row_from_histogram(
            "boosted_2tag", outcome["rows"]["boosted_2tag"], edges, chosen["boosted"]
        )
        a, solver2 = expected_mu95([chosen["resolved"], boosted], mu_max)
        b, solver3 = expected_mu95([chosen["resolved"], one, two], mu_max)
        ratio = a / b
        result.update({
            "status": "PASS_MATCHED_SPLIT_CONTROL",
            "matched_two_channel_expected_mu95": a,
            "matched_three_channel_expected_mu95": b,
            "two_over_three_expected_mu95_ratio": ratio,
            "percent_limit_reduction": (1 - 1 / ratio) * 100,
            "solver_two": solver2, "solver_three": solver3,
            "matched_bin_yields": {
                ch: {"signal": r["signal_bins"],
                     "background": r["background_bins"],
                     "background_sumw2": r["background_sumw2_bins"]}
                for ch, r in (("boosted", boosted), ("boosted_1tag", one),
                              ("boosted_2tag", two))
            },
        })
        print(f"Matched thresholds: two mu95={a:.8g}; "
              f"three mu95={b:.8g}; ratio={ratio:.6f}", flush=True)
    else:
        print(f"Not comparable at fixed cut + >=2 mT bins: "
              f"{outcome['reasons']}", flush=True)

    dest = root / "sr_optimization" / era / f"m{mass}" / (
        f"prompt_matched_split_top{shortlist}.json"
    )
    if dest.exists():
        raise FileExistsError(dest)
    temp = dest.with_suffix(".json.tmp")
    if temp.exists():
        raise FileExistsError(temp)
    try:
        temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Wrote matched-split diagnostic: {dest}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", choices=("run2", "run3"), required=True)
    p.add_argument("--mass", type=int, required=True)
    p.add_argument("--shortlist", type=int, default=3)
    p.add_argument("--mu-max", type=float, default=1e5)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    compare_matched(a.output_root, a.era, a.mass,
                    a.shortlist, a.mu_max)


if __name__ == "__main__":
    main()
