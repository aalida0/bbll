"""Stage 1: select bbll events once, retaining both category definitions.

Input: Analysis_NOSYS in a ROOT ntuple and its normalized-weight sidecar.
Output: one selected ROOT tree with frozen ML inputs and both category IDs.
No ATLAS-specific paths or batch-system dependencies are assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config

CONFIG = load_config()
CUTS = CONFIG["selection"]
RESOLVED = tuple(CONFIG["features"]["resolved"])
BOOSTED = tuple(CONFIG["features"]["boosted"])
META = (
    "event_uid", "sample_class", "is_signal", "signal_mass_GeV",
    "signal_decay_id", "mt_hh_mH125", "n_tags_inside_largeR",
)
FLAGS = tuple(dict.fromkeys((
    CUTS["common"], CUTS["resolved_candidate"], *CUTS["resolved_b_tags"],
    CUTS["boosted_candidate"], CUTS["trimmed_candidate"],
    "min_dR_lepton_bjet_valid",
)))
BRANCHES = tuple(dict.fromkeys(RESOLVED + BOOSTED + META + FLAGS))

# category3 = -1 (unselected), 0 (resolved), 1 (boosted 1 tag),
#             2 (boosted >=2 tags).
# category2 = -1 (unselected), 0 (resolved), 1 (boosted >=1 tag).
CAT3 = {"resolved": 0, "boosted_1tag": 1, "boosted_2tag": 2}
CAT2 = {"resolved": 0, "boosted": 1}


def categories(a: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Exclusive boosted-first topology; identical selected events in both schemes."""
    common = np.asarray(a[CUTS["common"]], dtype=bool)
    tags = np.asarray(a[CUTS["boosted_n_tags"]])
    boosted = (
        common & np.asarray(a[CUTS["boosted_candidate"]], dtype=bool)
        & np.asarray(a[CUTS["trimmed_candidate"]], dtype=bool) & (tags >= 1)
    )
    resolved = (
        common & ~boosted
        & np.asarray(a[CUTS["resolved_candidate"]], dtype=bool)
        & np.asarray(a[CUTS["resolved_b_tags"][0]], dtype=bool)
        & np.asarray(a[CUTS["resolved_b_tags"][1]], dtype=bool)
    )

    three = np.full(len(common), -1, dtype=np.int8)
    three[resolved] = CAT3["resolved"]
    three[boosted & (tags == 1)] = CAT3["boosted_1tag"]
    three[boosted & (tags >= 2)] = CAT3["boosted_2tag"]

    two = np.full(len(common), -1, dtype=np.int8)
    two[resolved] = CAT2["resolved"]
    two[boosted] = CAT2["boosted"]
    return three, two


def reduce_sample(
    source: Path, weight_source: Path, output: Path, chunk_size: int = 250_000
) -> dict:
    """Reduce one sample, preserving signed normalized weights and event order."""
    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install -e .") from exc
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if output.exists() or output.with_suffix(".selection.json").exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".root.tmp")
    if tmp.exists():
        raise FileExistsError(f"temporary output already exists: {tmp}")

    counts = {"input": 0, "common": 0, "resolved": 0,
              "boosted_1tag": 0, "boosted_2tag": 0}
    signed_sumw = {name: 0.0 for name in CAT3}
    selected_total = 0
    try:
        with uproot.open(source) as f, uproot.open(weight_source) as wf:
            if CONFIG["input"]["nominal_tree"] not in f or CONFIG["input"]["weight_tree"] not in wf:
                raise ValueError("missing Analysis_NOSYS or Weights tree")
            tree, weights = f[CONFIG["input"]["nominal_tree"]], wf[CONFIG["input"]["weight_tree"]]
            if CONFIG["input"]["weight_branch"] not in weights.keys():
                raise ValueError("missing Weights/weight_nominal")
            if tree.num_entries != weights.num_entries:
                raise ValueError("nominal/weight entry counts differ")
            missing = sorted(set(BRANCHES) - set(tree.keys()))
            if missing:
                raise ValueError(f"missing Analysis_NOSYS branches: {missing}")

            with uproot.recreate(tmp) as dest:
                outtree = None
                for first in range(0, tree.num_entries, chunk_size):
                    stop = min(first + chunk_size, tree.num_entries)
                    a = tree.arrays(BRANCHES, entry_start=first,
                                    entry_stop=stop, library="np")
                    w = weights[CONFIG["input"]["weight_branch"]].array(
                        entry_start=first, entry_stop=stop, library="np"
                    ).astype(np.float64)
                    if not np.all(np.isfinite(w)):
                        raise ValueError("non-finite normalized event weight")
                    three, two = categories(a)
                    chosen = three >= 0
                    if not np.array_equal(chosen, two >= 0):
                        raise AssertionError("two-/three-category selection differs")

                    counts["input"] += len(three)
                    counts["common"] += int(np.count_nonzero(a[CUTS["common"]]))
                    for label, cat_id in CAT3.items():
                        in_cat = three == cat_id
                        counts[label] += int(np.count_nonzero(in_cat))
                        signed_sumw[label] += float(np.sum(w[in_cat], dtype=np.float64))
                    if not np.any(chosen):
                        continue

                    data = {key: np.asarray(a[key][chosen])
                            for key in dict.fromkeys(RESOLVED + BOOSTED + META)}
                    # Missing geometric associations are represented by NaN.
                    invalid = ~np.asarray(
                        a["min_dR_lepton_bjet_valid"][chosen], dtype=bool
                    )
                    dr = data["min_dR_lepton_bjet"].astype(np.float32, copy=True)
                    dr[invalid] = np.nan
                    data["min_dR_lepton_bjet"] = dr
                    if not np.all(np.isfinite(data["mt_hh_mH125"])):
                        raise ValueError("non-finite selected mt_hh_mH125")
                    data["physics_weight"] = w[chosen]
                    data["train_weight_base"] = np.abs(w[chosen])
                    # Stable entry index for later nominal/variation alignment.
                    data["source_entry"] = np.arange(first, stop, dtype=np.int64)[chosen]
                    data["category3"] = three[chosen]
                    data["category2"] = two[chosen]
                    if outtree is None:
                        outtree = dest.mktree(
                            "events", {k: v.dtype for k, v in data.items()}
                        )
                    outtree.extend(data)
                    selected_total += int(np.count_nonzero(chosen))
                if outtree is None:
                    raise ValueError("sample has no selected events")
        if selected_total != sum(counts[name] for name in CAT3):
            raise AssertionError("selected event count does not close")
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    report = {
        "input": str(source), "weights": str(weight_source),
        "output": str(output), "counts": counts,
        "selected": selected_total, "sumw_by_category3": signed_sumw,
        "two_categories": ["resolved", "boosted"],
        "three_categories": list(CAT3),
        "category_closure": True,
        "no_hbb_mass_window": True,
    }
    output.with_suffix(".selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input nominal ROOT ntuple")
    parser.add_argument("output", type=Path, help="selected output ROOT ntuple")
    parser.add_argument("--weights", type=Path, help="normalized weight sidecar")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()
    weights = args.weights or args.input.with_name(
        args.input.stem + CONFIG["input"]["weight_suffix"]
    )
    report = reduce_sample(args.input, weights, args.output, args.chunk_size)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
