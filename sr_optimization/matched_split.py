"""Controlled BDT category-split ablation: identical cuts and mT bin edges.

Read validated OOF Run-2/Run-3 boosted events; reuse the resolved template.
All boosted categories use the *same* inclusive-boosted BDT threshold.
A common coarsened mT binning is imposed on inclusive and split categories.
Only the boosted split changes between likelihoods.
This is PROMPT-ONLY. Never cite as publication 95% CLs limits.
"""

from __future__ import annotations

import numpy as np

from bbll.config import analysis_weights, load_stage_config
from .scan import channel_mask, histogram, load_scores, neff

CONF = load_stage_config("sr_optimization")
MIN_NEFF = float(CONF["minimum_background_neff_per_bin"])
def configured_min_bins():
    """Separate matched-control bin policy from frozen SR-screening policy."""
    value = CONF.get("matched_split_control", {}).get(
        "minimum_mt_bins", CONF["minimum_mt_bins"]
    )
    if type(value) is not int or value not in (1, 2):
        raise ValueError(
            "sr_optimization/config.yaml: matched_split_control."
            "minimum_mt_bins must be the integer 1 or 2"
        )
    return value


def select_min_bins(override=None):
    if override is None:
        return configured_min_bins()
    if type(override) is not int or override not in (1, 2):
        raise ValueError("minimum mT bins must be 1 or 2")
    return override


def boosted_histograms(data, era, mass, threshold, edges):
    """Histogram identical selected events for inclusive and split boosted."""
    tags = np.asarray(data["category3"])
    is_signal = np.asarray(data["is_signal"], dtype=bool)
    score = np.asarray(data["ml_score"])
    mass_labels = np.asarray(data["signal_mass_GeV"])
    mt = np.asarray(data["mt_hh_mH125"])
    weights = analysis_weights(data["physics_weight"], era)
    selected = score >= float(threshold)
    sig = selected & is_signal & (mass_labels == mass)
    bkg = selected & (~is_signal)

    rows = {}
    for channel in ("boosted", "boosted_1tag", "boosted_2tag"):
        chosen = channel_mask(data, channel)
        s, s2 = histogram(mt[sig & chosen], weights[sig & chosen], edges)
        b, b2 = histogram(mt[bkg & chosen], weights[bkg & chosen], edges)
        rows[channel] = {
            "s": s, "s2": s2, "b": b, "b2": b2,
            "n_signal": int(np.count_nonzero(sig & chosen)),
            "n_background": int(np.count_nonzero(bkg & chosen)),
        }

    # Require event-level partition and signed yield/variance closure.
    inclusive = rows["boosted"]
    one, two = rows["boosted_1tag"], rows["boosted_2tag"]
    for key in ("s", "s2", "b", "b2"):
        if not np.allclose(inclusive[key], one[key] + two[key],
                           rtol=5e-9, atol=1e-8):
            raise ValueError(f"boosted tag-split closure fails for {key}")
    for key in ("n_signal", "n_background"):
        if inclusive[key] != one[key] + two[key]:
            raise ValueError(f"boosted tag-split event closure fails for {key}")
    return rows



def merge_adjacent_bins(rows, boundary):
    """Merge boundary-1 and boundary without rescanning millions of events."""
    result = {}
    for channel, row in rows.items():
        item = dict(row)
        for key in ("s", "s2", "b", "b2"):
            values = np.asarray(row[key], dtype=np.float64)
            if not (1 <= boundary < len(values)):
                raise ValueError("invalid merge boundary")
            merged = np.delete(values, boundary)
            merged[boundary - 1] = values[boundary - 1] + values[boundary]
            item[key] = merged
        result[channel] = item

    # Recheck sumw and sumw2 closure after each coarsening move.
    inc = result["boosted"]
    one, two = result["boosted_1tag"], result["boosted_2tag"]
    for key in ("s", "s2", "b", "b2"):
        if not np.allclose(inc[key], one[key] + two[key],
                           rtol=5e-9, atol=1e-8):
            raise ValueError(f"merged tag-split closure fails: {key}")
    return result


def invalid_template_reasons(rows, edges, min_bins=None):
    """Guard all three views against negative yields and low signed N_eff."""
    min_bins = select_min_bins(min_bins)
    reasons = []
    if len(edges) - 1 < min_bins:
        reasons.append(f"less_than_{min_bins}_mt_bins")
    for channel, row in rows.items():
        s, b, b2 = row["s"], row["b"], row["b2"]
        if float(np.sum(s)) <= 0:
            reasons.append(f"{channel}:nonpositive_signal")
        if np.any(s < -1e-10):
            reasons.append(f"{channel}:negative_signal_bin")
        if np.any(b <= 0):
            reasons.append(f"{channel}:nonpositive_background_bin")
        if np.any(b2 <= 0):
            reasons.append(f"{channel}:nonpositive_background_variance")
        if np.any(neff(b, b2) < MIN_NEFF):
            reasons.append(f"{channel}:low_background_neff")
    return reasons


def violation_score(rows):
    """Deterministic ranking of adjacent-bin merges; lower is better."""
    nbad = 0
    deficit = 0.0
    for row in rows.values():
        b, b2, s = row["b"], row["b2"], row["s"]
        values = neff(b, b2)
        fail = (b <= 0) | (b2 <= 0) | (values < MIN_NEFF)
        nbad += int(np.count_nonzero(fail))
        deficit += float(np.sum(np.maximum(0, MIN_NEFF - values)))
        nbad += int(np.count_nonzero(s < -1e-10))
        deficit += float(np.sum(np.maximum(0, -s)))
        if np.sum(s) <= 0:
            nbad += 100
    return nbad, deficit


def matched_templates(data, era, mass, threshold, original_edges,
                      reference_boosted=None, min_mt_bins=None):
    """Find common coarsening; never loosen score cuts or relax N_eff."""
    min_bins = select_min_bins(min_mt_bins)
    edges = np.asarray(original_edges, dtype=np.float64)
    if (len(edges) < min_bins + 1
            or not np.all(np.isfinite(edges))
            or not np.all(np.diff(edges) > 0)):
        raise ValueError("invalid starting bin edges")
    rows = boosted_histograms(data, era, mass, threshold, edges)

    if reference_boosted is not None:
        # Prove the input's optimized inclusive template was rebuilt exactly.
        for key, field in (("s", "signal_bins"),
                           ("b", "background_bins"),
                           ("b2", "background_sumw2_bins")):
            if not np.allclose(rows["boosted"][key],
                               np.asarray(reference_boosted[field], dtype=float),
                               rtol=5e-8, atol=1e-7):
                raise ValueError(f"input optimized-boosted template mismatch: {field}")

    original_nbins = len(edges) - 1
    history = []
    while True:
        reasons = invalid_template_reasons(rows, edges, min_bins=min_bins)
        if not reasons:
            return {
                "status": "MATCHED_TEMPLATES_VALID",
                "edges": edges, "rows": rows,
                "original_nbins": original_nbins,
                "min_mt_bins": min_bins,
                "coarsening_steps": history,
                "reasons": [],
            }
        if len(edges) - 1 <= min_bins:
            return {
                "status": "NOT_COMPARABLE_UNDER_CONFIGURED_GUARDS",
                "edges": edges, "rows": rows,
                "original_nbins": original_nbins,
                "coarsening_steps": history,
                "reasons": reasons,
            }
        options = []
        for k in range(1, len(edges) - 1):
            trial = np.delete(edges, k)
            trial_rows = merge_adjacent_bins(rows, k)
            options.append((violation_score(trial_rows), k, trial, trial_rows))
        _, boundary, trial_edges, trial_rows = min(
            options, key=lambda x: (x[0][0], x[0][1], x[1])
        )
        history.append(float(edges[boundary]))
        edges, rows = trial_edges, trial_rows
