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
MIN_BINS = int(CONF["minimum_mt_bins"])


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


def invalid_template_reasons(rows, edges):
    """Guard all three views against negative yields and low signed N_eff."""
    reasons = []
    if len(edges) - 1 < MIN_BINS:
        reasons.append("less_than_two_mt_bins")
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
                      reference_boosted=None):
    """Find common coarsening; never loosen score cuts or relax N_eff."""
    edges = np.asarray(original_edges, dtype=np.float64)
    if len(edges) < MIN_BINS + 1 or not np.all(np.diff(edges) > 0):
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
        reasons = invalid_template_reasons(rows, edges)
        if not reasons:
            return {
                "status": "MATCHED_TEMPLATES_VALID",
                "edges": edges, "rows": rows,
                "original_nbins": original_nbins,
                "coarsening_steps": history,
                "reasons": [],
            }
        if len(edges) - 1 <= MIN_BINS:
            return {
                "status": "NOT_COMPARABLE_UNDER_FROZEN_GUARDS",
                "edges": edges, "rows": rows,
                "original_nbins": original_nbins,
                "coarsening_steps": history,
                "reasons": reasons,
            }
        options = []
        for k in range(1, len(edges) - 1):
            trial = np.delete(edges, k)
            trial_rows = boosted_histograms(data, era, mass, threshold, trial)
            options.append((violation_score(trial_rows), k, trial, trial_rows))
        _, boundary, trial_edges, trial_rows = min(
            options, key=lambda x: (x[0][0], x[0][1], x[1])
        )
        history.append(float(edges[boundary]))
        edges, rows = trial_edges, trial_rows
