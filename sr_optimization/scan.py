"""Screen mass-dependent SR thresholds and signed-MC-stat-safe mT binning.

This does NOT compute final 95% CLs limits: FakeSim backgrounds, correlated
systematics and statistical inference are not yet integrated. Candidate tables
are diagnostic inputs for those later validation gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from bbll.config import analysis_weights, load_config, load_stage_config

ANALYSIS = load_config()
CFG = load_stage_config("sr_optimization")
SAMPLES = ANALYSIS["samples"]
CHANNEL_TO_TOPOLOGY = {
    "resolved": "resolved",
    "boosted": "boosted",
    "boosted_1tag": "boosted",
    "boosted_2tag": "boosted",
}
COLS = (
    "ml_score", "mt_hh_mH125", "physics_weight", "is_signal",
    "signal_mass_GeV", "signal_decay_id", "category3", "category2",
)


def load_scores(root: Path, era: str, topology: str) -> dict:
    """Load complete validated OOF sample inventory, retaining sample identity."""
    if era not in ANALYSIS["eras"] or topology not in ("resolved", "boosted"):
        raise ValueError("unknown era or topology")
    folder = root / "bdt_scores" / era / topology
    reference = ANALYSIS["reference_selected_events"][era][topology]
    manifest_path = folder / "oof_campaign.json"
    validation_path = folder / "oof_validation.json"
    if not manifest_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            f"{folder}: OOF campaign and independent validation must both exist"
        )
    for file in (manifest_path, validation_path):
        manifest = json.loads(file.read_text())
        if (manifest.get("status") != "PASS"
                or manifest.get("era") != era
                or manifest.get("topology") != topology
                or int(manifest.get("events", -1)) != reference):
            raise ValueError(f"{file}: failed OOF audit or count closure")
    masses = SAMPLES["signal_masses_GeV"]
    names = (
        [f"X{m}_{era}" for m in masses]
        + [f"{s}_{era}" for s in SAMPLES["prompt_backgrounds"]]
    )
    found = {p.stem for p in folder.glob("*.root")}
    if found != set(names):
        raise ValueError(
            f"{folder}: missing={sorted(set(names)-found)} "
            f"extra={sorted(found-set(names))}"
        )
    arrays = {name: [] for name in COLS}
    n = 0
    for stem in sorted(names):
        path = folder / f"{stem}.root"
        with uproot.open(path) as f:
            if not set(COLS).issubset(f["events"].keys()):
                raise ValueError(f"{path}: missing score fields")
        for a in uproot.iterate(f"{path}:events", COLS,
                                step_size="150 MB", library="np"):
            expected = stem.startswith("X")
            if not np.all(a["is_signal"] == expected):
                raise ValueError(f"{path}: invalid sample labels")
            if not np.all(np.isfinite(a["ml_score"])):
                raise ValueError(f"{path}: invalid scores")
            if not np.all(np.isfinite(a["physics_weight"])):
                raise ValueError(f"{path}: invalid physical weights")
            n += len(a["ml_score"])
            for name in COLS:
                arrays[name].append(a[name])
    if n != reference:
        raise ValueError(f"{era}/{topology}: score rows {n} != {reference}")
    return {key: np.concatenate(value) for key, value in arrays.items()}


def channel_mask(data: dict, channel: str) -> np.ndarray:
    if channel == "resolved":
        return data["category3"] == 0
    if channel == "boosted":
        return data["category3"] >= 1
    if channel == "boosted_1tag":
        return data["category3"] == 1
    if channel == "boosted_2tag":
        return data["category3"] == 2
    raise ValueError(f"unsupported channel: {channel}")


def positive_weighted_quantile(x, q, w):
    """Frozen development CDF interpolation, using positive absolute weights."""
    if not 0 <= q <= 1:
        raise ValueError("q must lie in [0,1]")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = np.asarray(x)[ok], np.asarray(w)[ok]
    if len(x) == 0:
        raise ValueError("empty weighted quantile")
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w, dtype=np.float64)
    if cdf[-1] <= 0:
        raise ValueError("nonpositive quantile weights")
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, x))


def balanced_signal_for_cut(data: dict, mass: int, channel: str):
    sig = (
        data["is_signal"].astype(bool)
        & (data["signal_mass_GeV"] == mass)
        & channel_mask(data, channel)
    )
    decays = sorted(np.unique(data["signal_decay_id"][sig]).tolist())
    if len(decays) != 3:
        raise ValueError(f"{mass}/{channel}: expected 3 decay channels, found {decays}")
    absw = np.abs(data["physics_weight"].astype(np.float64))
    w = np.zeros(len(absw), dtype=np.float64)
    for decay in decays:
        m = sig & (data["signal_decay_id"] == decay)
        total = float(np.sum(absw[m], dtype=np.float64))
        if total <= 0 or not np.isfinite(total):
            raise ValueError(f"{mass}/{channel}: invalid signal decay cell {decay}")
        w[m] = absw[m] / total
    w[sig] /= np.sum(w[sig], dtype=np.float64)
    return w, sig


def histogram(mt, weight, edges):
    edges = np.asarray(edges, dtype=np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("nonmonotonic histogram edges")
    x = np.asarray(mt, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(w)):
        raise ValueError("nonfinite histogram inputs")
    if CFG["clip_underflow_overflow"]:
        x = np.clip(x, edges[0] + 1e-6, edges[-1] - 1e-6)
    sw, _ = np.histogram(x, bins=edges, weights=w)
    sw2, _ = np.histogram(x, bins=edges, weights=w*w)
    return sw.astype(np.float64), sw2.astype(np.float64)


def neff(sw, sw2):
    sw, sw2 = np.asarray(sw), np.asarray(sw2)
    return np.divide(sw*sw, sw2,
                     out=np.zeros_like(sw, dtype=np.float64),
                     where=sw2 > 0)


def adaptive_edges(mt, w, threshold=10.0):
    """Historical right-to-left merge, with signed-background Neff guard."""
    master = np.asarray(CFG["master_mt_edges_GeV"], dtype=np.float64)
    sw, sw2 = histogram(mt, w, master)
    desc = [float(master[-1])]
    accw = accw2 = 0.0
    for i in range(len(sw)-1, -1, -1):
        accw += float(sw[i])
        accw2 += float(sw2[i])
        if accw > 0 and float(neff(accw, accw2)) >= threshold:
            desc.append(float(master[i]))
            accw = accw2 = 0.0
    edges = np.array(sorted(set(desc)), dtype=np.float64)
    if len(edges) < 2:
        return None
    if edges[0] != master[0]:
        edges[0] = master[0]
    while True:
        sw, sw2 = histogram(mt, w, edges)
        values = neff(sw, sw2)
        bad = np.where((sw <= 0) | (values < threshold))[0]
        if not len(bad):
            return edges
        if len(edges) <= 2:
            return None
        i = int(bad[0])
        if i == 0:
            edges = np.delete(edges, 1)
        elif i == len(sw)-1:
            edges = np.delete(edges, -2)
        else:
            left = float(neff(sw[i-1] + sw[i], sw2[i-1] + sw2[i]))
            right = float(neff(sw[i] + sw[i+1], sw2[i] + sw2[i+1]))
            edges = np.delete(edges, i if left >= right else i+1)


def candidate(data, era, channel, mass, efficiency, signal_cache=None):
    # Reuse equal-decay signal weights across threshold scans of a mass/channel.
    signal_w, sig = (balanced_signal_for_cut(data, mass, channel)
                     if signal_cache is None else signal_cache)
    threshold = positive_weighted_quantile(
        data["ml_score"][sig], 1.0-efficiency, signal_w[sig]
    )
    chan = channel_mask(data, channel)
    bkg = (~data["is_signal"].astype(bool)) & chan & (data["ml_score"] >= threshold)
    s = sig & (data["ml_score"] >= threshold)
    if not np.any(s) or not np.any(bkg):
        return None, "empty_signal_or_background"
    mt_b = data["mt_hh_mH125"][bkg]
    w_b = analysis_weights(data["physics_weight"][bkg], era)
    edges = adaptive_edges(mt_b, w_b, CFG["minimum_background_neff_per_bin"])
    if edges is None:
        return None, "invalid_signed_background_binning"
    if len(edges)-1 < CFG["minimum_mt_bins"]:
        return None, "too_few_mt_bins"
    sw, sw2 = histogram(mt_b, w_b, edges)
    values = neff(sw, sw2)
    if np.any(sw <= 0) or np.any(values < CFG["minimum_background_neff_per_bin"]):
        return None, "background_neff_failure"
    mt_s = data["mt_hh_mH125"][s]
    w_s = analysis_weights(data["physics_weight"][s], era)
    sw_s, sw2_s = histogram(mt_s, w_s, edges)
    tol = 1e-12 * max(1.0, np.max(np.abs(sw_s)))
    if np.any(sw_s < -tol):
        return None, "negative_signal_bin"
    if np.sum(sw_s) <= 0:
        return None, "nonpositive_signal_yield"
    sw_s = np.maximum(sw_s, 0)
    return {
        "channel": channel, "mass_GeV": int(mass),
        "efficiency_target": float(efficiency),
        "score_threshold": float(threshold),
        "edges_GeV": edges.tolist(),
        "signal_yield": float(np.sum(sw_s)),
        "background_yield": float(np.sum(sw)),
        "signal_bins": sw_s.tolist(),
        "signal_sumw2_bins": sw2_s.tolist(),
        "background_bins": sw.tolist(),
        "background_sumw2_bins": sw2.tolist(),
        "background_neff_bins": values.tolist(),
        "min_background_neff": float(np.min(values)),
        "n_signal_selected": int(np.count_nonzero(s)),
        "n_background_selected": int(np.count_nonzero(bkg)),
        "n_mt_bins": int(len(edges)-1),
    }, "ok"


def screen(root: Path, era: str, mass: int, output: Path,
           by_topology=None):
    if int(mass) not in SAMPLES["signal_masses_GeV"]:
        raise ValueError(f"mass {mass} is not in frozen inventory")
    if output.exists():
        raise FileExistsError(f"screen output exists: {output}")
    if by_topology is None:
        by_topology = {
            "resolved": load_scores(root, era, "resolved"),
            "boosted": load_scores(root, era, "boosted"),
        }
    tables, rejections = {}, {}
    for channel, topology in CHANNEL_TO_TOPOLOGY.items():
        data = by_topology[topology]
        rows, rejected = [], {}
        signal_cache = balanced_signal_for_cut(data, mass, channel)
        for eff in CFG["signal_efficiency_grids"][channel]:
            item, reason = candidate(
                data, era, channel, mass, float(eff),
                signal_cache=signal_cache,
            )
            if item is None:
                rejected[str(eff)] = reason
            else:
                rows.append(item)
        tables[channel], rejections[channel] = rows, rejected
        print(f"{era}/{channel} mX={mass}: valid={len(rows)}, "
              f"rejected={len(rejected)}", flush=True)
    if all(not rows for rows in tables.values()):
        raise ValueError("no viable prompt-only candidate for any channel")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"screen output exists: {output}")
    result = {
        "schema_version": 1,
        "status": "PROMPT_ONLY_SCREEN_NOT_FINAL",
        "era": era, "mass_GeV": int(mass),
        "two_category_channels": ["resolved", "boosted"],
        "three_category_channels": ["resolved", "boosted_1tag", "boosted_2tag"],
        "minimum_background_neff_per_bin": CFG["minimum_background_neff_per_bin"],
        "minimum_mt_bins": CFG["minimum_mt_bins"],
        "fake_backgrounds_included": False,
        "systematic_uncertainties_included": False,
        "expected_limits_computed": False,
        "candidates": tables, "rejected": rejections,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS: saved prompt-only SR screen to {output}", flush=True)
    return result


def validate_existing_screen(path, era, mass):
    """Read-only, strict resume of an existing diagnostic mass-point JSON."""
    doc = json.loads(path.read_text())
    if (doc.get("schema_version") != 1
            or doc.get("status") != "PROMPT_ONLY_SCREEN_NOT_FINAL"
            or doc.get("era") != era
            or doc.get("mass_GeV") != int(mass)
            or doc.get("fake_backgrounds_included") is not False
            or doc.get("systematic_uncertainties_included") is not False
            or doc.get("expected_limits_computed") is not False
            or doc.get("minimum_background_neff_per_bin")
                != CFG["minimum_background_neff_per_bin"]
            or doc.get("minimum_mt_bins") != CFG["minimum_mt_bins"]
            or set(doc.get("candidates", {})) != set(CHANNEL_TO_TOPOLOGY)
            or set(doc.get("rejected", {})) != set(CHANNEL_TO_TOPOLOGY)):
        raise ValueError(f"{path}: cannot resume incompatible SR screening")
    for channel in CHANNEL_TO_TOPOLOGY:
        candidates = doc["candidates"][channel]
        rejects = doc["rejected"][channel]
        grid = CFG["signal_efficiency_grids"][channel]
        if len(candidates) + len(rejects) != len(grid):
            raise ValueError(f"{path}: incomplete threshold grid for {channel}")
        effective = []
        for row in candidates:
            if (row.get("channel") != channel
                    or row.get("mass_GeV") != int(mass)
                    or row.get("n_mt_bins", 0) < CFG["minimum_mt_bins"]
                    or row.get("min_background_neff", -1)
                        < CFG["minimum_background_neff_per_bin"]):
                raise ValueError(f"{path}: invalid candidate in {channel}")
            effective.append(float(row["efficiency_target"]))
        rejected_efficiencies = [float(key) for key in rejects]
        if (len(set(effective + rejected_efficiencies)) != len(grid)
                or set(round(x, 12) for x in effective + rejected_efficiencies)
                    != set(round(float(x), 12) for x in grid)):
            raise ValueError(f"{path}: mismatched efficiency grid for {channel}")
    return doc


def screen_all(root: Path, era: str, resume=False, masses=None):
    """Scan multiple masses while loading each large OOF topology only once."""
    if era not in ANALYSIS["eras"]:
        raise ValueError(f"unsupported era: {era}")
    all_masses = [int(m) for m in SAMPLES["signal_masses_GeV"]]
    masses = all_masses if masses is None else [int(m) for m in masses]
    if not masses or len(set(masses)) != len(masses):
        raise ValueError("mass scan requires a nonempty, unique mass inventory")
    if any(m not in all_masses for m in masses):
        raise ValueError("mass scan includes a mass outside the frozen inventory")

    targets = {
        mass: (root / "sr_optimization" / era / f"m{mass}"
               / "prompt_candidate_screen.json")
        for mass in masses
    }
    existing = {m for m, path in targets.items() if path.exists()}
    if existing and not resume:
        raise FileExistsError(
            f"SR screen outputs already exist for {era}: {sorted(existing)}; "
            "use --resume to validate and reuse these"
        )

    results = {}
    for mass in sorted(existing):
        results[mass] = validate_existing_screen(targets[mass], era, mass)
        print(f"{era}/m{mass}: validated existing prompt-only screen", flush=True)

    missing = [mass for mass in masses if mass not in existing]
    if missing:
        print(f"{era}: reading OOF ROOT scores once for {len(missing)} masses", flush=True)
        data = {
            "resolved": load_scores(root, era, "resolved"),
            "boosted": load_scores(root, era, "boosted"),
        }
        for mass in missing:
            results[mass] = screen(root, era, mass, targets[mass],
                                   by_topology=data)

    summary = {
        "schema_version": 1,
        "status": "PROMPT_ONLY_SCREEN_NOT_FINAL",
        "era": era,
        "mass_inventory_GeV": masses,
        "n_masses": len(masses),
        "fake_backgrounds_included": False,
        "systematic_uncertainties_included": False,
        "expected_limits_computed": False,
        "masses": {
            str(mass): {
                "valid": {
                    channel: len(results[mass]["candidates"][channel])
                    for channel in CHANNEL_TO_TOPOLOGY
                },
                "rejected": {
                    channel: len(results[mass]["rejected"][channel])
                    for channel in CHANNEL_TO_TOPOLOGY
                },
            }
            for mass in masses
        },
    }
    outpath = root / "sr_optimization" / era / "prompt_screen_campaign.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # A failed/incomplete campaign never overwrites the completed summary.
    tmp = outpath.with_suffix(".json.tmp")
    if tmp.exists():
        raise FileExistsError(f"unfinished screening summary: {tmp}")
    try:
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        tmp.replace(outpath)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"PASS: {era} prompt-only SR screen for {len(masses)} mass points; "
          f"summary={outpath}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", choices=("run2", "run3", "both"), required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mass", type=int)
    mode.add_argument("--all-masses", action="store_true")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="validate and reuse existing mass-point JSON outputs")
    args = p.parse_args()
    eras = ("run2", "run3") if args.era == "both" else (args.era,)
    if args.all_masses:
        for era in eras:
            screen_all(args.output_root, era, resume=args.resume)
    else:
        for era in eras:
            path = (args.output_root / "sr_optimization" / era
                    / f"m{args.mass}" / "prompt_candidate_screen.json")
            if args.resume and path.exists():
                validate_existing_screen(path, era, args.mass)
                print(f"PASS: resumed {era}/m{args.mass} prompt-only screen", flush=True)
            else:
                screen(args.output_root, era, args.mass, path)


if __name__ == "__main__":
    main()
