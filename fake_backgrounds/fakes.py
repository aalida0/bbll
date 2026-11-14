"""MC-only SS-to-OS nonprompt transfer-factor feasibility calculations.

No fake probabilities, collision data, or OS fake yields are invented.
This prototype cannot be used in the nominal SR likelihood until the
independent calibration/closure and prompt-vs-nonprompt overlap gates pass.
"""

from __future__ import annotations

import numpy as np

from bbll.config import load_stage_config

CFG = load_stage_config("fake_backgrounds")


def _finite_vector(name, values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: expected finite one-dimensional array")
    return x


def _positive_edges(edges):
    edges = _finite_vector("pt edges", edges)
    if len(edges) < 3 or np.any(np.diff(edges) <= 0):
        raise ValueError("invalid lepton-pT bin edges")
    return edges


def _require_pt_range(pt, edges):
    pt = _finite_vector("subleading pt", pt)
    if np.any((pt < edges[0]) | (pt > edges[-1])):
        raise ValueError("subleading lepton pT outside configured fake-factor range")
    return pt


def weighted_pt_bins(pt, weight, edges):
    """Sum signed MC weights and squared weights without rate clipping."""
    edges = _positive_edges(edges)
    pt = _require_pt_range(pt, edges)
    weight = _finite_vector("MC weight", weight)
    if len(pt) != len(weight):
        raise ValueError("pt/weight event counts differ")
    # Uppermost pT edge is included in final bin by np.histogram.
    return (
        np.histogram(pt, bins=edges, weights=weight)[0].astype(float),
        np.histogram(pt, bins=edges, weights=weight**2)[0].astype(float),
    )


def _neff(sumw, sumw2):
    return np.divide(
        sumw**2, sumw2, out=np.zeros_like(sumw, dtype=float),
        where=sumw2 > 0
    )


def calibrate_transfer(os_pt, os_weight, ss_pt, ss_weight,
                       edges=None, minimum_neff=None):
    """MC nonprompt OS/SS transfer factors and independent-sample variance.

    OS and SS must come from the same nonprompt truth class, process mixture
    and baseline selection with only lepton-charge condition reversed.
    The caller must ensure calibration events are disjoint from validation.
    """
    if edges is None:
        edges = CFG["subleading_lepton_pt_edges_GeV"]
    if minimum_neff is None:
        minimum_neff = CFG["minimum_transfer_neff_per_pt_bin"]
    if (not np.isfinite(minimum_neff) or minimum_neff <= 0):
        raise ValueError("minimum_neff must be finite and positive")
    edges = _positive_edges(edges)
    a, av = weighted_pt_bins(os_pt, os_weight, edges)
    b, bv = weighted_pt_bins(ss_pt, ss_weight, edges)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("nonpositive MC OS/SS nonprompt transfer-factor bin")
    if np.any(_neff(a, av) < minimum_neff) or np.any(
            _neff(b, bv) < minimum_neff):
        raise ValueError("insufficient nonprompt MC effective statistics in pt bin")
    transfer = a / b
    # Delta-method MC statistical variance; independent OS and SS samples.
    variance = av / b**2 + a**2 * bv / b**4
    if np.any(~np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("invalid transfer-factor variance")
    return {
        "pt_edges_GeV": edges.tolist(),
        "transfer_factors": transfer.tolist(),
        "transfer_variances": variance.tolist(),
        "mc_os_neff": _neff(a, av).tolist(),
        "mc_ss_neff": _neff(b, bv).tolist(),
        "mc_os_sumw": a.tolist(),
        "mc_ss_sumw": b.tolist(),
        "status": "MC_TRANSFER_FACTORS_CALCULATED_NOT_PHYSICS_VALIDATED",
    }


def _matrix(name, value, shape=None, nonnegative=False):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: requires a finite [pT,mT] matrix")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name}: incompatible [pT,mT] dimensions")
    if nonnegative and np.any(a < 0):
        raise ValueError(f"{name}: negative variance not permitted")
    return a


def predict_os_fake(transfer, ss_pseudodata, ss_prompt,
                    ss_pseudodata_sumw2, ss_prompt_sumw2):
    """Predict MC-closure fake mT shape from pseudo-SS-minus-prompt matrix.

    MC closure only: a pseudo-data SS sample must be independently generated,
    and cannot be the SAME MC events used to calibrate transfer factors.

    Covariance retains pT-factor correlations ACROSS mT bins. Independent
    poisson/pseudo-count statistics and MC prompt-subtraction variances
    are treated separately; not equivalent to a real-data fake estimate.
    """
    f = _finite_vector("transfer factors", transfer["transfer_factors"])
    var_f = _finite_vector("transfer variances",
                           transfer["transfer_variances"])
    if np.any(f <= 0) or np.any(var_f < 0) or len(f) == 0:
        raise ValueError("invalid transfer factors or variances")
    pseudodata = _matrix("SS pseudodata", ss_pseudodata)
    prompt = _matrix("SS prompt", ss_prompt, pseudodata.shape)
    dvar = _matrix("SS pseudodata sumw2", ss_pseudodata_sumw2,
                   pseudodata.shape, nonnegative=True)
    pvar = _matrix("SS prompt sumw2", ss_prompt_sumw2,
                   pseudodata.shape, nonnegative=True)
    if pseudodata.shape[0] != len(f):
        raise ValueError("wrong number of lepton-pT bins")

    fake_ss = pseudodata - prompt
    # Do not silently clip negative subtraction fluctuations to zero.
    if np.any(fake_ss < -1e-10):
        raise ValueError("negative SS prompt-subtracted fake bin: rebin or fit")
    if np.any(np.sum(fake_ss, axis=1) <= 0):
        raise ValueError("empty SS fake yield in a transfer-factor pt bin")

    per_pt_mt = f[:, None] * fake_ss
    prediction = np.sum(per_pt_mt, axis=0)
    # Diagonal SS statistical component; off-diagonal TF correlation.
    diagonal = np.sum((f[:, None]**2) * (dvar + pvar), axis=0)
    cov = np.diag(diagonal) + np.einsum(
        "p,pi,pj->ij", var_f, fake_ss, fake_ss
    )
    if np.any(~np.isfinite(cov)):
        raise ValueError("nonfinite estimated fake covariance")
    if np.any(prediction < -1e-10):
        raise ValueError("negative OS predicted fake yield")
    return {
        "status": "MC_SS_TO_OS_CLOSURE_PREDICTION_NOT_NOMINAL",
        "fake_mt_bins": prediction.tolist(),
        "fake_mt_covariance": cov.tolist(),
        "ss_fake_by_pt_mt": fake_ss.tolist(),
        "os_fake_by_pt_mt": per_pt_mt.tolist(),
        "physics_validation_complete": False,
    }


def compare_independent_closure(prediction, truth_os_sumw, truth_os_sumw2):
    """Numerical closure diagnostic, never an automatic physics approval."""
    expected = _finite_vector("predicted fake mT", prediction["fake_mt_bins"])
    truth = _finite_vector("independent true fake mT", truth_os_sumw)
    var = _finite_vector("independent truth mT sumw2", truth_os_sumw2)
    if (len(expected) != len(truth) or len(var) != len(truth)
            or np.any(var < 0)):
        raise ValueError("truth dimensions/variances invalid")
    cov = np.asarray(prediction["fake_mt_covariance"], dtype=np.float64)
    if cov.shape != (len(truth), len(truth)):
        raise ValueError("fake covariance shape mismatch")
    residual = expected - truth
    # Report per-bin pulls, not a p-value without examining correlations.
    sigma = np.sqrt(np.diag(cov) + var)
    if np.any(sigma <= 0):
        raise ValueError("no uncertainty for closure residual")
    return {
        "closure_pull_per_mt_bin": (residual / sigma).tolist(),
        "closure_ratio_integral": (
            float(np.sum(expected)/np.sum(truth))
            if float(np.sum(truth)) > 0 else None
        ),
        "closure_residual_per_mt_bin": residual.tolist(),
        "status": "INDEPENDENT_MC_CLOSURE_DIAGNOSTIC_NOT_NOMINAL",
    }


def main():
    raise SystemExit(
        "fake_backgrounds: MC transfer-factor prototype is implemented; "
        "no validated OS/SS inputs or prompt-origin labels are connected. "
        "Run 'python -m fake_backgrounds.audit --ntuple <raw_or_selected.root>' "
        "to inspect required truth/charge branches. No fake SR template is produced."
    )


if __name__ == "__main__":
    main()
