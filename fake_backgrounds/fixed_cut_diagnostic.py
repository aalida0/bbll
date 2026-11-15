"""Non-final fake-aware expected-limit diagnostic at frozen prompt SR cuts.

Runs ONLY the previously chosen prompt-only two- and three-channel working
points. It isolates the effect of adding the full 4M Run-3 calibrated FakeSim
background before attempting a NEW fake-aware SR reoptimization.

Provisional nuisance contract:
- 50% fully correlated fake normalization from public inclusive e-mu proxy;
- correlated one-axis binwise envelope of four calibrated fake benchmarks;
- independent per-bin shapesys from prompt MC sumw2 + FakeSim second moments.

Validity gates:
- exact candidate score/bin identity, >=2 mT bins and total bkg Neff>=10;
- strictly positive total yields and both histosys extrema (no clipping);
- diagonal shapesys approximation is BLOCKED if within-channel source-event
  correlations exceed 5%; full inter-category correlations unmodeled;
- prompt/fake source disjointness must be audited before publication.

The output limit is in frozen signal-template mu units, not certified fb.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bbll.config import load_config
from sr_optimization.exhaustive_campaign import validate_mass
from sr_optimization.prompt_likelihood import ARCHITECTURES
from .full_merge import OLD_KAPPAS
from .project_merged import mass_report

REFERENCE = "dror2016_global"
FAKE_NORM_FRACTION = 0.50
MIN_TOTAL_NEFF = 10.0
MAX_ABS_OFFDIAGONAL_CORRELATION = 0.05


def build_channel(projected, min_neff=MIN_TOTAL_NEFF,
                  max_abs_correlation=MAX_ABS_OFFDIAGONAL_CORRELATION):
    name = projected["channel"]
    sig = np.asarray(projected["signal_bins"], dtype=float)
    prompt = np.asarray(projected["prompt_background_bins"], dtype=float)
    p2 = np.asarray(projected["prompt_background_sumw2_bins"], dtype=float)
    models = projected["benchmarks"]
    if set(models) != set(OLD_KAPPAS):
        raise ValueError("FakeSim benchmark variation inventory is incomplete")
    f = np.asarray(models[REFERENCE]["sumw"], dtype=float)
    cov = np.asarray(models[REFERENCE]["event_level_covariance"], dtype=float)
    other = np.asarray([models[m]["sumw"] for m in OLD_KAPPAS],dtype=float)
    if (sig.ndim != 1 or len(sig) < 2 or prompt.shape != sig.shape
            or p2.shape != sig.shape or f.shape != sig.shape
            or other.shape != (4,len(sig))
            or cov.shape != (len(sig),len(sig))
            or not all(np.all(np.isfinite(x))
                       for x in (sig,prompt,p2,f,cov,other))
            or np.any(sig < 0) or np.sum(sig) <= 0
            or np.any(prompt <= 0) or np.any(p2 <= 0)):
        raise ValueError(f"{name}: invalid prompt/fake signal and background shapes")
    if not np.allclose(cov,cov.T,rtol=1e-10,atol=1e-8):
        raise ValueError(f"{name}: source-event covariance is asymmetric")
    f2 = np.diag(cov)
    bkg = prompt+f
    variance = p2+f2
    if np.any(variance <= 0):
        raise ValueError(f"{name}: nonpositive combined background variance")
    neff = bkg*bkg/variance
    if np.any(bkg<=0) or np.any(neff<min_neff):
        raise ValueError(
            f"{name}: frozen SR bins fail fake-aware total-background Neff "
            f"or positivity: min_Neff={np.min(neff):.3g}"
        )
    sd = np.sqrt(variance)
    rho = cov / np.outer(sd,sd)
    np.fill_diagonal(rho,0.)
    max_rho = float(np.max(np.abs(rho)))
    integrated_full = float(np.sum(variance) + np.sum(cov) - np.trace(cov))
    integrated_diag = float(np.sum(variance))
    ratio = math.sqrt(integrated_full/integrated_diag) if integrated_full>0 else 0.
    if max_rho > max_abs_correlation:
        raise ValueError(
            f"{name}: fake source-event correlations exceed diagonal shapesys "
            f"gate ({max_rho:.4g}>{max_abs_correlation})"
        )
    if abs(ratio-1.) > max_abs_correlation:
        raise ValueError(
            f"{name}: integrated MC uncertainty is sensitive to off-diagonal "
            f"covariance (ratio={ratio:.4g})"
        )

    norm_lo=prompt+(1.-FAKE_NORM_FRACTION)*f
    norm_hi=prompt+(1.+FAKE_NORM_FRACTION)*f
    model_lo=prompt+np.min(other,axis=0)
    model_hi=prompt+np.max(other,axis=0)
    if any(np.any(a<=0) for a in (norm_lo,norm_hi,model_lo,model_hi)):
        raise ValueError(f"{name}: nonpositive fake histosys endpoint; "
                         "adaptive rebin or different nuisance model required")

    return {
        "channel":name,
        "signal":sig.tolist(), "background":bkg.tolist(),
        "background_sumw2":variance.tolist(),
        "fake_nominal_bins":f.tolist(),
        "fake_norm_lo":norm_lo.tolist(),"fake_norm_hi":norm_hi.tolist(),
        "fake_model_lo":model_lo.tolist(),"fake_model_hi":model_hi.tolist(),
        "minimum_total_neff":float(np.min(neff)),
        "max_abs_mc_correlation":max_rho,
        "integrated_sigma_covariance_to_diagonal_ratio":ratio,
        "prompt_total":float(np.sum(prompt)),
        "fake_total":float(np.sum(f)),
        "fake_fraction":float(np.sum(f)/np.sum(bkg)),
        "mt_edges_GeV": projected["mt_edges_GeV"],
        "candidate_index":projected["candidate_index"],
        "score_threshold":projected["score_threshold"],
    }


def workspace_for(rows, mu_max=1e5):
    import pyhf
    if not (0 < mu_max < 1e12):
        raise ValueError("invalid mu_max")
    names = [r["channel"] for r in rows]
    if len(set(names))!=len(names):
        raise ValueError("duplicate channels")
    if "boosted" in names and any(
            x in names for x in ("boosted_1tag","boosted_2tag")):
        raise ValueError("inclusive and split boosted channels overlap")
    channel_specs=[]
    observations=[]
    for r in rows:
        # The construction gate must have passed before building this spec.
        channel_specs.append({
            "name":r["channel"],
            "samples":[
                {"name":"signal","data":r["signal"],"modifiers":[
                    {"name":"mu","type":"normfactor","data":None}
                ]},
                {"name":"background","data":r["background"],"modifiers":[
                    {"name":f"mcstat_{r['channel']}","type":"shapesys",
                     "data":np.sqrt(r["background_sumw2"]).tolist()},
                    {"name":"fake_norm","type":"histosys","data":{
                        "lo_data":r["fake_norm_lo"],
                        "hi_data":r["fake_norm_hi"],
                    }},
                    {"name":"fake_transfer_model","type":"histosys","data":{
                        "lo_data":r["fake_model_lo"],
                        "hi_data":r["fake_model_hi"],
                    }},
                ]},
            ],
        })
        observations.append({"name":r["channel"],"data":r["background"]})
    spec = {
        "channels":channel_specs,
        "observations":observations,
        "measurements":[{"name":"Measurement","config":{
            "poi":"mu","parameters":[{"name":"mu","inits":[1.],
             "bounds":[[0.,float(mu_max)]]}]}}],
        "version":"1.0.0",
    }
    return pyhf.Workspace(spec)


def evaluate_fixed(root, mass, skip_fits=False, mu_max=1e5):
    root=Path(root)
    if "run3" not in load_config()["eras"]:
        raise ValueError("no Run-3 contract")
    projected=mass_report(root,mass)
    winners=validate_mass(root,"run3",mass)
    index={(r["channel"],r["candidate_index"]):r
           for r in projected["candidates"]}
    results={}
    for arch, channels in ARCHITECTURES.items():
        winning=winners["architectures"][arch]["best_candidates"]
        rows=[]
        error=None
        for ch in channels:
            ix=int(winning[ch]["index"])
            try:
                row=build_channel(index[ch,ix])
            except (KeyError,ValueError) as exc:
                error=f"{ch}: {type(exc).__name__}: {exc}"
                break
            rows.append(row)
        item={
            "prompt_only_mu95":float(
                winners["architectures"][arch]["exhaustive_best_mu95"]),
            "fake_aware_fixed_cut_mu95":None,
            "fake_over_prompt_fixed_cut_limit_ratio":None,
            "status":"BLOCKED_TEMPLATE_GATE" if error else "FIXED_CUT_TEMPLATES_PASS",
            "error":error,
            "channels":rows,
            "mu_units":"frozen template mu, not certified fb",
        }
        if not error and not skip_fits:
            try:
                import pyhf
                from sr_optimization.limit_solver import median_expected_upper_limit
                pyhf.set_backend("numpy")
                workspace=workspace_for(rows,mu_max)
                model=workspace.model(measurement_name="Measurement")
                data=workspace.data(model,include_auxdata=True)
                limit,solver=median_expected_upper_limit(data,model,mu_max)
                if not math.isfinite(limit) or limit<=0:
                    raise ValueError("nonpositive expected upper limit")
                item["fake_aware_fixed_cut_mu95"]=float(limit)
                item["fake_over_prompt_fixed_cut_limit_ratio"]=float(
                    limit/item["prompt_only_mu95"]
                )
                item["solver"]=solver
                item["status"]="COMPLETE_FIXED_CUT_FAKESIM_DIAGNOSTIC"
            except Exception as exc:
                item["status"]="BLOCKED_FIT"
                item["error"]=f"{type(exc).__name__}: {str(exc)[:600]}"
        results[arch]=item
    return {
        "schema_version":1,"status":"FAKESIM_RUN3_FIXED_PROMPT_SELECTIONS_DIAGNOSTIC",
        "mass_GeV":int(mass),"era":"run3",
        "source_4M_merged_sha256":projected["merged_source_sha256"],
        "fixed_prompt_only_thresholds":True,
        "fake_aware_reoptimization_done":False,
        "nominal_fake_model":REFERENCE,
        "fake_norm_relative_uncertainty":FAKE_NORM_FRACTION,
        "fake_transfer_model":"four_benchmark_binwise_envelope_correlated",
        "MCstat":"diagonal_after_full_source_event_covariance_audit",
        "detector_theory_systematics_included":False,
        "signal_MCstat_included":False,
        "cross_channel_fake_MC_correlations_included":False,
        "physical_fake_rate_model_validated":False,
        "source_process_overlap_audited":False,
        "signal_normalization_in_fb_certified":False,
        "architectures":results,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--mass",type=int,required=True)
    p.add_argument("--skip-fits",action="store_true")
    p.add_argument("--mu-max",type=float,default=1e5)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    doc=evaluate_fixed(a.output_root,a.mass,a.skip_fits,a.mu_max)
    dest=(a.output or a.output_root/"fake_backgrounds"/"fixed_cut_diagnostics"/
          "run3"/f"m{a.mass}_fake_aware_fixed_cut.json")
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    print(f"RUN3 m{a.mass} FAKE-AWARE FIXED CUT DIAGNOSTIC")
    for arch,item in doc["architectures"].items():
        print(f"  {arch}: status={item['status']} "
              f"prompt_mu95={item['prompt_only_mu95']:.6g} "
              f"fake_mu95={item['fake_aware_fixed_cut_mu95']} "
              f"ratio={item['fake_over_prompt_fixed_cut_limit_ratio']}")
        if item["error"]:
            print(f"    BLOCKER: {item['error']}")
    print(f"Saved {dest}")
    return doc


if __name__=="__main__":
    main()
