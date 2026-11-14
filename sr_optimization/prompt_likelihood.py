"""Prompt-only HistFactory likelihood for exclusive bbll channels.

This model contains only aggregate prompt-background MC-stat shapesys. It is
NOT publication inference (fakes and other systematics are absent).
"""

import numpy as np
from bbll.config import load_stage_config

CFG=load_stage_config("sr_optimization")
ARCHITECTURES={
    "two_channel": ("resolved", "boosted"),
    "three_channel": ("resolved", "boosted_1tag", "boosted_2tag"),
}

def validate_template(row, min_mt_bins=None):
    # The published SR screen uses CFG minimum=2. The controlled split
    # diagnostic can explicitly allow one boosted cut-and-count bin.
    if min_mt_bins is None:
        min_mt_bins = CFG["minimum_mt_bins"]
    if type(min_mt_bins) is not int or min_mt_bins not in (1, 2):
        raise ValueError("min_mt_bins must be an integer 1 or 2")
    s=np.asarray(row["signal_bins"],dtype=float)
    b=np.asarray(row["background_bins"],dtype=float)
    b2=np.asarray(row["background_sumw2_bins"],dtype=float)
    edges=np.asarray(row["edges_GeV"],dtype=float)
    if (len(s)<min_mt_bins or len(s)!=len(b)
        or len(s)!=len(b2) or len(edges)!=len(s)+1
        or not all(np.all(np.isfinite(x)) for x in (s,b,b2,edges))
        or np.any(s<0) or np.any(b<=0) or np.any(b2<=0)
        or np.sum(s)<=0 or np.any(np.diff(edges)<=0)
        or np.any((b*b/b2)<CFG["minimum_background_neff_per_bin"]-1e-9)):
        raise ValueError("invalid or statistically unsafe template")
    if not np.isclose(np.sum(s),row["signal_yield"],rtol=1e-6):
        raise ValueError("signal yield mismatch")
    if not np.isclose(np.sum(b),row["background_yield"],rtol=1e-6):
        raise ValueError("background yield mismatch")
    return s,b,b2

def workspace_for(rows,mu_max=1e5,min_mt_bins=None):
    import pyhf
    names=[r["channel"] for r in rows]
    if len(names)!=len(set(names)):
        raise ValueError("duplicate likelihood channel")
    if "boosted" in names and any(c in names for c in ("boosted_1tag","boosted_2tag")):
        raise ValueError("inclusive and split boosted categories overlap")
    if not (0<mu_max<1e12):
        raise ValueError("invalid mu_max")
    channels=[]; observations=[]
    for row in rows:
        name=row["channel"]
        if name not in ("resolved","boosted","boosted_1tag","boosted_2tag"):
            raise ValueError("unknown channel")
        s,b,b2=validate_template(row,min_mt_bins=min_mt_bins)
        channels.append({"name":name,"samples":[
            {"name":"signal","data":s.tolist(),"modifiers":[
                {"name":"mu","type":"normfactor","data":None}]},
            {"name":"background","data":b.tolist(),"modifiers":[
                {"name":f"mcstat_{name}","type":"shapesys",
                 "data":np.sqrt(b2).tolist()}]}]})
        observations.append({"name":name,"data":b.tolist()})
    return pyhf.Workspace({
        "channels":channels,"observations":observations,
        "measurements":[{"name":"Measurement","config":{
            "poi":"mu","parameters":[{"name":"mu","inits":[1.],
             "bounds":[[0.,float(mu_max)]]}]}}],
        "version":"1.0.0"})

def expected_mu95(rows,mu_max=1e5,min_mt_bins=None):
    import pyhf
    from .limit_solver import median_expected_upper_limit
    pyhf.set_backend("numpy")
    workspace=workspace_for(rows,mu_max,min_mt_bins=min_mt_bins)
    model=workspace.model(measurement_name="Measurement")
    data=workspace.data(model,include_auxdata=True)
    limit,solver=median_expected_upper_limit(data,model,mu_max)
    if not (np.isfinite(limit) and limit>0):
        raise ValueError("nonfinite expected upper limit")
    return float(limit),solver
