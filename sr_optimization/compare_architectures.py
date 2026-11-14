"""Shortlisted prompt-only expected-CLs architecture comparison.

Reads saved SR candidate JSON. No ROOT reprocessing, no fakes or detector
systematics. Numerical mu is relative to saved signal template normalization.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from bbll.config import load_config
from .scan import validate_existing_screen
from .prompt_likelihood import ARCHITECTURES, expected_mu95


def reference(row,index,limit=None,solver=None):
    out={"index":index,"channel":row["channel"],
         "efficiency_target":row["efficiency_target"],
         "score_threshold":row["score_threshold"],
         "n_mt_bins":row["n_mt_bins"],
         "min_background_neff":row["min_background_neff"]}
    if limit is not None:
        out["expected_mu95"]=limit
        out["solver"]=solver
    return out


def compare(root,era,mass,shortlist=3,mu_max=1e5):
    if era not in ("run2","run3") or mass not in load_config()["samples"]["signal_masses_GeV"]:
        raise ValueError("unsupported era or mass")
    if shortlist<1:
        raise ValueError("shortlist must be >=1")
    root=Path(root)
    source=root/"sr_optimization"/era/f"m{mass}"/"prompt_candidate_screen.json"
    screen=validate_existing_screen(source,era,mass)
    output=source.parent/f"prompt_pyhf_top{shortlist}.json"
    if output.exists():
        raise FileExistsError(output)
    singles={}; failures=[]
    channels=("resolved","boosted","boosted_1tag","boosted_2tag")
    for channel in channels:
        results=[]
        for idx,row in enumerate(screen["candidates"][channel]):
            try:
                value,solver=expected_mu95([row],mu_max)
                results.append(reference(row,idx,value,solver))
            except Exception as e:
                failures.append({"stage":"single","channel":channel,
                    "index":idx,"type":type(e).__name__,"message":str(e)[:400]})
        results.sort(key=lambda x:x["expected_mu95"])
        singles[channel]=results
        print(f"{era}/m{mass} {channel}: {len(results)}/"
              f"{len(screen['candidates'][channel])} fits",flush=True)

    architectures={}
    for name,group in ARCHITECTURES.items():
        pools=[singles[ch][:shortlist] for ch in group]
        if not all(pools):
            architectures[name]={"status":"FAILED","best":None,
                                  "n_attempted":0,"n_success":0}
            failures.append({"stage":"combined","architecture":name,
                             "message":"empty candidate shortlist"})
            continue
        best=None; attempted=0; success=0
        for combo in itertools.product(*pools):
            attempted+=1
            rows=[screen["candidates"][ch][rec["index"]]
                  for ch,rec in zip(group,combo)]
            try:
                limit,solver=expected_mu95(rows,mu_max)
                success+=1
            except Exception as e:
                failures.append({"stage":"combined","architecture":name,
                    "indices":[rec["index"] for rec in combo],
                    "type":type(e).__name__,"message":str(e)[:400]})
                continue
            if best is None or limit<best["expected_mu95"]:
                best={"expected_mu95":limit,"solver":solver,
                    "candidates":{ch:reference(row,rec["index"])
                       for ch,row,rec in zip(group,rows,combo)}}
        architectures[name]={"status":("SHORTLIST_COMPLETE" if attempted==success
                        else "INCOMPLETE"),"n_attempted":attempted,
                        "n_success":success,"best":best}
        print(f"{era}/m{mass} {name}: {success}/{attempted} fits; "
              f"best={best['expected_mu95'] if best else 'NONE'}",flush=True)

    complete=(len(failures)==0 and
        all(a["status"]=="SHORTLIST_COMPLETE" for a in architectures.values()))
    two=architectures["two_channel"]["best"]
    three=architectures["three_channel"]["best"]
    ratio=(two["expected_mu95"]/three["expected_mu95"]
           if complete and two and three else None)
    result={
        "schema_version":1,
        "status":("COMPLETE_PROMPT_ONLY_DIAGNOSTIC" if complete
                  else "INCOMPLETE_PROMPT_ONLY_DIAGNOSTIC"),
        "era":era,"mass_GeV":mass,
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "shortlist_per_channel":shortlist,
        "shortlist_exhaustive":all(shortlist>=len(screen["candidates"][ch])
                                 for ch in channels),
        "model":"prompt_only_plus_aggregate_background_MCstat",
        "fake_backgrounds_included":False,
        "systematic_uncertainties_included":False,
        "signal_MCstat_included":False,
        "mu_units":"relative to saved input signal templates; fb unverified",
        "singles":singles,"architectures":architectures,
        "two_over_three_expected_mu95_ratio":ratio,
        "fit_failures":failures}
    tmp=output.with_suffix(".json.tmp")
    if tmp.exists():
        raise FileExistsError(tmp)
    try:
        tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{result['status']}: {output}",flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era",choices=("run2","run3"),required=True)
    p.add_argument("--mass",type=int,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--shortlist",type=int,default=3)
    p.add_argument("--mu-max",type=float,default=1e5)
    a=p.parse_args()
    compare(a.output_root,a.era,a.mass,a.shortlist,a.mu_max)


if __name__=="__main__":
    main()
