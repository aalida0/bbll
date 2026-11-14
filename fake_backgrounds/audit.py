"""Read-only audit of fake-background input ROOT schemas.

We do NOT silently derive SS control regions from OS-only Stage-1 files, nor
infer generator prompt truth from lepton PDG ID. This is a schema feasibility
check only; all output is diagnostic, not a fake yield.
"""

import argparse
import json
from pathlib import Path

from bbll.config import load_stage_config


def audit_file(path, tree_name=None):
    import uproot

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with uproot.open(path) as f:
        if tree_name is None:
            options = [
                key for key in ("Analysis_NOSYS", "events")
                if key in f
            ]
            if len(options) != 1:
                raise ValueError(
                    f"{path}: provide --tree to select input TTree; "
                    f"choices={options}"
                )
            tree_name = options[0]
        if tree_name not in f:
            raise ValueError(f"{path}: missing tree {tree_name}")
        tree = f[tree_name]
        branches = set(tree.keys())
        lower = {key.lower(): key for key in branches}

    config = load_stage_config("fake_backgrounds")
    required = config["input_contract"]["required_columns"]
    # Input schema must be explicitly mapped before creating physics templates:
    # these are normative field names, never guessed from similar names.
    missing = [column for column in required if column not in branches]
    candidates = sorted(
        key for key in branches
        if any(fragment in key.lower() for fragment in (
            "charge", "prompt", "truth", "origin", "parent", "match", "fake",
            "ss", "sign", "loose", "tight", "iso"
        ))
    )
    is_stage1 = tree_name == "events" and {
        "category3", "physics_weight", "mt_hh_mH125"
    }.issubset(branches)
    stage1_score_only = is_stage1 and "ml_score" in branches

    return {
        "schema_version": 1,
        "status": (
            "SCHEMA_HAS_REQUIRED_CANONICAL_COLUMNS_NEEDS_PHYSICS_VALIDATION"
            if not missing else "BLOCKED_MISSING_CANONICAL_CHARGE_TRUTH_FIELDS"
        ),
        "path": str(path),
        "tree": tree_name,
        "n_branches": len(branches),
        "missing_required_columns": missing,
        "relevant_available_branches": candidates,
        "is_selected_or_oof_tree": is_stage1,
        "is_oof_score_tree": stage1_score_only,
        "fake_factors_calculated": False,
        "ss_control_region_constructed": False,
        "fake_sr_template_produced": False,
        "comment": (
            "If raw branches use different names, define and verify an "
            "explicit field mapping upstream; do not assume missing "
            "truth-origin labels or charges from sample identity."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ntuple", type=Path, required=True)
    p.add_argument("--tree", help="Explicit ROOT TTree name if ambiguous")
    p.add_argument("--output", type=Path, help="Optional new JSON audit file")
    a = p.parse_args()
    result = audit_file(a.ntuple, a.tree)
    txt = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(txt)
    if a.output is not None:
        if a.output.exists():
            raise FileExistsError(a.output)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(txt)


if __name__ == "__main__":
    main()
