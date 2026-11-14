"""Load the single bundled analysis configuration."""

from pathlib import Path
import math
import numpy as np
import yaml


def load_config():
    path = Path(__file__).with_name("analysis.yaml")
    cfg = yaml.safe_load(path.read_text())
    if cfg["schema_version"] != 1 or set(cfg["eras"]) != {"run2", "run3"}:
        raise ValueError("unsupported global analysis contract")
    if len(cfg["features"]["resolved"]) != 20 or len(cfg["features"]["boosted"]) != 25:
        raise ValueError("expected frozen 20/25 classifier features")
    return cfg




def load_stage_config(stage):
    path = Path(__file__).resolve().parents[1] / stage / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    if cfg["schema_version"] != 1:
        raise ValueError(f"{stage}: unsupported config version")
    return cfg

def luminosity_scale(era, cfg=None):
    """Multiplicative sidecar-weight correction for physics yields, not selection."""
    cfg = load_config() if cfg is None else cfg
    settings = cfg["eras"][era]
    target = float(settings["luminosity_fb"])
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"{era}: luminosity_fb must be positive and finite")
    enabled = settings["rescale_weights"]
    if not isinstance(enabled, bool):
        raise ValueError(f"{era}: rescale_weights must be true or false")
    if not enabled:
        return 1.0
    source = settings["weights_luminosity_fb"]
    if source is None:
        raise ValueError(f"{era}: weights_luminosity_fb required to rescale")
    source = float(source)
    if not math.isfinite(source) or source <= 0:
        raise ValueError(f"{era}: weights_luminosity_fb must be positive and finite")
    return target / source


def analysis_weights(raw_weights, era, cfg=None):
    """Apply optional luminosity correction exactly once to sidecar-normalized weights.

    Use when building expected yields, histograms, and pyhf templates.
    Do not reapply the cross section or use this for Stage-1 reference closure.
    """
    values = np.asarray(raw_weights, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite input physics weights")
    return values * luminosity_scale(era, cfg)
