"""Load the single bundled analysis configuration."""

from pathlib import Path
import yaml


def load_config():
    path = Path(__file__).with_name("analysis.yaml")
    cfg = yaml.safe_load(path.read_text())
    cuts = cfg["selection"]
    if (
        cfg["schema_version"] != 1
        or cuts["priority"] != "boosted"
        or cuts["hbb_mass_window_required"] is not False
        or cuts["boosted_min_tags"] != 1
        or cuts["categories3"] != ["resolved", "boosted_1tag", "boosted_2tag"]
        or cuts["categories2"] != ["resolved", "boosted"]
    ):
        raise ValueError("unsupported selection contract in analysis.yaml")
    if len(cfg["features"]["resolved"]) != 20 or len(cfg["features"]["boosted"]) != 25:
        raise ValueError("expected frozen 20/25 classifier features")
    return cfg
