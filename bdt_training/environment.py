"""Reference-environment check for reproducible BDT model training."""

from __future__ import annotations

import sys

import numpy as np
import sklearn

from bbll.config import load_stage_config


def verify_runtime(python=None, numpy_version=None, sklearn_version=None, paths=None):
    """Reject the mixed Python/Key4hep environment before loading any ROOT file."""
    expected = load_stage_config("bdt_training")["runtime"]
    version = python or f"{sys.version_info.major}.{sys.version_info.minor}"
    numpy_version = numpy_version or np.__version__
    sklearn_version = sklearn_version or sklearn.__version__
    paths = sys.path if paths is None else paths

    errors = []
    if version != expected["python"]:
        errors.append(f"Python {version}, expected {expected['python']}")
    if numpy_version != expected["numpy"]:
        errors.append(f"NumPy {numpy_version}, expected {expected['numpy']}")
    if sklearn_version != expected["scikit_learn"]:
        errors.append(f"scikit-learn {sklearn_version}, expected {expected['scikit_learn']}")
    if any("key4hep" in str(p).lower() for p in paths):
        errors.append("Key4hep paths leaked into sys.path")

    if errors:
        raise RuntimeError(
            "BDT training environment is not reproducible: "
            + "; ".join(errors)
            + ". Build an isolated Python 3.9 virtual environment "
              "without inherited PYTHONPATH/PYTHONHOME."
        )
    return {
        "python": version,
        "numpy": numpy_version,
        "scikit_learn": sklearn_version,
        "executable": sys.executable,
    }


def main():
    print(verify_runtime())


if __name__ == "__main__":
    main()
