"""Luminosity corrections act only when converting stored weights into yields."""

from copy import deepcopy

import numpy as np
import pytest

from bbll.config import analysis_weights, load_config, luminosity_scale


def test_run3_lumi_scale_on_and_off_without_mutating_raw_weights():
    cfg = load_config()
    raw = np.array([318.0, -159.0, 0.0], dtype=np.float64)
    preserved = raw.copy()

    assert luminosity_scale("run3", cfg) == 308.0 / 318.0
    np.testing.assert_allclose(analysis_weights(raw, "run3", cfg),
                               [308.0, -154.0, 0.0])
    np.testing.assert_array_equal(raw, preserved)

    toggled = deepcopy(cfg)
    toggled["eras"]["run3"]["rescale_weights"] = False
    assert luminosity_scale("run3", toggled) == 1.0
    np.testing.assert_array_equal(analysis_weights(raw, "run3", toggled), raw)

    # Run 2 is explicitly left unchanged until its weight normalization
    # luminosity has been verified.
    assert luminosity_scale("run2", cfg) == 1.0


def test_luminosity_correction_rejects_unphysical_configuration():
    cfg = load_config()
    invalid = deepcopy(cfg)
    invalid["eras"]["run2"]["rescale_weights"] = True
    with pytest.raises(ValueError, match="weights_luminosity_fb"):
        luminosity_scale("run2", invalid)

    invalid = deepcopy(cfg)
    invalid["eras"]["run3"]["weights_luminosity_fb"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        luminosity_scale("run3", invalid)

    invalid = deepcopy(cfg)
    invalid["eras"]["run3"]["rescale_weights"] = "false"
    with pytest.raises(ValueError, match="true or false"):
        luminosity_scale("run3", invalid)

    with pytest.raises(ValueError, match="non-finite"):
        analysis_weights([np.nan], "run3", cfg)
