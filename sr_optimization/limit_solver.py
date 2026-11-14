"""Numerically guarded median-expected CLs solver from the bbll development repo.

Used for prompt-only model comparisons; not a substitute for the final
uncertainty-complete inference model.
"""

import numpy as np
import pyhf


def _median_expected_cls(mu, data, model):
    _, expected = pyhf.infer.hypotest(
        float(mu),
        data,
        model,
        test_stat="qtilde",
        return_expected_set=True,
    )
    values = np.asarray(expected, dtype=np.float64).reshape(-1)
    if values.size < 3:
        raise RuntimeError("pyhf did not return the expected CLs set")
    value = float(values[2])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite median expected CLs at mu={mu}: {value}")
    return value


def _median_expected_limit_bisection(
    data,
    model,
    mu_max,
    level=0.05,
    rtol=1.0e-5,
    atol=1.0e-5,
    max_iter=64,
):
    """Robust fallback for the median expected CLs upper limit.

    This brackets the first finite median-CLs crossing and then uses
    deterministic bisection. It avoids rare scipy/pyhf root failures caused by
    either endpoint reversal at machine precision or evaluation of an
    excessively large POI endpoint where the asymptotic calculation becomes
    non-finite.
    """

    mu_max = float(mu_max)
    if mu_max <= 0:
        raise ValueError("mu_max must be positive")

    low = 0.0
    high = min(1.0, mu_max)
    cls_high = _median_expected_cls(high, data, model)

    while cls_high > level and high < mu_max:
        low = high
        high = min(mu_max, 2.0 * high)
        cls_high = _median_expected_cls(high, data, model)

    if cls_high > level:
        raise RuntimeError(
            f"median expected CLs did not reach {level} below mu_max={mu_max}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        cls_mid = _median_expected_cls(mid, data, model)

        if cls_mid > level:
            low = mid
        else:
            high = mid

        width = high - low
        scale = max(1.0, abs(mid))
        if width <= max(atol, rtol * scale):
            break

    return 0.5 * (low + high)


def _is_recoverable_upper_limit_error(exc):
    text = str(exc)
    if "a and b are not an interval" in text:
        return True
    if (
        "The function value at x=" in text
        and "is NaN" in text
        and "solver cannot continue" in text
    ):
        return True
    return (
        "fit initialization parameter" in text
        and "lies outside of its bounds" in text
    )


def median_expected_upper_limit(data, model, mu_max, level=0.05):
    """Return (median expected upper limit, solver name).

    The normal pyhf upper-limit routine is used first. Known numerical
    root-finding failures trigger the deterministic median-CLs bisection
    fallback:

      * endpoint reversal: ``a and b are not an interval``;
      * a non-finite function value at an excessively large POI endpoint;
      * pyhf's automatic root scan proposing a fixed-POI initialization outside
        the model's explicit POI bound.

    Other exceptions are propagated as genuine failures.
    """

    try:
        _, expected = pyhf.infer.intervals.upper_limits.upper_limit(
            data,
            model,
            scan=None,
            level=level,
        )
        return float(expected[2]), "pyhf_upper_limit"
    except ValueError as exc:
        if not _is_recoverable_upper_limit_error(exc):
            raise

        limit = _median_expected_limit_bisection(
            data,
            model,
            mu_max=mu_max,
            level=level,
        )
        return float(limit), "median_cls_bisection"
