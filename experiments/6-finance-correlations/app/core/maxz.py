from __future__ import annotations

from typing import Iterable, Sequence


def max_abs_z(values: Iterable[float], expected_mean: float, expected_std: float) -> float:
    """Compute Max[Z] as the maximum absolute z-score across values.

    Per Scott Wilber (canon.yaml), eggs produce nominally independent streams
    with mean ~100 and variance ~50, so std ~7.071. Max[Z] over a recent window
    reflects aggregate anomaly intensity.
    """
    inv_std = 0.0 if expected_std == 0 else 1.0 / expected_std
    max_abs = 0.0
    has_any = False
    for v in values:
        try:
            z = abs((float(v) - expected_mean) * inv_std) if inv_std != 0.0 else 0.0
        except Exception:
            continue
        has_any = True
        if z > max_abs:
            max_abs = z
    return max_abs if has_any else 0.0


def max_abs_z_over_samples(
    samples: Sequence[Sequence[float]], expected_mean: float, expected_std: float
) -> float:
    """Compute Max[Z] across a 2D window of samples (time x eggs)."""
    if len(samples) == 0:
        return 0.0
    flat: list[float] = [v for row in samples for v in row]
    return max_abs_z(flat, expected_mean, expected_std)




