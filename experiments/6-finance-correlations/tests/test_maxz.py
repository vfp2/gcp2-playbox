from __future__ import annotations

from app.core.maxz import max_abs_z, max_abs_z_over_samples


def test_max_abs_z_basic() -> None:
    vals = [100.0, 107.0712, 92.9288]
    score = max_abs_z(vals, expected_mean=100.0, expected_std=7.0712)
    # 107.0712 -> z ~ 1.0; 92.9288 -> z ~ -1.0 => max abs 1.0
    assert abs(score - 1.0) < 1e-3


def test_max_abs_z_over_samples() -> None:
    samples = [
        [100.0, 100.0],
        [107.0712, 92.9288],
    ]
    score = max_abs_z_over_samples(samples, 100.0, 7.0712)
    assert abs(score - 1.0) < 1e-3




