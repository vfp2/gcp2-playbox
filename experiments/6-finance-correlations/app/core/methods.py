from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

from .maxz import max_abs_z_over_samples


MethodFunc = Callable[[Sequence[Sequence[float]]], float]


@dataclass
class MethodSpec:
    key: str
    compute: MethodFunc
    label: str


def build_registry(expected_mean: float, expected_std: float) -> Dict[str, MethodSpec]:
    def maxz(samples: Sequence[Sequence[float]]) -> float:
        return max_abs_z_over_samples(samples, expected_mean, expected_std)

    return {
        "maxz": MethodSpec(key="maxz", compute=maxz, label="Max[Z]"),
    }




