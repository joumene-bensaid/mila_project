"""Continual‑learning metric helpers."""

from typing import Dict, List


def cl_metrics(indiv: List[float], fused: List[float], idx: int) -> Dict[str, float]:
    return {
        "BWT": fused[idx] - indiv[idx],
        "FWT": fused[1 - idx] - indiv[1 - idx],
        "Retention%": 100 * fused[idx] / indiv[idx],
        "Transfer%": 100 * fused[1 - idx] / indiv[1 - idx] if indiv[1 - idx] else 0,
    }
