"""Match detected episodes to ground truth scenarios.

A detected episode matches a scenario on the same column when their time spans overlap,
with the scenario span extended by a six hour recovery tail. When several scenarios
overlap, the one with the largest overlap wins.
"""
from __future__ import annotations

import pandas as pd


def overlap_hours(a0, a1, b0, b1) -> float:
    lo = max(pd.Timestamp(a0), pd.Timestamp(b0))
    hi = min(pd.Timestamp(a1), pd.Timestamp(b1))
    return max((hi - lo).total_seconds() / 3600.0, 0.0)


def match(detected: list[dict], truth: list[dict], tail_h: float = 6.0) -> dict[str, str | None]:
    out = {}
    for d in detected:
        best, best_ov = None, 0.0
        for t in truth:
            if t["column"] != d["column"]:
                continue
            ov = overlap_hours(d["start"], d["end"], t["start"], pd.Timestamp(t["end"]) + pd.Timedelta(hours=tail_h))
            if ov > best_ov:
                best, best_ov = t["scenario_id"], ov
        out[d["episode_id"]] = best
    return out
