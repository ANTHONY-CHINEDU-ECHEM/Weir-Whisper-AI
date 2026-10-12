"""Data quality checks run before any modelling.

The checks mirror what a plant data engineer would run on a historian extract: sampling
regularity, missing values, flatlined tags, physically impossible values, and referential
integrity of the text records (every work order and log item has an identifier, a time
and at least one column it can be tied to).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import DataStore

LIMITS = {
    "feed_flow": (0, None), "steam_flow": (0, None), "steam_valve_op": (0, 100), "sump_level": (0, 100),
    "dp_rect": (0, None), "dp_strip": (0, None), "top_analyzer": (0, 100), "bottom_analyzer": (0, 100),
}


def historian_quality(df: pd.DataFrame, step_minutes: int) -> dict:
    idx = df.index
    gaps = np.diff(idx.values).astype("timedelta64[m]").astype(int)
    out = {
        "rows": int(len(df)),
        "start": idx[0].isoformat(),
        "end": idx[-1].isoformat(),
        "irregular_steps": int(np.sum(gaps != step_minutes)),
        "missing_values": int(df.isna().sum().sum()),
        "flatlined_tags": [],
        "saturated_at_limit": [],
        "out_of_range": {},
    }
    for c in df.columns:
        s = df[c].to_numpy()
        same = np.diff(s) == 0
        run = pd.Series(same).astype(int)
        longest = int(run.groupby((run != run.shift()).cumsum()).sum().max()) if len(run) else 0
        if longest > 288 and not c.endswith("_sp"):
            # a valve pinned at 0 or 100 percent is a process symptom (saturation), not a dead sensor
            flat_vals = s[1:][same]
            if c == "steam_valve_op" and len(flat_vals) and (np.median(flat_vals) >= 99.9 or np.median(flat_vals) <= 0.1):
                out["saturated_at_limit"].append(c)
            else:
                out["flatlined_tags"].append(c)
        if c in LIMITS:
            lo, hi = LIMITS[c]
            bad = 0
            if lo is not None:
                bad += int(np.sum(s < lo))
            if hi is not None:
                bad += int(np.sum(s > hi))
            if bad:
                out["out_of_range"][c] = bad
    return out


def run_quality(store: DataStore) -> dict:
    step = int(store.settings["simulation"]["step_minutes"])
    report = {"historian": {c: historian_quality(df, step) for c, df in store.historian.items()}}
    wo = store.work_orders
    report["work_orders"] = {
        "rows": int(len(wo)),
        "duplicate_ids": int(wo["wo_id"].duplicated().sum()),
        "missing_equipment": int(wo["equipment"].isna().sum()),
        "closed_without_notes": int(((wo["status"] == "CLOSED") & (wo["closure_notes"].fillna("") == "")).sum()),
        "completed_before_created": int((wo["completed"] < wo["created"]).sum()),
    }
    items = store.log_items
    report["shift_logs"] = {
        "logs": len(store.shift_logs),
        "items": int(len(items)),
        "duplicate_item_ids": int(items["item_id"].duplicated().sum()),
        "empty_items": int((items["text"].str.strip() == "").sum()),
        "mean_items_per_log": float(items.groupby("log_id").size().mean()),
    }
    report["alarms"] = {
        "rows": int(len(store.alarms)),
        "per_column": store.alarms.groupby("column").size().to_dict(),
        "chattering_tags": store.alarms.groupby("tag").size().sort_values(ascending=False).head(5).to_dict(),
    }
    lab = store.lab
    report["lab"] = {"rows": int(len(lab)), "flagged_high": int((lab["flag"] == "HIGH").sum())}
    report["knowledge"] = {"documents": len(store.knowledge), "kinds": pd.Series([d.kind for d in store.knowledge]).value_counts().to_dict()}
    issues = sum(v["irregular_steps"] + v["missing_values"] + len(v["flatlined_tags"]) for v in report["historian"].values())
    issues += report["work_orders"]["duplicate_ids"] + report["shift_logs"]["duplicate_item_ids"]
    report["blocking_issues"] = int(issues)
    return report
