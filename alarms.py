"""DCS alarm journal generated from the simulated historian.

Limits are set from the fault free reference period at the start of the year, which is
how alarm limits are usually rationalised after commissioning. Alarms are debounced over
two samples and logged on the rising edge with the value and limit at the time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tags import tag_for


def _rising_edges(cond: np.ndarray, debounce: int = 2) -> np.ndarray:
    c = cond.astype(int)
    if debounce > 1:
        s = pd.Series(c).rolling(debounce, min_periods=debounce).min().fillna(0).to_numpy().astype(int)
    else:
        s = c
    edges = np.where((s[1:] == 1) & (s[:-1] == 0))[0] + 1
    return edges


def build_alarms(column: str, col: dict, hist: dict[str, np.ndarray], index: pd.DatetimeIndex, ref_samples: int, start_id: int) -> list[dict]:
    span = col["bottom_temp_design_c"] - col["top_temp_design_c"]
    ref = slice(0, ref_samples)
    dpr_ref = float(np.median(hist["dp_rect"][ref]))
    dps_ref = float(np.median(hist["dp_strip"][ref]))
    rules = [
        ("dp_rect", "LO", hist["dp_rect"] < 0.75 * dpr_ref, 0.75 * dpr_ref, "HIGH", "Rectifying section dP low"),
        ("dp_rect", "HI", hist["dp_rect"] > 1.35 * dpr_ref, 1.35 * dpr_ref, "HIGH", "Rectifying section dP high"),
        ("dp_strip", "LO", hist["dp_strip"] < 0.75 * dps_ref, 0.75 * dps_ref, "HIGH", "Stripping section dP low"),
        ("dp_strip", "HI", hist["dp_strip"] > 1.35 * dps_ref, 1.35 * dps_ref, "HIGH", "Stripping section dP high"),
        ("top_analyzer", "HI", hist["top_analyzer"] > col["top_impurity_spec"], col["top_impurity_spec"], "HIGH", "Overhead impurity above specification"),
        ("bottom_analyzer", "HI", hist["bottom_analyzer"] > col["bottom_impurity_spec"], col["bottom_impurity_spec"], "HIGH", "Bottoms impurity above specification"),
        ("sens_temp", "DEV", np.abs(hist["sens_temp"] - hist["sens_temp_sp"]) > max(0.6, 0.025 * span), max(0.6, 0.025 * span), "MEDIUM", "Sensitive tray temperature deviation"),
        ("steam_flow", "DEV", np.abs(hist["steam_flow"] - hist["steam_sp"]) > 0.15 * np.maximum(hist["steam_sp"], 0.1), 0.15, "MEDIUM", "Reboiler steam flow deviation from setpoint"),
        ("steam_valve_op", "HI", hist["steam_valve_op"] > 97.0, 97.0, "LOW", "Steam valve output high"),
        ("sump_level", "HI", hist["sump_level"] > 68.0, 68.0, "MEDIUM", "Sump level high"),
        ("sump_level", "LO", hist["sump_level"] < 32.0, 32.0, "MEDIUM", "Sump level low"),
        ("top_pressure", "HI", hist["top_pressure"] > col["pressure_design_barg"] * 1.10, col["pressure_design_barg"] * 1.10, "HIGH", "Overhead pressure high"),
        ("feed_flow", "LO", hist["feed_flow"] < 0.62 * col["feed_design_t_h"], 0.62 * col["feed_design_t_h"], "LOW", "Feed flow low"),
        ("feed_temp", "LO", hist["feed_temp"] < col["feed_temp_design_c"] - 20.0, col["feed_temp_design_c"] - 20.0, "LOW", "Feed temperature low"),
    ]
    alarms = []
    for role, kind, cond, limit, prio, msg in rules:
        tag = tag_for(column, role)
        if tag.endswith("_OP"):
            tag = tag.replace("_OP", "")
        for i in _rising_edges(cond):
            alarms.append({
                "timestamp": index[i].isoformat(), "column": column, "tag": tag, "alarm_type": kind, "priority": prio,
                "message": f"{tag} {kind} {msg}", "value": round(float(hist[role][i]), 3), "limit": round(float(limit), 3),
            })
    alarms.sort(key=lambda a: a["timestamp"])
    for k, a in enumerate(alarms):
        a["alarm_id"] = f"AL{start_id + k:07d}"
    return alarms
