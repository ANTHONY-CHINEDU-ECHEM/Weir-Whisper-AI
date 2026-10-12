"""Recurrence agent.

Links an episode to earlier episodes with the same diagnosed cause on the same physical
asset (the recurrence family), counts earlier work orders on that asset, and surfaces
historical investigations from the library that mention the asset. Across the year it
rolls families up into a recurrence register: occurrences, interval, cumulative cost and
whether the repairs actually held.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..graph.builder import asset_history
from ..simulator.scenarios import CAUSE_LABELS
from ..simulator.tags import equipment

FAMILY_ASSET = {
    "turndown": ("column", "low rate operating practice"),
    "steam_valve_stiction": ("steam_valve", "reboiler steam control valve"),
    "reboiler_fouling": ("reboiler", "reboiler"),
    "tray_damage": ("column", "column internals"),
    "pressure_rise": ("pressure_controller", "pressure control and condenser"),
    "subcooled_feed": ("preheater", "feed preheater"),
    "flooding": ("column", "high rate operating practice"),
    "foaming": ("column", "feed contamination"),
    "analyzer_fault": ("top_analyzer", "overhead analyzer"),
}


def family_key(column: str, cause: str) -> tuple[str, str, str]:
    role, desc = FAMILY_ASSET[cause]
    asset = equipment(column)[role] if role in equipment(column) else column
    suffix = {"turndown": "TURNDOWN", "tray_damage": "TRAYS", "flooding": "HIGHRATE", "foaming": "CONTAMINATION"}.get(cause, asset)
    return f"FAM{column}_{suffix}", asset, desc


class RecurrenceAgent:
    name = "recurrence"

    def __init__(self, graph, knowledge_docs):
        self.g = graph
        self.kdocs = knowledge_docs

    def for_episode(self, ep: dict, cause: str, prior_reports: list[dict]) -> dict:
        fam, asset, desc = family_key(ep["column"], cause)
        t0 = pd.Timestamp(ep["start"])
        earlier = [r for r in prior_reports if r.get("family_id") == fam and pd.Timestamp(r["start"]) < t0]
        wos = [(wid, nd) for wid, nd in asset_history(self.g, asset) if pd.Timestamp(nd["time"]) < t0]
        relevant_codes = {"steam_valve_stiction": {"STICTION"}, "reboiler_fouling": {"FOULING"}, "subcooled_feed": {"LEAK", "FOULING"},
                          "analyzer_fault": {"INSTRUMENT"}, "tray_damage": {"INSPECTION", "MECHANICAL"}, "pressure_rise": {"FOULING"}}
        codes = relevant_codes.get(cause)
        text_cue = {"steam_valve_stiction": r"hunt|stick|cycl|stiction", "reboiler_fouling": r"foul|duty", "subcooled_feed": r"preheater",
                    "analyzer_fault": r"lab|bias|reading", "tray_damage": r"tray", "pressure_rise": r"condenser|pressure",
                    "turndown": r"low rates|rate cut|turndown", "flooding": r"flood|max rate", "foaming": r"foam"}.get(cause)
        prior_wos = [(w, nd) for w, nd in wos if (codes is None or nd.get("failure_code") in codes) and nd.get("failure_code") != "PM"
                     and (text_cue is None or re.search(text_cue, nd.get("text", ""), re.I))]
        hist_docs = [d.doc_id for d in self.kdocs if d.kind == "article" and d.doc_id >= "KB014" and (asset in d.text or (cause == "turndown" and ep["column"] in d.text and "rate" in d.text))]
        return {
            "family_id": fam, "asset": asset, "asset_description": desc,
            "earlier_episodes": [r["episode_id"] for r in earlier],
            "earlier_cost_gbp": float(sum(r["impact"]["total_cost_gbp"] for r in earlier)),
            "prior_work_orders": [w for w, _ in prior_wos][-6:],
            "historical_investigations": hist_docs,
            "occurrence_number": len(earlier) + 1,
        }


def recurrence_register(reports: list[dict]) -> list[dict]:
    """Roll material episodes up into families. Minor blips (under two hours and under GBP 1,000)
    are counted separately so they cannot inflate a recurrence count."""
    fams: dict[str, list[dict]] = {}
    minor: dict[str, int] = {}
    for r in reports:
        if r.get("material", True):
            fams.setdefault(r["family_id"], []).append(r)
        else:
            minor[r["family_id"]] = minor.get(r["family_id"], 0) + 1
    out = []
    for fam, rs in fams.items():
        rs = sorted(rs, key=lambda r: r["start"])
        starts = [pd.Timestamp(r["start"]) for r in rs]
        ends = [pd.Timestamp(r["end"]) for r in rs]
        # episodes less than seven days apart are fragments of one incident, not a recurrence
        incidents = 1
        inc_starts = [starts[0]]
        for k in range(1, len(rs)):
            if (starts[k] - max(ends[:k])).total_seconds() / 86400 > 7.0:
                incidents += 1
                inc_starts.append(starts[k])
        gaps = [(b - a).total_seconds() / 86400 for a, b in zip(inc_starts, inc_starts[1:])]
        out.append({
            "family_id": fam,
            "column": rs[0]["column"],
            "cause": rs[0]["root_cause"]["cause"],
            "cause_label": CAUSE_LABELS[rs[0]["root_cause"]["cause"]],
            "asset": rs[0]["recurrence"]["asset"],
            "occurrences": incidents,
            "episode_count": len(rs),
            "episodes": [r["episode_id"] for r in rs],
            "first": rs[0]["start"], "last": rs[-1]["start"],
            "mean_interval_days": float(np.mean(gaps)) if gaps else None,
            "total_cost_gbp": float(sum(r["impact"]["total_cost_gbp"] for r in rs)),
            "total_offspec_t": float(sum(r["impact"]["offspec_t"] for r in rs)),
            "total_excess_steam_t": float(sum(r["impact"]["excess_steam_t"] for r in rs)),
            "repair_work_orders": sorted({w for r in rs for w in r["recurrence"]["prior_work_orders"]}),
            "recurring": incidents >= 2,
            "minor_episodes": minor.get(fam, 0),
        })
    out.sort(key=lambda f: -f["total_cost_gbp"])
    return out
