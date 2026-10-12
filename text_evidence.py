"""Records analyst agent (shift logs, work orders, actions, scans, MOCs, lab).

Collects every record the knowledge graph attaches to the episode window, adds the top
hits from the hybrid retriever, extracts symptom families with the entity linker, and
converts structured cues (work order failure codes, operator actions, gamma scan findings,
MOC titles) into cause contributions. Text is weighted below hard signals because shift
notes are noisy and sometimes blame the wrong cause.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from ..graph.builder import SYMPTOM_CAUSE_WEIGHTS, evidence_for_episode
from ..nlp.entities import EntityLinker
from ..simulator.tags import equipment
from .base import EvidenceIndex
from .signal_analyst import Observation

TEXT_KINDS = ("shift_log", "work_order", "operator_action", "lab", "gamma_scan", "moc")


class RecordsAnalyst:
    name = "records_analyst"

    def __init__(self, graph, index, linker: EntityLinker, cfg: dict):
        self.g = graph
        self.index = index
        self.linker = linker
        self.cfg = cfg

    def collect(self, ep: dict, evidence: EvidenceIndex, query: str) -> list[tuple[str, dict]]:
        recs = {}
        for node, nd, w in evidence_for_episode(self.g, ep["episode_id"], TEXT_KINDS + ("alarm",)):
            if nd["kind"] == "alarm":
                continue
            recs[node] = nd
        t0, t1 = pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"])
        hits = self.index.search(query, column=ep["column"], t0=t0, t1=t1, k=int(self.cfg["top_k"]), mode="full",
                                 lookback_h=self.cfg["lookback_hours"], lookahead_h=self.cfg["lookahead_hours"],
                                 kinds=list(TEXT_KINDS))
        for h in hits:
            if h.doc.doc_id in self.g and h.doc.doc_id not in recs:
                nd = self.g.nodes[h.doc.doc_id]
                # retrieved records outside the window can inform history but not this episode's cause
                if nd.get("time") is not None:
                    tv = pd.Timestamp(nd["time"])
                    if t0 - pd.Timedelta(hours=self.cfg["lookback_hours"]) <= tv <= t1 + pd.Timedelta(hours=self.cfg["lookahead_hours"]):
                        recs[h.doc.doc_id] = nd
        out = sorted(recs.items(), key=lambda kv: pd.Timestamp(kv[1]["time"]).value)
        for node, nd in out:
            t_end = nd.get("time_end")
            if nd["kind"] == "work_order" and (t_end is None or pd.isna(t_end)):
                t_end = pd.Timestamp("2100-01-01")  # still open
            evidence.add(node, nd["kind"], nd.get("text", nd.get("label", "")), nd.get("time"),
                         time_end=pd.Timestamp(t_end).isoformat() if t_end is not None and not pd.isna(t_end) else None)
        return out

    def analyse(self, ep: dict, records: list[tuple[str, dict]]) -> list[Observation]:
        column = ep["column"]
        eq = equipment(column)
        sym_ids: dict[str, list[str]] = defaultdict(list)
        cues: list[Observation] = []
        for node, nd in records:
            text = nd.get("text", "")
            ex = self.linker.extract(text)
            if ex.columns and column not in ex.columns and nd["kind"] == "shift_log":
                continue  # a note about a neighbouring column in the same area
            for s in ex.symptoms:
                sym_ids[s].append(node)
            kind = nd["kind"]
            if kind == "work_order":
                code = nd.get("failure_code", "")
                equip = nd.get("equipment", "")
                if code == "STICTION" or (equip == eq["steam_valve"] and re.search(r"hunt|stick|cycl", text, re.I)):
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} reports {equip} steam valve hunting or sticking.", {"steam_valve_stiction": 2.5}))
                elif equip == eq["reboiler"] and code == "FOULING":
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} reports {equip} reboiler duty limited by fouling.", {"reboiler_fouling": 2.5}))
                elif equip == eq["preheater"]:
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} reports feed preheater {equip} out of service.", {"subcooled_feed": 2.5}))
                elif equip == eq["condenser"] and code == "FOULING":
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} reports overhead condenser {equip} performance loss.", {"pressure_rise": 2.0}))
                elif equip == eq["top_analyzer"] and code == "INSTRUMENT":
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} reports analyzer {equip} reading high versus lab.", {"analyzer_fault": 2.5}))
                elif equip == column and code in ("INSPECTION", "MECHANICAL") and "tray" in text.lower():
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} raises suspected tray damage on {column}.", {"tray_damage": 2.5}))
                elif equip == column and "foam" in text.lower():
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} investigates foaming on {column}.", {"foaming": 2.0}))
                elif equip == column and ("low rates" in text.lower() or "rate cut" in text.lower()):
                    cues.append(Observation(f"CUE_{node}", f"Work order {node} reports separation loss at low rates on {column}.", {"turndown": 1.5}))
            elif kind == "operator_action":
                low = text.lower()
                if "feed rate change" in low:
                    cues.append(Observation(f"CUE_{node}", f"Operator action {node}: feed setpoint changed ({text}).", {"turndown": 1.0, "flooding": 0.5}))
                elif "bypass" in low and "bypassed" in low.split("to")[-1]:
                    cues.append(Observation(f"CUE_{node}", f"Operator action {node}: feed preheater bypassed.", {"subcooled_feed": 2.5}))
                elif "pressure setpoint" in low:
                    cues.append(Observation(f"CUE_{node}", f"Operator action {node}: overhead pressure setpoint changed.", {"pressure_rise": 2.0}))
                elif "trip" in low:
                    cues.append(Observation(f"CUE_{node}", f"Operator action {node}: unit trip recorded.", {"tray_damage": 1.5}))
                elif "antifoam" in low:
                    cues.append(Observation(f"CUE_{node}", f"Operator action {node}: antifoam injection started.", {"foaming": 2.0}))
                elif "manual" in low and eq["steam_controller"] in text:
                    cues.append(Observation(f"CUE_{node}", f"Operator action {node}: steam controller taken to manual.", {"steam_valve_stiction": 1.0}))
            elif kind == "gamma_scan":
                if nd.get("damaged_trays"):
                    trays = nd["damaged_trays"]
                    cues.append(Observation(f"CUE_{node}", f"Gamma scan {node} found damaged trays {trays[0]} to {trays[-1]}.", {"tray_damage": 3.5}))
                else:
                    cues.append(Observation(f"CUE_{node}", f"Gamma scan {node} found tray hardware intact.", {"tray_damage": -2.5}))
            elif kind == "moc" and re.search(r"pressure", text, re.I) and re.search(r"raise", nd.get("label", "") + text, re.I):
                cues.append(Observation(f"CUE_{node}", f"MOC {node} raised the operating pressure of {column}.", {"pressure_rise": 2.5}))
            elif kind == "lab" and "contaminant" in text.lower():
                cues.append(Observation(f"CUE_{node}", f"Lab result {node} found feed contamination.", {"foaming": 2.0}))
            elif kind == "lab" and "analyzer validation" in text.lower():
                cues.append(Observation(f"CUE_{node}", f"Lab validation {node} found an analyzer bias.", {"analyzer_fault": 2.5}))
        obs = []
        for sym, ids in sym_ids.items():
            wmap = SYMPTOM_CAUSE_WEIGHTS.get(sym)
            if not wmap:
                continue
            n = len(set(ids))
            scale = 0.6 * min(1.0, 0.5 + 0.25 * n)
            refs = sorted(set(ids))[:4]
            obs.append(Observation(f"TXT_{sym.upper()}", f"{n} record(s) describe {sym.replace('_', ' ')}: " + ", ".join(refs) + ".",
                                   {c: w * scale for c, w in wmap.items()}, float(n), scale))
            obs[-1].refs = refs  # type: ignore[attr-defined]
        for c in cues:
            c.refs = [c.eid.replace("CUE_", "")]  # type: ignore[attr-defined]
        # repeated cues from the same record type only count once per cause and record
        seen = set()
        unique = []
        for c in cues:
            key = c.eid
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return obs + unique
