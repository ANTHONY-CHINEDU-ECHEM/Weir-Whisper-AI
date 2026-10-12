"""Plant knowledge graph.

Nodes: site areas, columns, equipment and instruments, detected episodes, symptoms, cause
hypotheses, knowledge documents, and every evidence record (shift log items, work orders,
alarms, operator actions, flagged lab results, gamma scans, MOCs).

Edges carry a relation name:

PART_OF        equipment to column, column to area
MENTIONS       text record to the column or equipment it names (entity linker)
ON_ASSET       structured record to its asset (alarm to tag, work order to equipment)
REPORTS        text record to a symptom family it describes
INDICATES      symptom to cause hypothesis, weighted from the symptom matrix
DESCRIBES      knowledge document to cause hypothesis or column
DETECTED_ON    episode to column
OVERLAPS       episode to evidence record inside its investigation window (weight decays with time)
PRECEDES       episode to the next episode on the same column

The graph is a networkx MultiDiGraph so it can be inspected, exported to GraphML, or
swapped for a graph database without changing the agents.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from ..ingest.loader import DataStore
from ..nlp.entities import EntityLinker
from ..simulator.scenarios import ALL_CAUSES, CAUSE_LABELS
from ..simulator.tags import EQUIPMENT_DESCRIPTIONS, equipment

LOG = logging.getLogger("weir_whisper.graph")

# symptom family to cause weights, a compact encoding of KB002 (symptom matrix)
SYMPTOM_CAUSE_WEIGHTS = {
    "rate_cut": {"turndown": 2.0},
    "cycling": {"steam_valve_stiction": 2.0},
    "valve_saturated": {"reboiler_fouling": 2.0},
    "chest_high": {"reboiler_fouling": 1.5},
    "fouling": {"reboiler_fouling": 1.2, "pressure_rise": 0.6},
    "pressure_high": {"pressure_rise": 2.0},
    "cold_feed": {"subcooled_feed": 2.0},
    "trip_upset": {"tray_damage": 1.5},
    "tray_damage": {"tray_damage": 2.0},
    "max_rates": {"flooding": 2.0},
    "flooding": {"flooding": 1.5},
    "dp_high": {"flooding": 1.0, "foaming": 1.0},
    "foaming": {"foaming": 2.0},
    "analyzer_issue": {"analyzer_fault": 2.0},
    "weeping": {c: 0.5 for c in ("turndown", "steam_valve_stiction", "reboiler_fouling", "tray_damage", "pressure_rise", "subcooled_feed")},
    "dp_low": {c: 0.5 for c in ("turndown", "steam_valve_stiction", "reboiler_fouling", "tray_damage", "pressure_rise", "subcooled_feed")},
}

CAUSE_DOCS = {
    "turndown": ["KB003", "KB015", "KB001"],
    "steam_valve_stiction": ["KB004", "KB014"],
    "reboiler_fouling": ["KB005"],
    "tray_damage": ["KB009", "KB008", "KB016"],
    "pressure_rise": ["KB006"],
    "subcooled_feed": ["KB007"],
    "flooding": ["KB002", "KB001"],
    "foaming": ["KB010"],
    "analyzer_fault": ["KB011"],
}


def build_graph(store: DataStore, episodes: list[dict], impacts: list[dict], lookback_h: float = 72, lookahead_h: float = 48) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    linker = EntityLinker(store.columns, store.tag_dict.to_dict("records"))
    site = store.settings.plant["site"]
    g.add_node("SITE", kind="site", label=site["name"])
    for area, cols in site["areas"].items():
        g.add_node(area, kind="area", label=area)
        g.add_edge(area, "SITE", rel="PART_OF")
        for c in cols:
            if c not in store.columns:
                continue
            col = store.design(c)
            g.add_node(c, kind="column", label=f"{c} {col['name']}", name=col["name"], service=col["service"], tray_type=col["tray_type"])
            g.add_edge(c, area, rel="PART_OF")
            for role, tag in equipment(c).items():
                if tag == c:
                    continue
                g.add_node(tag, kind="equipment", label=f"{tag} {EQUIPMENT_DESCRIPTIONS[role]}", role=role, column=c)
                g.add_edge(tag, c, rel="PART_OF")
    for tag, info in linker.registry.items():
        if tag not in g:
            g.add_node(tag, kind="instrument", label=f"{tag} {info['description']}", role=info["role"], column=info["column"])
            g.add_edge(tag, info["column"], rel="PART_OF")

    for cause in ALL_CAUSES:
        g.add_node(f"CAUSE:{cause}", kind="cause", label=CAUSE_LABELS[cause])
    for sym, wmap in SYMPTOM_CAUSE_WEIGHTS.items():
        g.add_node(f"SYM:{sym}", kind="symptom", label=sym.replace("_", " "))
        for cause, w in wmap.items():
            g.add_edge(f"SYM:{sym}", f"CAUSE:{cause}", rel="INDICATES", weight=w)
    for kd in store.knowledge:
        g.add_node(kd.doc_id, kind="knowledge", label=kd.title, doc_kind=kd.kind)
        if kd.column:
            g.add_edge(kd.doc_id, kd.column, rel="DESCRIBES")
    for cause, docs in CAUSE_DOCS.items():
        for d in docs:
            if d in g:
                g.add_edge(d, f"CAUSE:{cause}", rel="DESCRIBES")

    def link_text(node_id: str, text: str, default_cols: list[str]):
        ex = linker.extract(text)
        for e in ex.entities:
            if e in g:
                g.add_edge(node_id, e, rel="MENTIONS")
        cols = [c for c in ex.columns if c in default_cols] or (default_cols if len(default_cols) == 1 else [])
        for c in cols:
            g.add_edge(node_id, c, rel="MENTIONS")
        for s in ex.symptoms:
            if f"SYM:{s}" in g:
                g.add_edge(node_id, f"SYM:{s}", rel="REPORTS")
        return cols

    for it in store.log_items.itertuples():
        g.add_node(it.item_id, kind="shift_log", label=it.text[:80], text=it.text, time=it.start, time_end=it.end, author=it.author)
        cols = link_text(it.item_id, it.text, list(it.columns))
        g.nodes[it.item_id]["columns"] = cols
    for wo in store.work_orders.itertuples():
        text = f"{wo.short_text}. {wo.long_text} {wo.closure_notes or ''}"
        g.add_node(wo.wo_id, kind="work_order", label=wo.short_text, text=text, time=wo.created,
                   time_end=wo.completed if pd.notna(wo.completed) else None, equipment=wo.equipment, failure_code=wo.failure_code,
                   status=wo.status, columns=[wo.column])
        if wo.equipment in g:
            g.add_edge(wo.wo_id, wo.equipment, rel="ON_ASSET")
        g.add_edge(wo.wo_id, wo.column, rel="ON_ASSET")
        link_text(wo.wo_id, text, [wo.column])
    for a in store.alarms.itertuples():
        g.add_node(a.alarm_id, kind="alarm", label=a.message, text=a.message, time=a.timestamp, columns=[a.column], priority=a.priority,
                   value=a.value, limit=a.limit, tag=a.tag)
        if a.tag in g:
            g.add_edge(a.alarm_id, a.tag, rel="ON_ASSET")
        g.add_edge(a.alarm_id, a.column, rel="ON_ASSET")
    for a in store.actions.itertuples():
        text = f"{a.description}: {a.tag} {a.kind} {a.old_value} to {a.new_value}"
        g.add_node(a.action_id, kind="operator_action", label=text, text=text, time=a.timestamp, columns=[a.column])
        if a.tag in g:
            g.add_edge(a.action_id, a.tag, rel="ON_ASSET")
        g.add_edge(a.action_id, a.column, rel="ON_ASSET")
    flagged = store.lab[store.lab["flag"] == "HIGH"]
    for r in flagged.itertuples():
        text = f"{r.sample_point} {r.analysis} {r.value} {r.unit} (limit {r.spec_limit})"
        g.add_node(r.sample_id, kind="lab", label=text, text=text, time=r.timestamp, columns=[r.column], value=r.value)
        g.add_edge(r.sample_id, r.column, rel="ON_ASSET")
        link_text(r.sample_id, text, [r.column])
    for gs in store.gamma_scans:
        g.add_node(gs["scan_id"], kind="gamma_scan", label=f"Gamma scan {gs['scan_id']}", text=gs["text"], time=pd.Timestamp(gs["timestamp"]),
                   columns=[gs["column"]], damaged_trays=gs["damaged_trays"])
        g.add_edge(gs["scan_id"], gs["column"], rel="ON_ASSET")
        link_text(gs["scan_id"], gs["text"], [gs["column"]])
    for m in store.mocs:
        g.add_node(m["moc_id"], kind="moc", label=m["title"], text=m["text"], time=pd.Timestamp(m["timestamp"]), columns=[m["column"]])
        g.add_edge(m["moc_id"], m["column"], rel="ON_ASSET")
        link_text(m["moc_id"], f"{m['title']} {m['text']}", [m["column"]])

    # episodes and their temporal neighbourhood
    imp = {i["episode_id"]: i for i in impacts}
    by_col: dict[str, list[tuple]] = {}
    for n, d in g.nodes(data=True):
        if d.get("time") is not None and d.get("kind") in ("shift_log", "work_order", "alarm", "operator_action", "lab", "gamma_scan", "moc"):
            for c in d.get("columns") or []:
                by_col.setdefault(c, []).append((pd.Timestamp(d["time"]).value, n))
    for c in by_col:
        by_col[c].sort()
    prev: dict[str, str] = {}
    for e in episodes:
        eid = e["episode_id"]
        t0, t1 = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        g.add_node(eid, kind="episode", label=f"{eid} {e['kind']} on {e['column']}", time=t0, time_end=t1, episode_kind=e["kind"],
                   column=e["column"], duration_h=e["duration_h"], cost_gbp=imp.get(eid, {}).get("total_cost_gbp", 0.0))
        g.add_edge(eid, e["column"], rel="DETECTED_ON")
        if e["column"] in prev:
            g.add_edge(prev[e["column"]], eid, rel="PRECEDES")
        prev[e["column"]] = eid
        lo = (t0 - pd.Timedelta(hours=lookback_h)).value
        hi = (t1 + pd.Timedelta(hours=lookahead_h)).value
        arr = by_col.get(e["column"], [])
        times = np.array([a for a, _ in arr], dtype=np.int64)
        i0, i1 = np.searchsorted(times, lo), np.searchsorted(times, hi, side="right")
        for tv, node in arr[i0:i1]:
            dist_h = max(0.0, max(t0.value - tv, tv - t1.value) / 3.6e12)
            g.add_edge(eid, node, rel="OVERLAPS", weight=float(np.exp(-dist_h / 24.0)))
    # long running work orders that were open at any time during an episode
    for wo in store.work_orders.itertuples():
        end = wo.completed if pd.notna(wo.completed) else store.historian[wo.column].index[-1]
        for e in episodes:
            if e["column"] != wo.column:
                continue
            t0, t1 = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
            if wo.created <= t1 and end >= t0 and not g.has_edge(e["episode_id"], wo.wo_id):
                g.add_edge(e["episode_id"], wo.wo_id, rel="OVERLAPS", weight=0.8)
    LOG.info("graph built: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
    return g


def evidence_for_episode(g: nx.MultiDiGraph, episode_id: str, kinds: tuple[str, ...] | None = None) -> list[tuple[str, dict, float]]:
    out = []
    for _, node, data in g.out_edges(episode_id, data=True):
        if data.get("rel") != "OVERLAPS":
            continue
        nd = g.nodes[node]
        if kinds and nd.get("kind") not in kinds:
            continue
        out.append((node, nd, data.get("weight", 0.0)))
    out.sort(key=lambda x: pd.Timestamp(x[1]["time"]).value)
    return out


def asset_history(g: nx.MultiDiGraph, asset: str, kinds: tuple[str, ...] = ("work_order",)) -> list[tuple[str, dict]]:
    """Every record attached to an asset, oldest first (used for recurrence)."""
    out = []
    for src, _, data in g.in_edges(asset, data=True):
        if data.get("rel") in ("ON_ASSET", "MENTIONS") and g.nodes[src].get("kind") in kinds:
            out.append((src, g.nodes[src]))
    out = list({k: v for k, v in out}.items())
    out.sort(key=lambda x: pd.Timestamp(x[1]["time"]).value)
    return out
