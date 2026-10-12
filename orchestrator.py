"""Investigator: orchestrates the multi agent workflow for one episode.

    context -> signal analyst -> records analyst (graph + hybrid retrieval)
            -> root cause agent -> impact -> recurrence -> writer -> verifier -> report

Agents share an EvidenceIndex (what can be cited) and a FactRegistry (what numbers can be
stated). The verifier is the last step, so nothing reaches a report without passing it.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..graph.builder import CAUSE_DOCS, evidence_for_episode
from ..llm.backends import make_backend, parse_json_block
from ..llm.prompts import QA_TEMPLATE, SYSTEM_QA
from ..nlp.entities import EntityLinker
from ..simulator.scenarios import CAUSE_LABELS
from .base import Claim, EvidenceIndex, FactRegistry, gbp
from .recurrence import RecurrenceAgent, family_key, recurrence_register
from .root_cause import RootCauseAgent
from .signal_analyst import SignalAnalyst
from .text_evidence import RecordsAnalyst
from .verifier import Verifier
from .writer import Writer

LOG = logging.getLogger("weir_whisper.investigator")

METHOD_CONSTANTS = [1.0, 1.00, 72, 95, 65, 70, 2, 24, 48, 12, 6, 3, 5, 0.85]

QUERY_BY_KIND = {
    "weeping": "tray weeping loss of separation low dP temperature profile reboiler steam valve feed rate pressure trip",
    "flooding": "flooding high dP max rates entrainment carryover",
    "foaming": "foaming dP erratic level swinging contamination antifoam",
    "analyzer_fault": "analyzer reading high lab disagrees recalibration APC reflux",
}

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                      "october", "november", "december"], 1)}
CAUSE_WORDS = {
    "steam_valve_stiction": ["stiction", "sticky", "valve", "hunting", "cycling"],
    "reboiler_fouling": ["fouling", "fouled", "reboiler"],
    "turndown": ["turndown", "low rate", "rate cut", "low rates", "throughput"],
    "tray_damage": ["tray damage", "damaged", "trip", "gamma"],
    "pressure_rise": ["pressure"],
    "subcooled_feed": ["cold feed", "preheater", "subcooled", "feed temperature"],
    "flooding": ["flooding", "flood"],
    "foaming": ["foam"],
    "analyzer_fault": ["analyzer", "analyser"],
}


class Investigator:
    def __init__(self, settings, store, graph, index, episodes: list[dict], impacts: dict, feats: dict):
        self.settings = settings
        self.store = store
        self.g = graph
        self.index = index
        self.episodes = sorted(episodes, key=lambda e: e["start"])
        self.by_id = {e["episode_id"]: e for e in self.episodes}
        self.impacts = impacts
        self.feats = feats
        self.rcfg = settings["retrieval"]
        self.linker = EntityLinker(store.columns, store.tag_dict.to_dict("records"))
        self.signal = SignalAnalyst()
        self.records = RecordsAnalyst(graph, index, self.linker, self.rcfg)
        self.root = RootCauseAgent()
        self.recur = RecurrenceAgent(graph, store.knowledge)
        self.backend = make_backend(settings)
        self.writer = Writer(self.backend)
        self.verifier = Verifier(self.rcfg["lookback_hours"], self.rcfg["lookahead_hours"], METHOD_CONSTANTS)
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _timeline(self, ep: dict, evidence: EvidenceIndex, cited: set[str]) -> list[dict]:
        rows = []
        alarms = [(n, nd) for n, nd, _ in evidence_for_episode(self.g, ep["episode_id"], ("alarm",))]
        grouped: dict[str, list] = {}
        for n, nd in alarms:
            grouped.setdefault(nd["label"], []).append((n, nd))
        for label, items in grouped.items():
            first_id, first = items[0]
            text = f"{label} ({len(items)} activations, first at {pd.Timestamp(first['time']).strftime('%Y-%m-%d %H:%M')})"
            evidence.add(first_id, "alarm", text, first["time"])
            rows.append({"time": pd.Timestamp(first["time"]).isoformat(), "source": "alarm", "id": first_id, "text": text, "rank": 3})
        for eid, it in evidence.items.items():
            if it.kind in ("shift_log", "work_order", "operator_action", "lab", "gamma_scan", "moc") and it.time:
                ex = self.linker.extract(it.text)
                if it.kind in ("shift_log", "work_order", "moc") and not (set(ex.symptoms) - {"resolved"}) and eid not in cited:
                    continue
                rank = {"operator_action": 0, "moc": 0, "gamma_scan": 0, "work_order": 1, "lab": 2, "shift_log": 2}[it.kind]
                rows.append({"time": it.time, "source": it.kind, "id": eid, "text": it.text[:240], "rank": rank})
        rows.append({"time": pd.Timestamp(ep["start"]).isoformat(), "source": "detector", "id": "SIG_DETECTOR", "text": f"Episode {ep['episode_id']} starts ({ep['kind']})", "rank": 0})
        rows.append({"time": pd.Timestamp(ep["end"]).isoformat(), "source": "detector", "id": "SIG_DETECTOR", "text": f"Episode {ep['episode_id']} ends", "rank": 0})
        rows.sort(key=lambda r: (r["rank"], r["time"]))
        keep = sorted(rows[:34], key=lambda r: r["time"])
        for r in keep:
            r.pop("rank", None)
        return keep

    def _register_impact(self, imp: dict, evidence: EvidenceIndex, facts: FactRegistry, design: dict) -> None:
        econ = self.settings.plant["economics"]
        for k in ("excess_steam_t", "offspec_t", "offspec_top_t", "offspec_bottom_t", "steam_cost_gbp", "quality_cost_gbp", "co2_t",
                  "carbon_cost_gbp", "total_cost_gbp", "efficiency_loss_pct"):
            facts.add(k, imp[k], "t" if k.endswith("_t") else ("GBP" if k.endswith("gbp") else "percent"), "impact agent")
        facts.add("steam_price", econ["lp_steam_price_per_t"], "GBP/t", "site economics")
        facts.add("downgrade_price", design["downgrade_price_per_t"], "GBP/t", "column datasheet")
        evidence.add("SIG_IMPACT", "signal",
                     f"Impact method: steam measured minus normal behaviour model at the same feed; off specification product from the {imp['quality_source']}; "
                     f"steam at {econ['lp_steam_price_per_t']} GBP/t, downgrade at {design['downgrade_price_per_t']} GBP/t, "
                     f"{econ['steam_co2_t_per_t']} tCO2e per t steam at {econ['carbon_price_per_t']} GBP/tCO2e. Window {imp['window_start']} to {imp['window_end']}.",
                     imp["window_start"])

    def investigate(self, episode_id: str, use: tuple[str, ...] = ("signal", "records")) -> dict:
        ep = self.by_id[episode_id]
        col = ep["column"]
        design = self.store.design(col)
        imp = self.impacts[episode_id]
        evidence, facts = EvidenceIndex(), FactRegistry()
        for kd in self.store.knowledge:
            evidence.add(kd.doc_id, "knowledge", kd.title)
        sig_obs = self.signal.analyse(ep, design, evidence, facts)
        query = f"{col} {design['name']} {QUERY_BY_KIND.get(ep['kind'], '')}"
        records = self.records.collect(ep, evidence, query)
        rec_obs = self.records.analyse(ep, records)
        for o in rec_obs:
            if o.eid.startswith("TXT_"):
                facts.add(f"count_{o.eid}", o.value or 0, "records", "records analyst symptom count")
        dx = self.root.diagnose(sig_obs, rec_obs, use)
        for c, p in dx.ranked()[:5]:
            facts.add(f"posterior_{c}", p, "probability", "root cause agent")
        self._register_impact(imp, evidence, facts, design)
        fam, asset, desc = family_key(col, dx.cause)
        rec = self.recur.for_episode(ep, dx.cause, self.history)
        for e in rec["earlier_episodes"]:
            prev = next((r for r in self.history if r["episode_id"] == e), None)
            if prev:
                evidence.add(e, "episode", f"Earlier episode {e}: {prev['root_cause']['label']}, cost {gbp(prev['impact']['total_cost_gbp'])}", prev["start"])
        for w in rec["prior_work_orders"]:
            if w in self.g:
                nd = self.g.nodes[w]
                evidence.add(w, "work_order", nd.get("text", nd.get("label", "")), nd.get("time"), time_end=nd.get("time_end"))
        facts.add("earlier_cost_gbp", rec["earlier_cost_gbp"], "GBP", "sum of earlier episode costs in the family")
        facts.add("occurrence_number", rec["occurrence_number"], "count", "recurrence agent")

        obs_by_id = {o.eid: o for o in sig_obs + rec_obs}

        def cite(eids: list[str]) -> list[str]:
            out = []
            for e in eids:
                o = obs_by_id.get(e)
                refs = getattr(o, "refs", None) if o is not None else None
                out.extend(refs if refs else [e])
            return list(dict.fromkeys(out))

        ctx = {"episode": ep, "impact": imp, "diagnosis": dx, "recurrence": rec, "column_name": design["name"], "facts": facts, "cite": cite}
        offline = self.writer.offline_claims(ctx)
        llm_claims, llm_meta = self.writer.llm_claims(ctx, evidence, facts)
        claims = offline
        if llm_claims:
            kept_llm, _ = self.verifier.run(llm_claims, evidence, facts, ep["start"], ep["end"])
            by_section = {}
            for c in kept_llm:
                by_section.setdefault(c.section, []).append(c)
            claims = [c for c in offline if c.section not in by_section] + kept_llm
            claims_all = claims
        else:
            claims_all = claims
        kept, verification = self.verifier.run(claims_all, evidence, facts, ep["start"], ep["end"])
        if llm_meta:
            verification["llm"] = llm_meta
        timeline = self._timeline(ep, evidence, {e for c in kept for e in c.evidence})
        report = {
            "episode_id": episode_id,
            "column": col,
            "column_name": design["name"],
            "episode_kind": ep["kind"],
            "start": ep["start"],
            "end": ep["end"],
            "duration_h": ep["duration_h"],
            "material": bool(ep["duration_h"] >= 2.0 or abs(imp["total_cost_gbp"]) >= 1000.0),
            "headline": f"{col} {design['name']}: {dx.label.lower()} ({ep['kind'].replace('_', ' ')}), {gbp(imp['total_cost_gbp'])}",
            "root_cause": dx.to_dict(),
            "family_id": fam,
            "impact": imp,
            "recurrence": rec,
            "claims": [c.to_dict() for c in kept],
            "timeline": timeline,
            "observations": [{"id": o.eid, "statement": o.statement, "contributions": {k: round(v, 3) for k, v in o.support().items()}} for o in sig_obs + rec_obs],
            "facts": facts.to_dict(),
            "evidence": evidence.to_dict(),
            "verification": verification,
            "backend": self.backend.name,
            "knowledge_refs": CAUSE_DOCS.get(dx.cause, []),
        }
        self.history.append(report)
        self.history.sort(key=lambda r: r["start"])
        return report

    def recurrence_summary(self, reports: list[dict]) -> list[dict]:
        return recurrence_register(reports)

    # ------------------------------------------------------------------
    def ask(self, question: str, reports: list[dict] | None = None) -> dict:
        """Answer a free text question from investigation reports and retrieved records."""
        from ..utils.io import read_json
        if reports is None:
            path = self.settings.path("reports") / "all_reports.json"
            reports = read_json(path) if path.exists() else [self.investigate(e["episode_id"]) for e in self.episodes]
        q = question.lower()
        qx = self.linker.extract(question)
        cols = qx.columns
        months = [MONTHS[m] for m in MONTHS if m in q]
        causes = [c for c, words in CAUSE_WORDS.items() if any(w in q for w in words)]
        sel = [r for r in reports if (not cols or r["column"] in cols)]
        if months:
            sel = [r for r in sel if pd.Timestamp(r["start"]).month in months]
        if causes:
            sel = [r for r in sel if r["root_cause"]["cause"] in causes] or sel
        aggregate = any(w in q for w in ("most", "total", "how much", "worst", "biggest", "rank", "which column", "costliest"))
        recurring = any(w in q for w in ("recur", "again", "repeat", "keep", "how often"))
        evidence, facts = EvidenceIndex(), FactRegistry()
        lines: list[Claim] = []
        if not sel:
            lines.append(Claim("No investigated episode matches that question.", ["NONE"], "method", "answer"))
        elif recurring:
            reg = [f for f in recurrence_register(sel) if f["recurring"]]
            for f in reg[:5]:
                facts.add(f"fam_{f['family_id']}_n", f["occurrences"], "count", "recurrence register")
                facts.add(f"fam_{f['family_id']}_cost", round(f["total_cost_gbp"]), "GBP", "recurrence register")
                for e in f["episodes"]:
                    evidence.add(e, "episode", e)
                lines.append(Claim(f"{f['cause_label']} on {f['asset']} ({f['column']}) recurred {f['occurrences']} times for a total of {gbp(f['total_cost_gbp'])}.",
                                   f["episodes"][:4], "history", "answer"))
        elif aggregate:
            df = pd.DataFrame([{"column": r["column"], "cause": r["root_cause"]["cause"], "cost": r["impact"]["total_cost_gbp"], "id": r["episode_id"]} for r in sel])
            by_col = df.groupby("column")["cost"].sum().sort_values(ascending=False)
            for c, v in by_col.head(3).items():
                ids = df[df["column"] == c].sort_values("cost", ascending=False)["id"].tolist()[:3]
                top_cause = df[df["column"] == c].groupby("cause")["cost"].sum().idxmax()
                facts.add(f"col_{c}_cost", round(v), "GBP", "sum of episode costs")
                for e in ids:
                    evidence.add(e, "episode", e)
                lines.append(Claim(f"{c} {self.store.design(c)['name']} lost {gbp(v)} across {int((df['column'] == c).sum())} episodes, mostly to {CAUSE_LABELS[top_cause].lower()}.",
                                   ids, "quantity", "answer"))
                facts.add(f"col_{c}_n", int((df["column"] == c).sum()), "count", "episode count")
        else:
            for r in sorted(sel, key=lambda r: -r["impact"]["total_cost_gbp"])[:3]:
                summary = [c for c in r["claims"] if c["section"] == "summary"]
                evidence.add(r["episode_id"], "episode", r["headline"], r["start"])
                for c in summary[:2]:
                    for e in c["evidence"]:
                        it = r["evidence"].get(e)
                        if it:
                            evidence.add(e, it["kind"], it["text"], it["time"])
                    for k, f in r["facts"].items():
                        facts.add(f"{r['episode_id']}_{k}", f["value"], f["unit"], f["method"])
                    lines.append(Claim(f"{r['episode_id']} ({r['start'][:10]}): {c['text']}", c["evidence"], "history", "answer"))
        hits = self.index.search(question, column=cols[0] if len(cols) == 1 else None, k=5, mode="full" if cols else "hybrid")
        related = [{"id": h.doc.doc_id, "kind": h.doc.kind, "text": h.doc.text[:200]} for h in hits]
        backend_meta = None
        if self.backend.name != "offline" and sel:
            ev = "\n".join(f"{k} | {v.kind} | {v.time or ''} | {v.text[:200]}" for k, v in evidence.items.items())
            ev += "\n" + "\n".join(f"{h['id']} | {h['kind']} | | {h['text']}" for h in related)
            for h in hits:
                evidence.add(h.doc.doc_id, h.doc.kind, h.doc.text, h.doc.timestamp)
            res = self.backend.complete(SYSTEM_QA, QA_TEMPLATE.format(question=question, facts="\n".join(f"{k}: {v.value}" for k, v in facts.facts.items()), evidence=ev))
            data = parse_json_block(res.text) if res else None
            if data and data.get("answer"):
                llm_lines = [Claim(str(c.get("text", "")), [str(x).strip("[]") for x in c.get("evidence", [])], "history", "answer") for c in data["answer"]]
                kept, _ = self.verifier.run(llm_lines, evidence, facts, pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01"))
                if kept:
                    lines = kept
                    backend_meta = {"backend": res.backend, "model": res.model}
        md = [f"**Question:** {question}", ""]
        for c in lines:
            md.append(f"* {c.text} " + " ".join(f"[{e}]" for e in c.evidence if e != "NONE"))
        if related:
            md += ["", "Related records:"] + [f"* [{h['id']}] {h['text']}" for h in related]
        return {"question": question, "filters": {"columns": cols, "months": months, "causes": causes},
                "answer": [c.to_dict() for c in lines], "related": related, "markdown": "\n".join(md), "backend": backend_meta}
