"""Build the retrievable corpus from every text bearing source.

Each record becomes a Doc with an identifier that traces back to the source system, a
timestamp where one exists, the columns and entities it mentions, and its symptom tags.
Alarms are too granular to retrieve one by one, so they are rolled into hourly alarm
bursts per column; each burst keeps the list of underlying alarm identifiers so evidence
can always be traced to the individual DCS record.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ..ingest.loader import DataStore
from ..nlp.entities import EntityLinker


@dataclass
class Doc:
    doc_id: str
    kind: str
    text: str
    timestamp: pd.Timestamp | None
    columns: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    title: str = ""
    time_end: pd.Timestamp | None = None


def chunk_markdown(doc_id: str, text: str, max_tokens: int) -> list[tuple[str, str, str]]:
    """Split on headings, then pack paragraphs up to max_tokens words. Returns (chunk_id, heading, text)."""
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    k = 0
    for sec in sections:
        lines = sec.strip().split("\n")
        heading = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else ""
        body = "\n".join(lines[1:] if heading else lines).strip()
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        buf: list[str] = []
        count = 0
        for p in paras:
            n = len(p.split())
            if buf and count + n > max_tokens:
                k += 1
                chunks.append((f"{doc_id}#{k}", heading, "\n\n".join(buf)))
                buf, count = [], 0
            buf.append(p)
            count += n
        if buf:
            k += 1
            chunks.append((f"{doc_id}#{k}", heading, "\n\n".join(buf)))
    return chunks


def build_corpus(store: DataStore, linker: EntityLinker, chunk_tokens: int = 160) -> list[Doc]:
    docs: list[Doc] = []

    def add(doc_id, kind, text, ts, title="", source_ids=None, columns=None, time_end=None):
        ex = linker.extract(text)
        cols = list(dict.fromkeys((columns or []) + ex.columns))
        docs.append(Doc(doc_id=doc_id, kind=kind, text=text, timestamp=ts, columns=cols, entities=ex.entities,
                        symptoms=ex.symptoms, source_ids=source_ids or [doc_id], title=title, time_end=time_end))

    for kd in store.knowledge:
        for cid, heading, body in chunk_markdown(kd.doc_id, kd.text, chunk_tokens):
            add(cid, "knowledge" if kd.kind == "article" else "datasheet", f"{kd.title}. {heading}. {body}", None,
                title=kd.title, source_ids=[kd.doc_id], columns=[kd.column] if kd.column else None)
    for it in store.log_items.itertuples():
        # an item can only be about the columns of its own console area
        add(it.item_id, "shift_log", it.text, it.start, title=f"Shift log {it.log_id} ({it.shift}, crew author {it.author})",
            time_end=it.end)
        d = docs[-1]
        d.columns = [c for c in d.columns if c in it.columns] or ([it.columns[0]] if len(it.columns) == 1 else d.columns)
    for wo in store.work_orders.itertuples():
        text = f"{wo.short_text}. {wo.long_text}"
        if isinstance(wo.closure_notes, str) and wo.closure_notes:
            text += f" Closure: {wo.closure_notes}"
        add(wo.wo_id, "work_order", text, wo.created, title=f"Work order {wo.wo_id} {wo.equipment} {wo.priority} {wo.type}",
            columns=[wo.column], time_end=wo.completed if pd.notna(wo.completed) else None)
    for gs in store.gamma_scans:
        add(gs["scan_id"], "gamma_scan", gs["text"], pd.Timestamp(gs["timestamp"]), title=f"Gamma scan {gs['scan_id']}", columns=[gs["column"]])
    for m in store.mocs:
        add(m["moc_id"], "moc", f"{m['title']}. {m['text']}", pd.Timestamp(m["timestamp"]), title=m["title"], columns=[m["column"]])
    acts = store.actions
    for a in acts.itertuples():
        add(a.action_id, "operator_action", f"{a.column} {a.description}: {a.tag} {a.kind} changed from {a.old_value} to {a.new_value}",
            a.timestamp, title="DCS operator action", columns=[a.column])
    lab = store.lab
    flagged = lab[lab["flag"] == "HIGH"]
    for r in flagged.itertuples():
        add(r.sample_id, "lab", f"{r.sample_point} {r.analysis} result {r.value} {r.unit} above limit {r.spec_limit}", r.timestamp,
            title=f"Lab result {r.sample_id}", columns=[r.column])
    al = store.alarms.copy()
    al["hour"] = al["timestamp"].dt.floor("h")
    for (col, hour), grp in al.groupby(["column", "hour"]):
        counts = grp.groupby("message").size().sort_values(ascending=False)
        parts = [f"{msg} ({n} times)" if n > 1 else msg for msg, n in counts.items()]
        doc_id = f"AB{col}{hour.strftime('%Y%m%d%H')}"
        add(doc_id, "alarm_burst", f"{col} alarms {hour.strftime('%Y-%m-%d %H:00')}: " + "; ".join(parts), hour,
            title=f"Alarm burst {col} {hour.strftime('%Y-%m-%d %H:00')}", source_ids=grp["alarm_id"].tolist(), columns=[col],
            time_end=hour + pd.Timedelta(hours=1))
    return docs
