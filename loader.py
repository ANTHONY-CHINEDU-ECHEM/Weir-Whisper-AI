"""Load every operational data source into one in memory store.

The store is the only door the analytics, graph and agent layers use to reach data. It
deliberately has no access to the ground truth folder, which keeps the evaluation honest:
anything the system reports was derived from historian, alarms, logs, work orders, lab
results, scans, MOCs and the engineering library alone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..simulator.hydraulics import TrayGeometry, derive_geometry
from ..utils.io import read_json, read_jsonl

LOG = logging.getLogger("weir_whisper.ingest")


@dataclass
class KnowledgeDoc:
    doc_id: str
    title: str
    kind: str
    text: str
    column: str | None = None


@dataclass
class DataStore:
    settings: Settings
    raw: Path
    tag_dict: pd.DataFrame
    historian: dict[str, pd.DataFrame]
    alarms: pd.DataFrame
    actions: pd.DataFrame
    shift_logs: list[dict]
    work_orders: pd.DataFrame
    lab: pd.DataFrame
    gamma_scans: list[dict]
    mocs: list[dict]
    knowledge: list[KnowledgeDoc]
    geometry: dict[str, TrayGeometry] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return list(self.historian.keys())

    def design(self, column: str) -> dict:
        return self.settings.column(column)

    @cached_property
    def log_items(self) -> pd.DataFrame:
        rows = []
        for log in self.shift_logs:
            for it in log["items"]:
                rows.append({"item_id": it["item_id"], "log_id": log["log_id"], "area": log["area"], "shift": log["shift"],
                             "author": log["author"], "start": pd.Timestamp(log["start"]), "end": pd.Timestamp(log["end"]),
                             "text": it["text"], "columns": log["columns"]})
        return pd.DataFrame(rows)


def load_store(settings: Settings) -> DataStore:
    raw = settings.path("raw")
    if not (raw / "manifest.json").exists():
        raise FileNotFoundError(f"No dataset at {raw}. Run: python run.py generate")
    tag_dict = pd.read_csv(raw / "tag_dictionary.csv")
    historian = {}
    for column in settings.columns:
        df = pd.read_parquet(raw / "historian" / f"{column}.parquet")
        roles = tag_dict[tag_dict["column"] == column].set_index("tag")["role"].to_dict()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
        df = df.rename(columns=roles).set_index("timestamp")
        historian[column] = df
    alarms = pd.DataFrame(read_jsonl(raw / "alarms.jsonl"))
    alarms["timestamp"] = pd.to_datetime(alarms["timestamp"], format="ISO8601").astype("datetime64[ns]")
    actions = pd.DataFrame(read_jsonl(raw / "operator_actions.jsonl"))
    if len(actions):
        actions["timestamp"] = pd.to_datetime(actions["timestamp"], format="ISO8601").astype("datetime64[ns]")
    wos = pd.DataFrame(read_jsonl(raw / "work_orders.jsonl"))
    wos["created"] = pd.to_datetime(wos["created"], format="ISO8601").astype("datetime64[ns]")
    wos["completed"] = pd.to_datetime(wos["completed"], format="ISO8601").astype("datetime64[ns]")
    lab = pd.read_csv(raw / "lab_results.csv")
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], format="ISO8601").astype("datetime64[ns]")
    kdocs = []
    kroot = settings.path("knowledge")
    for m in read_json(kroot / "manifest.json"):
        kdocs.append(KnowledgeDoc(doc_id=m["doc_id"], title=m["title"], kind=m["kind"],
                                  text=(kroot / m["file"]).read_text(encoding="utf8"), column=m.get("column")))
    store = DataStore(
        settings=settings, raw=raw, tag_dict=tag_dict, historian=historian, alarms=alarms, actions=actions,
        shift_logs=read_jsonl(raw / "shift_logs.jsonl"), work_orders=wos, lab=lab,
        gamma_scans=read_jsonl(raw / "gamma_scans.jsonl"), mocs=read_jsonl(raw / "mocs.jsonl"), knowledge=kdocs,
        geometry={c: derive_geometry(c, settings.column(c)) for c in settings.columns},
    )
    LOG.info("loaded store: %d columns, %d alarms, %d log items, %d work orders, %d lab results, %d knowledge docs",
             len(historian), len(alarms), len(store.log_items), len(wos), len(lab), len(kdocs))
    return store
