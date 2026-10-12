"""REST API.

GET  /health                      service and data status
GET  /columns                     column design summaries
GET  /episodes                    detected episodes (filter by column, kind, month)
GET  /episodes/{episode_id}       one episode with its impact
POST /investigate/{episode_id}    run the agents and return the verified report
GET  /reports/{episode_id}.md     the report as Markdown
GET  /recurrence                  recurrence register
POST /ask                         free text question over the investigation history
POST /search                      hybrid GraphRAG search over plant records
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..reporting.render import report_markdown


class AskBody(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class SearchBody(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    column: str | None = None
    start: str | None = None
    end: str | None = None
    k: int = Field(10, ge=1, le=50)
    mode: str = Field("full", pattern="^(bm25|dense|hybrid|hybrid_graph|full)$")


def create_app(settings) -> FastAPI:
    app = FastAPI(title="Weir Whisper AI", version="1.0.0",
                  description="Multi agent GraphRAG investigator for tray weeping and efficiency loss in distillation columns.")

    @lru_cache(maxsize=1)
    def runtime():
        from ..pipeline import load_runtime
        return load_runtime(settings)

    reports: dict[str, dict] = {}

    @app.get("/health")
    def health():
        inv = runtime()
        return {"status": "ok", "columns": inv.store.columns, "episodes": len(inv.episodes), "indexed_documents": len(inv.index.docs),
                "graph_nodes": inv.g.number_of_nodes(), "backend": inv.backend.name}

    @app.get("/columns")
    def columns():
        inv = runtime()
        return [{"column": c, "name": inv.store.design(c)["name"], "service": inv.store.design(c)["service"], "tray_type": inv.store.design(c)["tray_type"],
                 "design_feed_t_h": inv.store.design(c)["feed_design_t_h"]} for c in inv.store.columns]

    @app.get("/episodes")
    def episodes(column: str | None = None, kind: str | None = None, month: int | None = None):
        inv = runtime()
        out = []
        for e in inv.episodes:
            if column and e["column"] != column:
                continue
            if kind and e["kind"] != kind:
                continue
            if month and pd.Timestamp(e["start"]).month != month:
                continue
            out.append({k: e[k] for k in ("episode_id", "column", "kind", "start", "end", "duration_h", "mean_prob")} |
                       {"cost_gbp": inv.impacts[e["episode_id"]]["total_cost_gbp"]})
        return out

    @app.get("/episodes/{episode_id}")
    def episode(episode_id: str):
        inv = runtime()
        if episode_id not in inv.by_id:
            raise HTTPException(404, f"unknown episode {episode_id}")
        return {"episode": inv.by_id[episode_id], "impact": inv.impacts[episode_id]}

    @app.post("/investigate/{episode_id}")
    def investigate(episode_id: str):
        inv = runtime()
        if episode_id not in inv.by_id:
            raise HTTPException(404, f"unknown episode {episode_id}")
        rep = inv.investigate(episode_id)
        reports[episode_id] = rep
        return rep

    @app.get("/reports/{episode_id}.md", response_class=PlainTextResponse)
    def report_md(episode_id: str):
        inv = runtime()
        if episode_id not in inv.by_id:
            raise HTTPException(404, f"unknown episode {episode_id}")
        rep = reports.get(episode_id) or inv.investigate(episode_id)
        reports[episode_id] = rep
        return report_markdown(rep)

    @app.get("/recurrence")
    def recurrence():
        from ..utils.io import read_json
        path = settings.path("reports") / "recurrence.json"
        if not path.exists():
            raise HTTPException(409, "run: python run.py investigate")
        return read_json(path)

    @app.post("/ask")
    def ask(body: AskBody):
        return runtime().ask(body.question)

    @app.post("/search")
    def search(body: SearchBody):
        inv = runtime()
        t0 = pd.Timestamp(body.start) if body.start else None
        t1 = pd.Timestamp(body.end) if body.end else t0
        hits = inv.index.search(body.query, column=body.column, t0=t0, t1=t1, k=body.k, mode=body.mode)
        return [{"id": h.doc.doc_id, "kind": h.doc.kind, "time": h.doc.timestamp.isoformat() if h.doc.timestamp is not None else None,
                 "columns": h.doc.columns, "score": round(h.score, 4), "text": h.doc.text[:400]} for h in hits]

    return app
