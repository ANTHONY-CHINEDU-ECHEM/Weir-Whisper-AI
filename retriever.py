"""Hybrid GraphRAG retriever.

Four candidate channels are fused with reciprocal rank fusion and then reranked:

1. BM25 lexical channel, which is strong on exact tags such as FV207 or WO numbers.
2. Dense channel (LSA by default), which matches paraphrases such as "hunting" and
   "cycling" and survives typos through character n grams.
3. Graph and time channel: records attached to the same column and its equipment in the
   knowledge graph, inside the investigation window, ranked by temporal proximity. This is
   the channel that finds a shift log saying "steam flow swinging" that shares no words
   with the query but sits on the right asset at the right time.
4. Knowledge channel: library chunks linked in the graph to the symptoms observed.

The reranker is a linear model over interpretable features (fused rank, lexical and
semantic scores, same asset, temporal proximity, entity and symptom overlap, record type).
Its weights default to engineering judgement and can be refitted on historical
investigations, which the evaluation suite does on a strict temporal split.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ingest.loader import DataStore
from ..nlp.entities import EntityLinker
from .bm25 import BM25
from .corpus import Doc, build_corpus
from .dense import make_encoder

LOG = logging.getLogger("weir_whisper.retriever")

KINDS = ["shift_log", "work_order", "alarm_burst", "knowledge", "datasheet", "gamma_scan", "moc", "lab", "operator_action"]
RERANK_FEATURES = ["rrf", "bm25", "dense", "same_column", "time_prox", "in_window", "entity_overlap", "symptom_overlap"] + [f"kind_{k}" for k in KINDS]

DEFAULT_WEIGHTS = {
    "rrf": 18.0, "bm25": 0.8, "dense": 1.2, "same_column": 1.5, "time_prox": 2.2, "in_window": 1.0,
    "entity_overlap": 1.0, "symptom_overlap": 0.6,
    "kind_shift_log": 0.3, "kind_work_order": 0.5, "kind_alarm_burst": -0.2, "kind_knowledge": 0.2, "kind_datasheet": -0.3,
    "kind_gamma_scan": 0.6, "kind_moc": 0.5, "kind_lab": 0.0, "kind_operator_action": 0.4,
}


@dataclass
class Hit:
    doc: Doc
    score: float
    features: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)


class HybridIndex:
    def __init__(self, docs: list[Doc], linker: EntityLinker, n_components: int, seed: int, rrf_k: int = 60):
        self.docs = docs
        self.linker = linker
        self.rrf_k = rrf_k
        texts = [f"{d.title}. {d.text}" for d in docs]
        self.bm25 = BM25(texts)
        self.encoder = make_encoder(n_components, seed)
        self.emb = self.encoder.fit_transform(texts)
        self.ts = np.array([d.timestamp.value if d.timestamp is not None else -1 for d in docs], dtype=np.int64)
        self.ts_end = np.array([(d.time_end or d.timestamp).value if d.timestamp is not None else -1 for d in docs], dtype=np.int64)
        self.kind = np.array([d.kind for d in docs])
        self.col_sets = [set(d.columns) for d in docs]
        self.ent_sets = [set(d.entities) for d in docs]
        self.sym_sets = [set(d.symptoms) for d in docs]
        self.by_id = {d.doc_id: i for i, d in enumerate(docs)}
        self.source_to_doc: dict[str, str] = {}
        for d in docs:
            for s in d.source_ids:
                self.source_to_doc.setdefault(s, d.doc_id)
        self.weights = dict(DEFAULT_WEIGHTS)

    # channels ---------------------------------------------------------------
    def _graph_time_channel(self, column, t0, t1, lookback_h, lookahead_h) -> np.ndarray:
        score = np.zeros(len(self.docs), dtype=np.float32)
        if column is None or t0 is None:
            return score
        lo = (t0 - pd.Timedelta(hours=lookback_h)).value
        hi = (t1 + pd.Timedelta(hours=lookahead_h)).value
        has_t = self.ts >= 0
        overlap = has_t & (self.ts <= hi) & (self.ts_end >= lo)
        same = np.array([column in s for s in self.col_sets])
        mid = (t0.value + t1.value) / 2
        half = max((t1.value - t0.value) / 2, pd.Timedelta(hours=6).value)
        dist = np.maximum(np.abs(self.ts - mid) - half, 0) / pd.Timedelta(hours=24).value
        score[overlap & same] = np.exp(-dist[overlap & same]) + 0.5
        return score

    def _knowledge_channel(self, symptoms: set[str], hypotheses: list[str] | None) -> np.ndarray:
        score = np.zeros(len(self.docs), dtype=np.float32)
        if not symptoms and not hypotheses:
            return score
        for i, d in enumerate(self.docs):
            if d.kind in ("knowledge", "datasheet"):
                score[i] = len(self.sym_sets[i] & symptoms)
        return score

    @staticmethod
    def _ranks(scores: np.ndarray, top: int = 300) -> dict[int, int]:
        nz = np.where(scores > 0)[0]
        if len(nz) == 0:
            return {}
        order = nz[np.argsort(-scores[nz])][:top]
        return {int(i): r for r, i in enumerate(order, 1)}

    # main entry ---------------------------------------------------------------
    def search(self, query: str, column: str | None = None, t0: pd.Timestamp | None = None, t1: pd.Timestamp | None = None,
               k: int = 12, mode: str = "full", lookback_h: float = 72, lookahead_h: float = 48,
               kinds: list[str] | None = None, extra_symptoms: list[str] | None = None) -> list[Hit]:
        qx = self.linker.extract(query)
        q_entities = set(qx.entities) | ({column} if column else set())
        q_symptoms = set(qx.symptoms) | set(extra_symptoms or [])
        bm = self.bm25.scores(query)
        qv = self.encoder.encode([query])[0]
        dn = self.emb @ qv
        dn_pos = np.clip(dn, 0, None)
        channels = {"bm25": bm, "dense": dn_pos}
        if mode in ("hybrid_graph", "full"):
            channels["graph"] = self._graph_time_channel(column, t0, t1, lookback_h, lookahead_h)
            channels["knowledge"] = self._knowledge_channel(q_symptoms, None)
        mask = np.ones(len(self.docs), dtype=bool)
        if kinds:
            mask &= np.isin(self.kind, kinds)

        if mode == "bm25":
            fused = bm.copy()
        elif mode == "dense":
            fused = dn.copy()
        else:
            fused = np.zeros(len(self.docs))
            rank_maps = {name: self._ranks(np.where(mask, sc, 0)) for name, sc in channels.items()}
            for name, rm in rank_maps.items():
                for i, r in rm.items():
                    fused[i] += 1.0 / (self.rrf_k + r)
        fused = np.where(mask, fused, -np.inf)
        pool = int(min(len(self.docs), max(k * 8, 120)))
        cand = np.argpartition(-fused, pool - 1)[:pool] if len(fused) > pool else np.arange(len(fused))
        cand = cand[np.isfinite(fused[cand])]
        cand = cand[np.argsort(-fused[cand])]
        if mode != "full":
            return [Hit(self.docs[i], float(fused[i]), {}, {n: float(c[i]) for n, c in channels.items()}) for i in cand[:k]]
        feats = self.rerank_features(cand, fused, bm, dn, column, t0, t1, q_entities, q_symptoms, lookback_h, lookahead_h)
        w = np.array([self.weights.get(f, 0.0) for f in RERANK_FEATURES])
        scores = feats @ w
        order = np.argsort(-scores)[:k]
        return [Hit(self.docs[cand[j]], float(scores[j]), dict(zip(RERANK_FEATURES, feats[j].round(4).tolist())),
                    {n: float(c[cand[j]]) for n, c in channels.items()}) for j in order]

    def rerank_features(self, cand, fused, bm, dn, column, t0, t1, q_entities, q_symptoms, lookback_h, lookahead_h) -> np.ndarray:
        F = np.zeros((len(cand), len(RERANK_FEATURES)), dtype=np.float32)
        bmax = max(float(bm[cand].max()) if len(cand) else 1.0, 1e-9)
        for r, i in enumerate(cand):
            row = {
                "rrf": float(fused[i]),
                "bm25": float(bm[i]) / bmax,
                "dense": float(dn[i]),
                "same_column": float(column in self.col_sets[i]) if column else 0.0,
                "time_prox": 0.0,
                "in_window": 0.0,
                "entity_overlap": len(self.ent_sets[i] & q_entities) / max(len(q_entities), 1),
                "symptom_overlap": float(len(self.sym_sets[i] & q_symptoms)),
            }
            if t0 is not None and self.ts[i] >= 0:
                mid = (t0.value + t1.value) / 2
                half = max((t1.value - t0.value) / 2, pd.Timedelta(hours=6).value)
                dist = max(abs(self.ts[i] - mid) - half, 0) / pd.Timedelta(hours=24).value
                row["time_prox"] = float(np.exp(-dist))
                lo = (t0 - pd.Timedelta(hours=lookback_h)).value
                hi = (t1 + pd.Timedelta(hours=lookahead_h)).value
                row["in_window"] = float(self.ts[i] <= hi and self.ts_end[i] >= lo)
            row[f"kind_{self.kind[i]}"] = 1.0
            F[r] = [row.get(f, 0.0) for f in RERANK_FEATURES]
        return F

    def doc(self, doc_id: str) -> Doc | None:
        i = self.by_id.get(doc_id)
        return self.docs[i] if i is not None else None


def build_index(store: DataStore, settings) -> HybridIndex:
    cfg = settings["retrieval"]
    linker = EntityLinker(store.columns, store.tag_dict.to_dict("records"))
    docs = build_corpus(store, linker, int(cfg["chunk_tokens"]))
    idx = HybridIndex(docs, linker, int(cfg["lsa_components"]), settings.seed, int(cfg["rrf_k"]))
    from ..config import ROOT
    from ..utils.io import read_json
    wpath = settings.path("models") / "reranker_weights.json"
    if wpath.exists():
        idx.weights.update(read_json(wpath))
        LOG.info("loaded learned reranker weights from %s", wpath.relative_to(ROOT))
    kinds = pd.Series([d.kind for d in docs]).value_counts().to_dict()
    LOG.info("indexed %d documents: %s", len(docs), kinds)
    return idx
