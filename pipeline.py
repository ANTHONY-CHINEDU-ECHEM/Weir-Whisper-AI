"""Pipeline orchestration used by run.py and the API.

Each stage reads the outputs of the previous one from disk, so stages can be rerun
independently: regenerate data, retrain, rebuild the graph or rewrite reports without
repeating everything upstream.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .config import load_settings
from .utils.io import read_json, setup_logging, timed, write_json, write_pickle

LOG = logging.getLogger("weir_whisper")

USAGE = """usage: python run.py <stage> [argument]

stages:
  generate              build the synthetic plant dataset
  analyse               quality checks, soft sensors, detector, episodes, impact
  graph                 build the knowledge graph
  index                 build the hybrid retrieval index
  investigate [id]      investigate one detected episode, or all of them
  ask "question"        ask a free text question about the plant history
  evaluate              score every stage against ground truth
  figures               render README figures
  serve                 start the REST API
  all                   generate, analyse, graph, index, investigate, evaluate, figures
"""


def stage_generate(settings):
    from .simulator.generator import generate, write_sample
    workers = int(os.environ.get("WEIR_WORKERS", "2"))
    stats = generate(settings, workers=workers)
    if settings.profile == "default":
        write_sample(settings)
    return stats


def stage_analyse(settings):
    from .analytics.detector import run_detection
    from .analytics.episodes import detect_episodes
    from .analytics.features import build_features
    from .analytics.impact import episode_impact
    from .evaluation.truth import sample_labels
    from .ingest.loader import load_store
    from .ingest.quality import run_quality

    store = load_store(settings)
    proc = settings.path("processed")
    metrics = settings.path("metrics")
    det = settings["detector"]
    step = int(settings["simulation"]["step_minutes"])
    ref_days = float(det["reference_days"])
    with timed("data quality"):
        write_json(metrics / "data_quality.json", run_quality(store))
    feats, nbms = {}, {}
    with timed("feature engineering"):
        for c in store.columns:
            feats[c], nbms[c] = build_features(store.historian[c], store.lab, c, store.design(c), store.geometry[c],
                                               ref_days, det["window_short"], det["window_long"])
            feats[c].to_parquet(proc / f"features_{c}.parquet")
    labels = {c: sample_labels(settings, c) for c in store.columns}
    with timed("detector training"):
        res = run_detection(feats, labels, det, settings.seed, ref_days)
    models = settings.path("models")
    write_pickle(models / "detector.pkl", res["model"])
    write_pickle(models / "normal_models.pkl", nbms)
    write_json(metrics / "feature_importance.json", res["importance"])
    for c in store.columns:
        np.save(proc / f"probs_oof_{c}.npy", res["oof"][c])
        np.save(proc / f"probs_forward_{c}.npy", res["forward"][c])
        if c in res["loco"]:
            np.save(proc / f"probs_loco_{c}.npy", res["loco"][c])
        for name, sc in res["baselines"][c].items():
            np.save(proc / f"baseline_{name}_{c}.npy", sc)
    np.save(proc / "forward_test_mask.npy", res["test_mask"])
    with timed("episode segmentation and impact"):
        episodes = detect_episodes(res["oof"], feats, det, step, ref_days)
        econ = settings.plant["economics"]
        impacts = [episode_impact(e, store.historian[e.column], feats[e.column], store.lab, store.design(e.column), econ, step) for e in episodes]
    write_json(proc / "episodes.json", [e.to_dict() for e in episodes])
    write_json(proc / "impacts.json", impacts)
    LOG.info("detected %d episodes (%d weeping)", len(episodes), sum(e.kind == "weeping" for e in episodes))
    return episodes


def stage_graph(settings):
    from .graph.builder import build_graph
    from .ingest.loader import load_store
    store = load_store(settings)
    episodes = read_json(settings.path("processed") / "episodes.json")
    impacts = read_json(settings.path("processed") / "impacts.json")
    g = build_graph(store, episodes, impacts)
    write_pickle(settings.path("graph") / "graph.pkl", g)
    write_json(settings.path("metrics") / "graph_stats.json", graph_stats(g))
    return g


def graph_stats(g) -> dict:
    kinds = pd.Series([d.get("kind") for _, d in g.nodes(data=True)]).value_counts().to_dict()
    rels = pd.Series([d.get("rel") for _, _, d in g.edges(data=True)]).value_counts().to_dict()
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "node_kinds": kinds, "edge_relations": rels}


def stage_index(settings):
    from .ingest.loader import load_store
    from .retrieval.retriever import build_index
    store = load_store(settings)
    idx = build_index(store, settings)
    write_pickle(settings.path("index") / "index.pkl", idx)
    return idx


def load_runtime(settings):
    """Everything the investigator needs, loaded once."""
    from .agents.orchestrator import Investigator
    from .ingest.loader import load_store
    from .utils.io import read_pickle
    store = load_store(settings)
    g = read_pickle(settings.path("graph") / "graph.pkl")
    idx = read_pickle(settings.path("index") / "index.pkl")
    episodes = read_json(settings.path("processed") / "episodes.json")
    impacts = {i["episode_id"]: i for i in read_json(settings.path("processed") / "impacts.json")}
    feats = {c: pd.read_parquet(settings.path("processed") / f"features_{c}.parquet") for c in store.columns}
    return Investigator(settings, store, g, idx, episodes, impacts, feats)


def stage_investigate(settings, args):
    from .reporting.render import write_register, write_report
    inv = load_runtime(settings)
    targets = args or [e["episode_id"] for e in inv.episodes]
    reports = []
    with timed(f"investigating {len(targets)} episodes"):
        for eid in targets:
            rep = inv.investigate(eid)
            write_report(settings.path("reports"), rep)
            reports.append(rep)
    if not args:
        recurrence = inv.recurrence_summary(reports)
        write_json(settings.path("reports") / "recurrence.json", recurrence)
        write_register(settings.path("reports"), reports, recurrence)
        write_json(settings.path("reports") / "all_reports.json", reports)
    return reports


def stage_ask(settings, args):
    inv = load_runtime(settings)
    question = " ".join(args) if args else "Which column lost the most money to weeping and why?"
    ans = inv.ask(question)
    print(ans["markdown"])
    return ans


def stage_evaluate(settings):
    from .evaluation.suite import run_suite
    return run_suite(settings)


def stage_figures(settings):
    from .reporting.figures import make_all
    return make_all(settings)


def stage_serve(settings):
    import uvicorn
    from .api.app import create_app
    cfg = settings["api"]
    uvicorn.run(create_app(settings), host=cfg["host"], port=int(cfg["port"]))


def main(argv: list[str]) -> int:
    setup_logging()
    if not argv or argv[0] in ("help", "h"):
        print(USAGE)
        return 0
    settings = load_settings()
    cmd, args = argv[0], argv[1:]
    stages = {
        "generate": lambda: stage_generate(settings),
        "analyse": lambda: stage_analyse(settings),
        "graph": lambda: stage_graph(settings),
        "index": lambda: stage_index(settings),
        "investigate": lambda: stage_investigate(settings, args),
        "ask": lambda: stage_ask(settings, args),
        "evaluate": lambda: stage_evaluate(settings),
        "figures": lambda: stage_figures(settings),
        "serve": lambda: stage_serve(settings),
    }
    if cmd == "all":
        for name in ("generate", "analyse", "graph", "index", "investigate", "evaluate", "figures"):
            with timed(f"stage {name}"):
                stages[name]()
        return 0
    if cmd not in stages:
        print(USAGE)
        return 1
    with timed(f"stage {cmd}"):
        stages[cmd]()
    return 0
