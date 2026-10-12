"""Evaluation suite: scores every stage against the simulator ground truth.

Outputs (outputs/metrics):
  detection.json        sample level and event level detection, baselines, cross column transfer
  root_cause.json       diagnosis accuracy, ablations, calibration, learned baseline, confusion matrix
  impact.json           cost and efficiency estimates against true counterfactual costs
  retrieval.json        hybrid GraphRAG retrieval against the evidence map, with ablations and a learned reranker
  recurrence.json       pairwise recurrence linking quality
  faithfulness.json     verifier coverage on real reports and detection rate on injected errors
  entity_linking.json   column linking accuracy on scenario related shift log items
  summary.md            human readable roll up used to write the README
"""
from __future__ import annotations

import copy
import logging
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from ..simulator.scenarios import ALL_CAUSES, SAMPLE_CLASSES, WEEP_CAUSES
from ..utils.io import read_json, write_json
from . import truth as T
from .matching import match

LOG = logging.getLogger("weir_whisper.evaluation")


# ---------------------------------------------------------------- detection
def _sample_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = p >= thr
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if y.any() and (~y).any() else None,
        "average_precision": round(float(average_precision_score(y, p)), 4) if y.any() else None,
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "positives": int(y.sum()), "samples": int(len(y)),
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 37)
    f = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def evaluate_detection(settings, store_columns: list[str]) -> dict:
    proc = settings.path("processed")
    te = np.load(proc / "forward_test_mask.npy")
    ys, cls_all, P = [], [], {k: [] for k in ("oof", "forward", "loco", "physics_rule", "isolation_forest")}
    for c in store_columns:
        lab = T.sample_labels(settings, c)
        cls_all.append(lab)
        ys.append(lab == 1)
        P["oof"].append(np.load(proc / f"probs_oof_{c}.npy"))
        P["forward"].append(np.load(proc / f"probs_forward_{c}.npy"))
        loco = proc / f"probs_loco_{c}.npy"
        P["loco"].append(np.load(loco) if loco.exists() else np.full_like(P["oof"][-1], np.nan))
        P["physics_rule"].append(np.load(proc / f"baseline_physics_rule_{c}.npy"))
        P["isolation_forest"].append(np.load(proc / f"baseline_isolation_forest_{c}.npy"))
    y = np.concatenate(ys)
    cls = np.concatenate(cls_all)
    test = np.tile(te, len(store_columns))
    train = ~test
    out = {"protocols": {}}
    # thresholds are chosen on the training period only, then applied to the test period. The learned
    # protocols take their threshold from out of fold training period scores, never in sample ones.
    oof_p = np.concatenate(P["oof"])[:, 1]
    learned_thr = best_threshold(y[train], oof_p[train])
    for name in P:
        arr = np.concatenate(P[name])
        p = arr[:, 1] if arr.ndim == 2 else arr
        if np.isnan(p).all():
            continue
        thr = learned_thr if name in ("oof", "forward", "loco") else best_threshold(y[train], p[train])
        out["protocols"][name] = {"threshold": round(thr, 3), "test_period": _sample_metrics(y[test], p[test], thr),
                                  "full_year": _sample_metrics(y, p, thr)}
    # per column transfer
    per_col = {}
    for i, c in enumerate(store_columns):
        lc = P["loco"][i]
        if np.isnan(lc).all():
            continue
        per_col[c] = {"loco_ap": round(float(average_precision_score(ys[i], lc[:, 1])), 4) if ys[i].any() else None,
                      "oof_ap": round(float(average_precision_score(ys[i], P["oof"][i][:, 1])), 4) if ys[i].any() else None}
    out["leave_one_column_out"] = per_col
    # multiclass on the forward test period
    fwd = np.concatenate(P["forward"])
    pred = fwd.argmax(axis=1)
    cm = confusion_matrix(cls[test], pred[test], labels=list(range(len(SAMPLE_CLASSES))))
    present = [i for i in range(len(SAMPLE_CLASSES)) if (cls[test] == i).any()]
    out["multiclass_test"] = {
        "labels": SAMPLE_CLASSES, "confusion_matrix": cm.tolist(),
        "classes_present_in_test": [SAMPLE_CLASSES[i] for i in present],
        "macro_f1_present_classes": round(float(f1_score(cls[test], pred[test], average="macro", labels=present, zero_division=0)), 4),
        "per_class_f1": {SAMPLE_CLASSES[i]: round(float(v), 4) for i, v in zip(present, f1_score(cls[test], pred[test], average=None, labels=present, zero_division=0))},
    }
    # the out of fold scores cover every month, so every class is scored at least once
    oof_pred = np.concatenate(P["oof"]).argmax(axis=1)
    out["multiclass_full_year_oof"] = {
        "macro_f1": round(float(f1_score(cls, oof_pred, average="macro", zero_division=0)), 4),
        "per_class_f1": {k: round(float(v), 4) for k, v in zip(SAMPLE_CLASSES, f1_score(cls, oof_pred, average=None, labels=list(range(len(SAMPLE_CLASSES))), zero_division=0))},
        "confusion_matrix": confusion_matrix(cls, oof_pred, labels=list(range(len(SAMPLE_CLASSES)))).tolist(),
    }
    # event level on the detected episodes (out of fold probabilities)
    episodes = read_json(proc / "episodes.json")
    truth = T.episodes(settings)
    weep_truth = [t for t in truth if t["cause"] in WEEP_CAUSES and t["material_weeping"]]
    weep_det = [e for e in episodes if e["kind"] == "weeping"]
    m = match(weep_det, truth)
    found = {}
    for d in weep_det:
        sid = m[d["episode_id"]]
        if sid:
            found.setdefault(sid, []).append(d)
    detected_scen = [t for t in weep_truth if t["scenario_id"] in found]
    delays = []
    for t in detected_scen:
        first = min(pd.Timestamp(d["start"]) for d in found[t["scenario_id"]])
        delays.append((first - pd.Timestamp(t["weep_start"])).total_seconds() / 3600.0)
    # a detected weeping episode is correct if it overlaps a weeping scenario or at least 30 minutes of true weeping samples
    sig = {c: T.signals(settings, c)["class"] for c in store_columns}
    tp_eps, fp_eps, unscheduled = 0, 0, 0
    for d in weep_det:
        sid = m[d["episode_id"]]
        t = next((x for x in truth if x["scenario_id"] == sid), None)
        if t is not None and t["cause"] in WEEP_CAUSES:
            tp_eps += 1
            continue
        s = sig[d["column"]].loc[d["start"]:d["end"]]
        if (s == "weeping").sum() * int(settings["simulation"]["step_minutes"]) >= 30:
            tp_eps += 1
            unscheduled += 1
        else:
            fp_eps += 1
    per_cause = {}
    for cause in WEEP_CAUSES:
        ts = [t for t in weep_truth if t["cause"] == cause]
        per_cause[cause] = {"scenarios": len(ts), "detected": sum(t["scenario_id"] in found for t in ts)}
    lookalike_truth = [t for t in truth if t["cause"] not in WEEP_CAUSES]
    la_det = [e for e in episodes if e["kind"] != "weeping"]
    m2 = match(la_det, truth)
    la_found = {m2[d["episode_id"]] for d in la_det if m2[d["episode_id"]]}
    misfire = sum(1 for t in lookalike_truth if t["scenario_id"] in found)
    out["event_level"] = {
        "weeping_scenarios_with_material_weeping": len(weep_truth),
        "scenarios_detected": len(detected_scen),
        "scenario_recall": round(len(detected_scen) / max(len(weep_truth), 1), 4),
        "detected_weeping_episodes": len(weep_det),
        "episode_precision": round(tp_eps / max(len(weep_det), 1), 4),
        "unscheduled_true_weeping_episodes": unscheduled,
        "false_alarm_episodes": fp_eps,
        "false_alarms_per_column_month": round(fp_eps / (len(store_columns) * 12.0), 3),
        "onset_delay_hours_median": round(float(np.median(delays)), 2) if delays else None,
        "onset_delay_hours_p90": round(float(np.percentile(delays, 90)), 2) if delays else None,
        "per_cause": per_cause,
        "lookalike_scenarios": len(lookalike_truth),
        "lookalike_scenarios_detected_as_lookalike": sum(t["scenario_id"] in la_found for t in lookalike_truth),
        "lookalike_scenarios_mistaken_for_weeping": misfire,
    }
    return out


# ---------------------------------------------------------------- root cause
def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def evaluate_root_cause(settings, inv, reports: list[dict]) -> dict:
    truth = {t["scenario_id"]: t for t in T.episodes(settings)}
    m = match([{"episode_id": r["episode_id"], "column": r["column"], "start": r["start"], "end": r["end"]} for r in reports], list(truth.values()))
    rows = []
    for r in reports:
        sid = m[r["episode_id"]]
        if not sid:
            continue
        rows.append((r, truth[sid]))
    out = {"matched_episodes": len(rows)}
    variants = {"signals_and_records": ("signal", "records"), "signals_only": ("signal",), "records_only": ("records",)}
    saved_hist = inv.history
    for name, use in variants.items():
        inv.history = []
        top1 = top3 = 0
        conf, correct, preds, gold = [], [], [], []
        for r, t in rows:
            rep = r if use == ("signal", "records") else inv.investigate(r["episode_id"], use=use)
            ranked = list(rep["root_cause"]["posterior"].items())
            pred = ranked[0][0]
            top1 += pred == t["cause"]
            top3 += t["cause"] in [c for c, _ in ranked[:3]]
            conf.append(ranked[0][1])
            correct.append(pred == t["cause"])
            preds.append(pred)
            gold.append(t["cause"])
        n = max(len(rows), 1)
        res = {"top1_accuracy": round(top1 / n, 4), "top3_accuracy": round(top3 / n, 4),
               "expected_calibration_error": round(_ece(np.array(conf), np.array(correct, dtype=float)), 4),
               "mean_confidence": round(float(np.mean(conf)), 4) if conf else None}
        if name == "signals_and_records":
            res["confusion_matrix"] = {"labels": ALL_CAUSES, "matrix": confusion_matrix(gold, preds, labels=ALL_CAUSES).tolist()}
            res["per_cause_accuracy"] = {c: round(float(np.mean([p == g for p, g in zip(preds, gold) if g == c])), 4) if c in gold else None for c in ALL_CAUSES}
            res["errors"] = [{"episode_id": r["episode_id"], "truth": t["cause"], "predicted": p} for (r, t), p in zip(rows, preds) if p != t["cause"]]
        out[name] = res
    inv.history = saved_hist
    # learned baseline: logistic regression on episode signatures, leave one episode out
    from ..analytics.episodes import SIGNATURE_FEATURES
    keys = SIGNATURE_FEATURES + [f + "_shift" for f in SIGNATURE_FEATURES] + ["pre_feed_min", "pre_pressure_max", "chest_ua_trend_30d"]
    X = np.array([[inv.by_id[r["episode_id"]]["signature"].get(k, 0.0) for k in keys] for r, _ in rows])
    y = np.array([t["cause"] for _, t in rows])
    if len(rows) >= 10:
        preds = []
        for tr_i, te_i in LeaveOneOut().split(X):
            if len(set(y[tr_i])) < 2:
                preds.append(y[tr_i][0])
                continue
            sc = StandardScaler().fit(X[tr_i])
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr_i]), y[tr_i])
            preds.append(clf.predict(sc.transform(X[te_i]))[0])
        out["learned_signature_classifier_loo"] = {"top1_accuracy": round(float(np.mean(np.array(preds) == y)), 4),
                                                   "note": "logistic regression on episode signatures, leave one episode out; needs labelled history"}
    return out


# ---------------------------------------------------------------- impact
def evaluate_impact(settings, reports: list[dict]) -> dict:
    truth = {t["scenario_id"]: t for t in T.episodes(settings)}
    m = match([{"episode_id": r["episode_id"], "column": r["column"], "start": r["start"], "end": r["end"]} for r in reports], list(truth.values()))
    agg: dict[str, dict] = {}
    for r in reports:
        sid = m[r["episode_id"]]
        if not sid or truth[sid]["cause"] not in WEEP_CAUSES:
            continue
        a = agg.setdefault(sid, {"est_cost": 0.0, "est_steam": 0.0, "est_offspec": 0.0, "eff": [], "dur": 0.0})
        a["est_cost"] += r["impact"]["total_cost_gbp"]
        a["est_steam"] += r["impact"]["excess_steam_t"]
        a["est_offspec"] += r["impact"]["offspec_t"]
        a["eff"].append((r["impact"]["efficiency_loss_pct"], r["duration_h"]))
        a["dur"] += r["duration_h"]
    rows = []
    for sid, a in agg.items():
        t = truth[sid]
        eff = sum(e * d for e, d in a["eff"]) / max(sum(d for _, d in a["eff"]), 1e-9)
        rows.append({"scenario_id": sid, "cause": t["cause"], "column": t["column"], "true_cost": t["true_total_cost_gbp"], "est_cost": a["est_cost"],
                     "true_offspec": t["true_extra_offspec_t"], "est_offspec": a["est_offspec"], "true_steam": t["true_extra_steam_t"],
                     "est_steam": a["est_steam"], "true_eff": t["mean_efficiency_loss_pct"], "est_eff": eff})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"scenarios": 0}
    big = df[df["true_cost"].abs() >= 5000]
    ape = (big["est_cost"] - big["true_cost"]).abs() / big["true_cost"].abs()
    from scipy.stats import spearmanr
    out = {
        "scenarios": int(len(df)),
        "total_true_cost_gbp": round(float(df["true_cost"].sum())),
        "total_estimated_cost_gbp": round(float(df["est_cost"].sum())),
        "total_cost_error_pct": round(float((df["est_cost"].sum() - df["true_cost"].sum()) / df["true_cost"].sum() * 100), 2),
        "median_ape_cost_pct_events_over_5k": round(float(ape.median() * 100), 2) if len(ape) else None,
        "share_within_25pct": round(float((ape <= 0.25).mean()), 4) if len(ape) else None,
        "spearman_cost": round(float(spearmanr(df["true_cost"], df["est_cost"]).correlation), 4),
        "offspec_total_true_t": round(float(df["true_offspec"].sum())),
        "offspec_total_est_t": round(float(df["est_offspec"].sum())),
        "efficiency_mae_pct_points": round(float((df["est_eff"] - df["true_eff"]).abs().mean()), 2),
        "by_cause": df.groupby("cause")[["true_cost", "est_cost"]].sum().round(0).to_dict(orient="index"),
        "rows": df.round(2).to_dict(orient="records"),
    }
    return out


# ---------------------------------------------------------------- retrieval
def _relevant_docs(idx, settings, sid: str, links: dict) -> set[str]:
    rel = set()
    for rid in links.get(sid, []) + links.get(sid + "_alarms", []):
        d = idx.source_to_doc.get(rid)
        if d:
            rel.add(d)
    return rel


def _rank_metrics(ranked: list[str], rel: set[str], ks=(5, 10, 20)) -> dict:
    out = {}
    for k in ks:
        hit = len(set(ranked[:k]) & rel)
        out[f"recall@{k}"] = hit / max(len(rel), 1)
        out[f"capped_recall@{k}"] = hit / max(min(len(rel), k), 1)
        out[f"precision@{k}"] = hit / k
    rr = 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    dcg = sum(1.0 / np.log2(i + 1) for i, d in enumerate(ranked[:10], 1) if d in rel)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(rel), 10) + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def evaluate_retrieval(settings, idx) -> dict:
    from ..retrieval.retriever import RERANK_FEATURES, DEFAULT_WEIGHTS
    rcfg = settings["retrieval"]
    links = T.evidence_links(settings)
    truth = T.episodes(settings)
    queries = []
    for t in truth:
        rel = _relevant_docs(idx, settings, t["scenario_id"], links)
        rel = {d for d in rel if not d.startswith("AB")} or rel  # score on human records; alarm bursts are scored separately below
        if not rel:
            continue
        col = t["column"]
        name = settings.column(col)["name"]
        q = f"Why did {col} {name} lose separation efficiency between {t['start'][:10]} and {t['end'][:10]}?"
        queries.append((t, q, rel))
    modes = ["bm25", "dense", "hybrid", "hybrid_graph", "full"]
    res = {m: [] for m in modes}
    # the ablation always uses the engineering default reranker weights; learned weights are scored separately
    saved_weights = dict(idx.weights)
    idx.weights = dict(DEFAULT_WEIGHTS)
    kinds = ["shift_log", "work_order", "operator_action", "lab", "gamma_scan", "moc"]
    for t, q, rel in queries:
        t0, t1 = pd.Timestamp(t["start"]), pd.Timestamp(t["end"])
        for mode in modes:
            hits = idx.search(q, column=t["column"], t0=t0, t1=t1, k=20, mode=mode, kinds=kinds,
                              lookback_h=rcfg["lookback_hours"], lookahead_h=rcfg["lookahead_hours"])
            res[mode].append(_rank_metrics([h.doc.doc_id for h in hits], rel))
    idx.weights = saved_weights
    out = {"queries": len(queries), "modes": {m: {k: round(float(np.mean([r[k] for r in v])), 4) for k in v[0]} for m, v in res.items() if v}}
    # learned reranker on a strict temporal split
    t_end = pd.Timestamp(settings["detector"]["train_end"])
    Xtr, ytr = [], []
    for t, q, rel in queries:
        if pd.Timestamp(t["start"]) >= t_end:
            continue
        t0, t1 = pd.Timestamp(t["start"]), pd.Timestamp(t["end"])
        hits = idx.search(q, column=t["column"], t0=t0, t1=t1, k=120, mode="full", kinds=kinds,
                          lookback_h=rcfg["lookback_hours"], lookahead_h=rcfg["lookahead_hours"])
        for h in hits:
            Xtr.append([h.features.get(f, 0.0) for f in RERANK_FEATURES])
            ytr.append(h.doc.doc_id in rel)
    learned = None
    if len(set(ytr)) == 2:
        sc = StandardScaler().fit(np.array(Xtr))
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(sc.transform(np.array(Xtr)), np.array(ytr))
        coef = clf.coef_[0] / sc.scale_
        learned = {f: float(c) for f, c in zip(RERANK_FEATURES, coef)}
        default = dict(saved_weights)
        test_q = [(t, q, rel) for t, q, rel in queries if pd.Timestamp(t["start"]) >= t_end]
        scores = {}
        for label, w in (("default_weights", DEFAULT_WEIGHTS), ("learned_weights", learned)):
            idx.weights = dict(w)
            rr = []
            for t, q, rel in test_q:
                hits = idx.search(q, column=t["column"], t0=pd.Timestamp(t["start"]), t1=pd.Timestamp(t["end"]), k=20, mode="full", kinds=kinds,
                                  lookback_h=rcfg["lookback_hours"], lookahead_h=rcfg["lookahead_hours"])
                rr.append(_rank_metrics([h.doc.doc_id for h in hits], rel))
            scores[label] = {k: round(float(np.mean([r[k] for r in rr])), 4) for k in rr[0]} if rr else {}
        idx.weights = default
        out["reranker_temporal_split"] = {"train_queries": int(sum(pd.Timestamp(t["start"]) < t_end for t, _, _ in queries)),
                                          "test_queries": len(test_q), **scores, "learned_coefficients": {k: round(v, 4) for k, v in learned.items()}}
        write_json(settings.path("models") / "reranker_weights.json", learned)
    # knowledge grounding: does the right guide appear for the cause
    from ..graph.builder import CAUSE_DOCS
    hits_at3 = []
    for t in truth:
        cause = t["cause"]
        q = {"turndown": "trays weeping at low feed rates minimum boilup", "steam_valve_stiction": "steam flow cycling valve sticking sawtooth",
             "reboiler_fouling": "steam valve wide open chest pressure high duty limited", "tray_damage": "separation poor after trip relief valve lifted dP low",
             "pressure_rise": "pressure setpoint raised temperature controller cutting steam", "subcooled_feed": "feed preheater isolated cold feed top section",
             "flooding": "dP high at max rates entrainment", "foaming": "dP erratic level swinging contamination", "analyzer_fault": "analyzer reading high lab disagrees"}[cause]
        hits = idx.search(q, k=3, mode="hybrid", kinds=["knowledge"])
        got = {h.doc.source_ids[0] for h in hits}
        hits_at3.append(bool(got & set(CAUSE_DOCS[cause])))
    out["knowledge_hit@3"] = round(float(np.mean(hits_at3)), 4)
    return out


# ---------------------------------------------------------------- recurrence
def evaluate_recurrence(settings, reports: list[dict]) -> dict:
    truth = {t["scenario_id"]: t for t in T.episodes(settings)}
    m = match([{"episode_id": r["episode_id"], "column": r["column"], "start": r["start"], "end": r["end"]} for r in reports], list(truth.values()))
    items = [(r["family_id"], truth[m[r["episode_id"]]]["family_id"], m[r["episode_id"]]) for r in reports if m[r["episode_id"]]]
    tp = fp = fn = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][2] == items[j][2]:
                continue  # two fragments of one scenario are not a recurrence
            same_pred = items[i][0] == items[j][0]
            same_true = items[i][1] == items[j][1]
            tp += same_pred and same_true
            fp += same_pred and not same_true
            fn += (not same_pred) and same_true
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    fams_true = pd.Series([t["family_id"] for t in truth.values()]).value_counts()
    return {"pairs_true_positive": tp, "pairs_false_positive": fp, "pairs_false_negative": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4), "true_recurring_families": int((fams_true >= 2).sum())}


# ---------------------------------------------------------------- faithfulness
def evaluate_faithfulness(inv, reports: list[dict], seed: int) -> dict:
    from ..agents.base import Claim, EvidenceIndex, EvidenceItem, Fact, FactRegistry
    tot = sum(r["verification"]["claims_total"] for r in reports)
    sup = sum(r["verification"]["claims_supported"] for r in reports)
    nums = sum(r["verification"]["numbers_checked"] for r in reports)
    rng = random.Random(seed)
    counts = {k: [0, 0] for k in ("number_changed", "fake_evidence_id", "out_of_window_evidence", "uncited_claim")}
    clean_flags = [0, 0]
    for r in reports:
        ev = EvidenceIndex()
        for k, v in r["evidence"].items():
            ev.items[k] = EvidenceItem(**v)
        fr = FactRegistry()
        for k, v in r["facts"].items():
            fr.facts[k] = Fact(**v)
        claims = [Claim(**c) for c in r["claims"]]
        t0, t1 = r["start"], r["end"]
        for c in claims:
            ok = inv.verifier.check(c, ev, fr, pd.Timestamp(t0), pd.Timestamp(t1)).ok
            clean_flags[0] += 1
            clean_flags[1] += not ok
        quants = [c for c in claims if c.kind == "quantity"]
        if quants:
            c = copy.deepcopy(rng.choice(quants))
            from ..agents.verifier import NUM_RE
            nums_ = [mm for mm in NUM_RE.finditer(c.text) if "GBP" in c.text[max(0, mm.start() - 4):mm.start()] or c.text[mm.end():mm.end() + 2] == " t"]
            if nums_:
                mm = rng.choice(nums_)
                val = abs(float(mm.group(0).replace(",", "").lstrip("+")))
                new = f"{val * 1.37 + 3:,.0f}"
                c.text = c.text[:mm.start()] + new + c.text[mm.end():]
                counts["number_changed"][0] += 1
                counts["number_changed"][1] += not inv.verifier.check(c, ev, fr, pd.Timestamp(t0), pd.Timestamp(t1)).ok
        c = copy.deepcopy(claims[0])
        c.evidence = c.evidence + [f"WO{rng.randint(900000, 999999)}"]
        counts["fake_evidence_id"][0] += 1
        counts["fake_evidence_id"][1] += not inv.verifier.check(c, ev, fr, pd.Timestamp(t0), pd.Timestamp(t1)).ok
        # pick a record from a different month entirely
        other = [x for x in inv.store.log_items.sample(50, random_state=rng.randint(0, 10_000)).itertuples()
                 if abs((x.start - pd.Timestamp(t0)).days) > 20 and x.columns and r["column"] in x.columns]
        if other:
            o = other[0]
            ev.add(o.item_id, "shift_log", o.text, o.start)
            c = copy.deepcopy(claims[0])
            c.kind = "observation"
            c.evidence = [o.item_id]
            c.text = "Operators reported the problem at the time."
            counts["out_of_window_evidence"][0] += 1
            counts["out_of_window_evidence"][1] += not inv.verifier.check(c, ev, fr, pd.Timestamp(t0), pd.Timestamp(t1)).ok
        c = Claim("The column lost separation because of an unrecorded cause.", [], "inference", "summary")
        counts["uncited_claim"][0] += 1
        counts["uncited_claim"][1] += not inv.verifier.check(c, ev, fr, pd.Timestamp(t0), pd.Timestamp(t1)).ok
    return {
        "reports": len(reports), "claims_total": tot, "claims_supported": sup, "citation_coverage": round(sup / max(tot, 1), 4),
        "numbers_checked": nums,
        "false_rejection_rate_on_clean_claims": round(clean_flags[1] / max(clean_flags[0], 1), 4),
        "injected_error_detection": {k: {"injected": v[0], "caught": v[1], "rate": round(v[1] / max(v[0], 1), 4)} for k, v in counts.items()},
    }


# ---------------------------------------------------------------- entity linking
def evaluate_entity_linking(settings, store, linker) -> dict:
    links = T.evidence_links(settings)
    truth = {t["scenario_id"]: t for t in T.episodes(settings)}
    item_col = {}
    for sid, ids in links.items():
        if sid.endswith("_alarms") or sid not in truth:
            continue
        for i in ids:
            if i.startswith("SL"):
                item_col[i] = truth[sid]["column"]
    items = store.log_items.set_index("item_id")
    correct = linked = 0
    alias_forms = 0
    for iid, col in item_col.items():
        text = items.loc[iid, "text"]
        ex = linker.extract(text)
        linked += bool(ex.columns)
        correct += col in ex.columns
        alias_forms += any(m.surface != m.entity for m in ex.mentions)
    n = max(len(item_col), 1)
    return {"scenario_log_items": len(item_col), "items_with_a_column_link": round(linked / n, 4), "correct_column_linked": round(correct / n, 4),
            "items_using_alias_or_variant_spelling": round(alias_forms / n, 4)}


# ---------------------------------------------------------------- runner
def run_suite(settings) -> dict:
    from ..pipeline import load_runtime
    metrics = settings.path("metrics")
    inv = load_runtime(settings)
    reports = read_json(settings.path("reports") / "all_reports.json")
    results = {}
    results["detection"] = evaluate_detection(settings, inv.store.columns)
    write_json(metrics / "detection.json", results["detection"])
    results["root_cause"] = evaluate_root_cause(settings, inv, reports)
    write_json(metrics / "root_cause.json", results["root_cause"])
    results["impact"] = evaluate_impact(settings, reports)
    write_json(metrics / "impact.json", results["impact"])
    results["retrieval"] = evaluate_retrieval(settings, inv.index)
    write_json(metrics / "retrieval.json", results["retrieval"])
    results["recurrence"] = evaluate_recurrence(settings, reports)
    write_json(metrics / "recurrence.json", results["recurrence"])
    results["faithfulness"] = evaluate_faithfulness(inv, reports, settings.seed)
    write_json(metrics / "faithfulness.json", results["faithfulness"])
    results["entity_linking"] = evaluate_entity_linking(settings, inv.store, inv.linker)
    write_json(metrics / "entity_linking.json", results["entity_linking"])
    write_summary(metrics, results)
    LOG.info("evaluation complete")
    return results


def write_summary(folder, R: dict) -> None:
    d, rc, im, rt, rr, f, el = (R["detection"], R["root_cause"], R["impact"], R["retrieval"], R["recurrence"], R["faithfulness"], R["entity_linking"])
    lines = ["# Evaluation summary", ""]
    ev = d["event_level"]
    lines.append(f"Weeping scenarios detected: {ev['scenarios_detected']} of {ev['weeping_scenarios_with_material_weeping']} (recall {ev['scenario_recall']:.0%}), "
                 f"episode precision {ev['episode_precision']:.0%}, median onset delay {ev['onset_delay_hours_median']} h.")
    for name, p in d["protocols"].items():
        tp = p["test_period"]
        lines.append(f"{name}: test AP {tp['average_precision']}, AUC {tp['roc_auc']}, F1 {tp['f1']} at threshold {p['threshold']}")
    lines.append(f"Root cause top1 {rc['signals_and_records']['top1_accuracy']:.0%} (signals only {rc['signals_only']['top1_accuracy']:.0%}, records only {rc['records_only']['top1_accuracy']:.0%})")
    if im.get("scenarios"):
        lines.append(f"Impact: total estimated GBP {im['total_estimated_cost_gbp']:,} vs true GBP {im['total_true_cost_gbp']:,} ({im['total_cost_error_pct']:+.1f} percent)")
    for m, v in rt["modes"].items():
        lines.append(f"Retrieval {m}: capped recall@10 {v['capped_recall@10']:.3f}, MRR {v['mrr']:.3f}, nDCG@10 {v['ndcg@10']:.3f}")
    lines.append(f"Recurrence pairwise F1 {rr['f1']}")
    lines.append(f"Verifier coverage {f['citation_coverage']:.1%}; injected error detection " + ", ".join(f"{k} {v['rate']:.0%}" for k, v in f["injected_error_detection"].items()))
    lines.append(f"Entity linking correct column {el['correct_column_linked']:.1%}")
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")
