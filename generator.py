"""Dataset generator: runs the plant, writes every data source and the ground truth.

Outputs under data/raw (or the profile path):

historian/{column}.parquet      five minute process data, one file per column
tag_dictionary.csv              instrument index (tag, role, unit, description)
alarms.jsonl                    DCS alarm journal
operator_actions.jsonl          DCS operator change journal
shift_logs.jsonl                free text shift logs, one record per area and shift
work_orders.jsonl               maintenance management system extract
lab_results.csv                 laboratory information system extract
gamma_scans.jsonl               gamma scan reports
mocs.jsonl                      management of change records
knowledge/                      engineering library, datasheets and procedures
ground_truth/                   scenarios, per sample labels, true costs, evidence map

Only the evaluation code reads ground_truth/.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..config import Settings
from ..utils.io import timed, write_json, write_jsonl
from .alarms import build_alarms
from .hydraulics import derive_geometry
from .knowledge import write_library
from .plant import simulate_column
from .scenarios import SAMPLE_CLASSES, WEEP_CAUSES, build_exogenous, schedule_scenarios
from .tags import tag_dictionary, tag_for
from .text_gen import TextWriter

LOG = logging.getLogger("weir_whisper.generator")

TRUTH_KEYS = ["true_wf_rect", "true_wf_strip", "true_eff_rect", "true_eff_strip", "true_margin_rect",
              "true_margin_strip", "true_flood_rect", "true_flood_strip", "true_xt", "true_xb",
              "true_offspec_t", "true_steam"]


def _run_column(args):
    """Three passes: actual plant, weeping suppressed, and the healthy counterfactual."""
    column, col, geom, ex, ex_healthy, seed, hidden = args
    actual = simulate_column(col, geom, ex, seed, suppress_weep=False, hidden=hidden)
    no_weep = simulate_column(col, geom, ex, seed, suppress_weep=True, hidden=hidden)
    healthy = simulate_column(col, geom, ex_healthy, seed, suppress_weep=True, hidden=hidden)
    return column, actual, no_weep, healthy


def sample_classes(ex, out, col, scenarios, column) -> np.ndarray:
    n = len(ex.feed_mult)
    cls = np.zeros(n, dtype=int)
    wf = np.maximum(out["true_wf_rect"], out["true_wf_strip"])
    flood = np.maximum(out["true_flood_rect"], out["true_flood_strip"])
    for k, sc in enumerate(scenarios):
        if sc.column != column:
            continue
        m = ex.active_id == k
        if sc.cause == "flooding":
            cls[m & (flood > 0.86)] = SAMPLE_CLASSES.index("flooding")
        elif sc.cause == "foaming":
            cls[m & (ex.foam_dp > 1.10)] = SAMPLE_CLASSES.index("foaming")
        elif sc.cause == "analyzer_fault":
            cls[m & (ex.analyzer_bias_top > 0.15 * col["top_impurity_spec"])] = SAMPLE_CLASSES.index("analyzer_fault")
    cls[wf > 0.03] = SAMPLE_CLASSES.index("weeping")
    return cls


def _costs(actual, counter, sl, col, econ, dt_h, prefix):
    steam = float(np.sum(actual["true_steam"][sl] - counter["true_steam"][sl]) * dt_h)
    offspec = float(np.sum(actual["true_offspec_t"][sl] - counter["true_offspec_t"][sl]))
    steam_cost = steam * econ["lp_steam_price_per_t"]
    co2 = steam * econ["steam_co2_t_per_t"]
    carbon = co2 * econ["carbon_price_per_t"]
    quality = offspec * col["downgrade_price_per_t"]
    return {
        f"{prefix}_extra_steam_t": steam, f"{prefix}_extra_offspec_t": offspec, f"{prefix}_steam_cost_gbp": steam_cost,
        f"{prefix}_quality_cost_gbp": quality, f"{prefix}_co2_t": co2, f"{prefix}_carbon_cost_gbp": carbon,
        f"{prefix}_total_cost_gbp": steam_cost + quality + carbon,
    }


def episode_truth(sc, actual, no_weep, healthy, col, econ, index, step_min) -> dict:
    n = len(index)
    per_h = 60 // step_min
    i0 = sc.start_idx
    i1 = min(sc.end_idx + 6 * per_h, n)
    tail = min(sc.end_idx + 12 * per_h, n)
    wf = np.maximum(actual["true_wf_rect"], actual["true_wf_strip"])
    weep = np.where(wf[i0:i1] > 0.03)[0]
    dt_h = step_min / 60.0
    window = slice(i0, tail)
    event = _costs(actual, no_weep if sc.cause == "turndown" else healthy, window, col, econ, dt_h, "true")
    weep_only = _costs(actual, no_weep, window, col, econ, dt_h, "weep")
    e0 = col["murphree_eff"]
    if len(weep):
        sel = slice(i0 + weep[0], i0 + weep[-1] + 1)
        eff = np.minimum(actual["true_eff_rect"][sel], actual["true_eff_strip"][sel])
        eff_loss = float((1 - eff.mean() / e0) * 100)
        sec = "strip" if actual["true_wf_strip"][sel].mean() >= actual["true_wf_rect"][sel].mean() else "rect"
    else:
        eff_loss, sec = 0.0, None
    return {
        **sc.to_dict(),
        "weep_samples": int(len(weep)),
        "weep_minutes": int(len(weep) * step_min),
        "weep_start": index[i0 + weep[0]].isoformat() if len(weep) else None,
        "weep_end": index[i0 + weep[-1]].isoformat() if len(weep) else None,
        "peak_weep_fraction": float(wf[i0:i1].max()) if i1 > i0 else 0.0,
        "weep_section": sec,
        "mean_efficiency_loss_pct": eff_loss,
        "material_weeping": bool(len(weep) * step_min >= 60),
        **event,
        **weep_only,
    }


def generate(settings: Settings, workers: int = 2) -> dict:
    sim = settings["simulation"]
    raw = settings.path("raw")
    (raw / "historian").mkdir(parents=True, exist_ok=True)
    gt_dir = raw / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    step = int(sim["step_minutes"])
    index = pd.date_range(sim["start"], sim["end"], freq=f"{step}min")
    n = len(index)
    rng = np.random.default_rng(settings.seed)
    geoms = {c: derive_geometry(c, settings.column(c)) for c in settings.columns}
    scenarios = schedule_scenarios(settings, geoms, index, rng)
    LOG.info("scheduled %d scenarios over %d columns and %d samples", len(scenarios), len(settings.columns), n)

    exo = {}
    jobs = []
    hidden_all = {}
    for k, column in enumerate(settings.columns):
        ex = build_exogenous(column, settings.column(column), scenarios, n, step, index, np.random.default_rng(settings.seed + 101 + k))
        ex_h = build_exogenous(column, settings.column(column), scenarios, n, step, index, np.random.default_rng(settings.seed + 101 + k),
                               keep_causes={"turndown", "flooding"})
        exo[column] = ex
        hrng = np.random.default_rng(settings.seed + 555 + k)
        hidden = {"hole_area_factor": float(hrng.uniform(0.90, 1.12)), "k2_offset_delta": float(hrng.uniform(-1.5, 1.5)),
                  "reflux_meter_bias": float(hrng.uniform(-0.04, 0.04))}
        hidden_all[column] = hidden
        jobs.append((column, settings.column(column), geoms[column], ex, ex_h, settings.seed + 1000 + k, hidden))
    results = {}
    with timed("plant simulation"):
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for column, actual, no_weep, healthy in pool.map(_run_column, jobs):
                    results[column] = (actual, no_weep, healthy)
        else:
            for job in jobs:
                column, actual, no_weep, healthy = _run_column(job)
                results[column] = (actual, no_weep, healthy)

    hist = {c: results[c][0] for c in settings.columns}
    truth = {c: results[c][0] for c in settings.columns}

    # historian and tag dictionary
    tag_rows = []
    measure_roles = [r for r in hist[settings.columns[0]] if not r.startswith("true_")]
    for column in settings.columns:
        col = settings.column(column)
        tag_rows.extend(tag_dictionary(column, col["name"]))
        frame = {"timestamp": index}
        for role in measure_roles:
            frame[tag_for(column, role)] = np.round(hist[column][role], 4)
        pd.DataFrame(frame).to_parquet(raw / "historian" / f"{column}.parquet", index=False)
        cls = sample_classes(exo[column], truth[column], col, scenarios, column)
        sid = np.array([scenarios[k].scenario_id if k >= 0 else "" for k in exo[column].active_id])
        gt = {"timestamp": index, "class": np.array(SAMPLE_CLASSES)[cls], "scenario_id": sid}
        for key in TRUTH_KEYS:
            gt[key] = np.round(truth[column][key], 5)
        gt["noweep_steam"] = np.round(results[column][1]["true_steam"], 5)
        gt["healthy_steam"] = np.round(results[column][2]["true_steam"], 5)
        gt["healthy_offspec_t"] = np.round(results[column][2]["true_offspec_t"], 5)
        pd.DataFrame(gt).to_parquet(gt_dir / f"signals_{column}.parquet", index=False)
    pd.DataFrame(tag_rows).to_csv(raw / "tag_dictionary.csv", index=False)

    # alarms
    alarms = []
    ref_samples = int(sim["reference_days"] * 1440 / step)
    for column in settings.columns:
        alarms.extend(build_alarms(column, settings.column(column), hist[column], index, ref_samples, 1 + len(alarms)))
    alarms.sort(key=lambda a: a["timestamp"])
    write_jsonl(raw / "alarms.jsonl", alarms)

    # text records
    with timed("text generation"):
        writer = TextWriter(settings, geoms, scenarios, hist, truth, index, np.random.default_rng(settings.seed + 7))
        bundle = writer.build_all()
    write_jsonl(raw / "shift_logs.jsonl", bundle.shift_logs)
    write_jsonl(raw / "work_orders.jsonl", bundle.work_orders)
    write_jsonl(raw / "operator_actions.jsonl", sorted(bundle.operator_actions, key=lambda a: a["timestamp"]))
    write_jsonl(raw / "gamma_scans.jsonl", bundle.gamma_scans)
    write_jsonl(raw / "mocs.jsonl", bundle.mocs)
    pd.DataFrame(bundle.lab_results).to_csv(raw / "lab_results.csv", index=False)

    # alarms inside the scenario window on the same column count as evidence for retrieval truth
    evidence = {k: list(v) for k, v in bundle.evidence.items()}
    alarm_df = pd.DataFrame(alarms)
    alarm_df["ts"] = pd.to_datetime(alarm_df["timestamp"])
    for sc in scenarios:
        m = (alarm_df["column"] == sc.column) & (alarm_df["ts"] >= sc.start) & (alarm_df["ts"] <= sc.end + pd.Timedelta(hours=6))
        evidence.setdefault(sc.scenario_id, [])
        evidence[sc.scenario_id + "_alarms"] = alarm_df.loc[m, "alarm_id"].tolist()

    # knowledge library
    manifest_docs = write_library(settings.path("knowledge"), settings.plant["columns"], geoms)
    write_json(settings.path("knowledge") / "manifest.json", manifest_docs)

    # ground truth
    econ = settings.plant["economics"]
    episodes = [episode_truth(sc, *results[sc.column], settings.column(sc.column), econ, index, step) for sc in scenarios]
    write_json(gt_dir / "episodes.json", episodes)
    write_json(gt_dir / "evidence_links.json", evidence)
    write_json(raw / "plant_geometry.json", {c: g.to_dict() for c, g in geoms.items()})
    write_json(gt_dir / "hidden_plant_deviations.json", hidden_all)

    stats = {
        "profile": settings.profile,
        "period": [index[0].isoformat(), index[-1].isoformat()],
        "step_minutes": step,
        "columns": settings.columns,
        "historian_rows_per_column": n,
        "historian_values": int(n * len(measure_roles) * len(settings.columns)),
        "scenarios": len(scenarios),
        "weeping_scenarios": int(sum(1 for e in episodes if e["cause"] in WEEP_CAUSES)),
        "alarms": len(alarms),
        "shift_logs": len(bundle.shift_logs),
        "shift_log_items": int(sum(len(l["items"]) for l in bundle.shift_logs)),
        "work_orders": len(bundle.work_orders),
        "lab_results": len(bundle.lab_results),
        "gamma_scans": len(bundle.gamma_scans),
        "mocs": len(bundle.mocs),
        "operator_actions": len(bundle.operator_actions),
        "knowledge_documents": len(manifest_docs),
    }
    write_json(raw / "manifest.json", stats)
    LOG.info("dataset written: %s", json.dumps(stats))
    return stats


def write_sample(settings: Settings, days: int = 14) -> None:
    """Write a small committed sample so the repository shows real records without the full dataset."""
    from ..config import ROOT
    from ..utils.io import read_jsonl

    raw = settings.path("raw")
    out = ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    column = settings.columns[1] if len(settings.columns) > 1 else settings.columns[0]
    h = pd.read_parquet(raw / "historian" / f"{column}.parquet")
    h.head(days * 288).to_csv(out / f"historian_{column}_{days}days.csv", index=False)
    for name in ("shift_logs", "work_orders", "alarms", "gamma_scans", "mocs"):
        rows = read_jsonl(raw / f"{name}.jsonl")
        write_jsonl(out / f"{name}_sample.jsonl", rows[:: max(1, len(rows) // 40)][:40])
    pd.read_csv(raw / "lab_results.csv").head(200).to_csv(out / "lab_results_sample.csv", index=False)
    pd.read_csv(raw / "tag_dictionary.csv").to_csv(out / "tag_dictionary.csv", index=False)
