"""Unstructured and semi structured plant records.

This module writes the messy human side of the dataset: shift logs, maintenance work
orders, laboratory results, gamma scan reports and management of change records. Text
is assembled from large phrase banks with tag aliasing, abbreviations, occasional typos
and deliberate misattribution so that retrieval and entity linking face the same
problems they face on a real site. Every generated item that relates to a scenario is
recorded in a separate evidence map that only the evaluation code reads.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tags import equipment

CREWS = ["A", "B", "C", "D"]
OPERATORS = {
    "AREA2": {"A": ["JMK", "RTD"], "B": ["SLH", "PDW"], "C": ["AOC", "NBV"], "D": ["KEF", "TGL"]},
    "AREA3": {"A": ["DMR", "CJP"], "B": ["LWS", "HYA"], "C": ["FNE", "BKO"], "D": ["GRU", "MIT"]},
}


def crew_for(ts: pd.Timestamp) -> str:
    # four crew continental rota: two days, two nights, four off
    day_index = (ts.normalize() - pd.Timestamp("2025-01-06")).days
    slot = (day_index // 2) % 4
    if ts.hour >= 19 or ts.hour < 7:
        slot = (slot + 1) % 4
    return CREWS[slot]


def alias(tag: str, rng: np.random.Generator) -> str:
    """Render an equipment or instrument tag the way an operator might type it."""
    letters = "".join(ch for ch in tag if ch.isalpha())
    digits = tag[len(letters):]
    form = rng.random()
    if form < 0.55:
        return tag
    if form < 0.72:
        return f"{letters}-{digits}"
    if form < 0.86:
        return tag.lower()
    return f"{letters} {digits}"


def typo(text: str, rate: float, rng: np.random.Generator) -> str:
    if rng.random() > rate * 2:
        return text
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
    if not candidates:
        return text
    i = int(rng.choice(candidates))
    w = words[i]
    j = int(rng.integers(1, len(w) - 2))
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def pick(options: list[str], rng: np.random.Generator) -> str:
    return options[int(rng.integers(0, len(options)))]


ROUTINE = [
    "Handover received, all units steady.",
    "Flare quiet all shift.",
    "Cooling water supply temp {cw} C, normal.",
    "{pump} changed over to {pumpb} for running hours equalisation.",
    "Sample points flushed and sample cabinet tidy.",
    "Permit issued for scaffold erection at {exch}.",
    "Fire and gas panel healthy, no inhibits.",
    "Drained water from {drum} boot, small amount only.",
    "Instrument air dew point normal.",
    "Walked the plot, no leaks found.",
    "Steam traps on {col} tracing checked, two passing, tagged.",
    "Nitrogen purge on {drum} confirmed.",
    "Contractor briefing done for lagging work near {col}.",
    "Daily DCS alarm summary reviewed with shift lead.",
    "Rainwater in bund at {exch} pumped out.",
    "{pump} seal pot level topped up.",
    "Housekeeping round complete, no issues.",
    "LP steam header pressure steady at {lp} barg.",
    "Closed out permits from day shift.",
    "Isolation certificate for {pump} returned, pump back on standby.",
]

DISTRACTORS = [
    "{col} reflux pump {pump} small seal weep at gland, monitoring, WO raised.",
    "{col} sight glass on {drum} dirty, cleaned.",
    "{col} bottoms sample slightly hazy, lab informed, repeat sample OK.",
    "{col} level transmitter cross check against gauge glass OK.",
    "{col} insulation damaged on bottoms line, reported.",
    "{col} PDT impulse lines blown down as per routine.",
]

TRIGGER = {
    "turndown": [
        "{col} feed cut to {pct}% of design per scheduling, {reason}.",
        "Rates reduced on {col} to about {pct}%. Reason: {reason}.",
        "Instructed to bring {col} feed down to {flow} t/h ({reason}).",
        "{col} on minimum throughput from {time}, {reason}.",
    ],
    "steam_valve_stiction": [],
    "reboiler_fouling": [],
    "tray_damage": [
        "{col} TRIPPED at {time} on high overhead pressure, {psv} lifted briefly. Unit restarted after checks.",
        "Big upset on {col} around {time}: pressure spike, {psv} popped, feed tripped. Back on line after {hrs} hrs.",
        "{col} shutdown at {time} following high pressure trip. {psv} lifted and reseated. Restart completed.",
    ],
    "pressure_rise_moc": [
        "{col} pressure setpoint raised to {sp} barg per {moc}.",
        "Implemented {moc}: {pic} SP moved up to {sp} barg on {col}.",
    ],
    "pressure_rise_condenser": [
        "{col} overhead pressure creeping up, {cond} struggling in the afternoon heat.",
        "{col} running high on pressure, {cond} approach temp looks poor.",
    ],
    "subcooled_feed": [
        "{pre} feed preheater isolated at {time} due to {reason}. {col} feed now cold, about {tf} C.",
        "Lost {pre} ({reason}), bypassed. Feed to {col} down to {tf} C.",
    ],
    "flooding": [
        "{col} pushed to max rates for cargo nomination, feed {flow} t/h.",
        "Max throughput campaign started on {col}, feed at {pct}% of design.",
    ],
    "foaming": [
        "Lab reports {contaminant} in {col} feed sample.",
        "Suspect {contaminant} in {col} feed, upstream unit having issues.",
    ],
    "analyzer_fault": [],
}

SYMPTOM_WEEP = [
    "{col} dP across {section} section dropped off, {pdi} reading low for the rate.",
    "{col} temperature profile flattening, {tic} wandering.",
    "{col} looks like trays dumping, lost the profile in the {section} section.",
    "{col} {pdi} low and noisy, bottoms level swinging.",
    "Separation poor on {col}, {ai} trending up while rates are low.",
    "{col} sounds like the trays are weeping, dP low, temps sagging.",
    "{col} efficiency down, had to push reflux to hold spec.",
    "{col} {tic} struggling to hold setpoint, steam demand up and down.",
    "{col} {section} section dP way below normal.",
    "{col} not fractionating well, top and bottom temps converging.",
]

SYMPTOM_CAUSE = {
    "steam_valve_stiction": [
        "{fic} steam flow cycling again, OP sawtoothing between {lo} and {hi}%.",
        "{fv} hunting, steam to {reb} swinging by about {amp} t/h.",
        "{col} reboiler steam flow oscillating, period roughly {per} min.",
        "Steam control on {col} poor, {fv} sticking then jumping.",
        "{fic} in auto but flow will not settle, looks like a sticky valve.",
    ],
    "reboiler_fouling": [
        "{fv} wide open on {col} and still struggling to hold {tic}.",
        "{reb} chest pressure high at {chest} barg, steam valve near 100%.",
        "{col} boilup limited, {fic} cannot reach setpoint.",
        "{reb} duty falling off, steam flow down while valve at max.",
        "{col} bottoms temp low with steam valve pinned at {op}%.",
    ],
    "tray_damage": [
        "Since the trip {col} separation poor even at normal rates.",
        "{col} {section} dP lower than before the trip at same feed.",
        "{col} still not right after the upset, profile not recovered.",
        "Gamma scan requested for {col}, suspect tray damage after the trip.",
    ],
    "pressure_rise": [
        "{col} pressure at {p} barg, {tic} SP not adjusted, steam backing off.",
        "{col} running at higher pressure, bottoms temp controller cutting steam.",
        "{pic} holding {p} barg on {col}, steam lower than normal for the rate.",
    ],
    "subcooled_feed": [
        "{col} feed still cold at {tf} C with {pre} out.",
        "{col} top section dP low since feed went cold.",
        "Cold feed on {col}, overhead {ai} drifting up.",
    ],
    "flooding": [
        "{col} dP high at max rates, {pdi} near alarm.",
        "{col} top temp swinging, suspect entrainment at these rates.",
        "{col} carryover suspected, overhead colour off.",
    ],
    "foaming": [
        "{col} dP erratic and spiking, level in sump swinging badly.",
        "{col} behaving like foaming, {pdi} jumping around.",
        "Antifoam injection started on {col} feed.",
    ],
    "analyzer_fault": [
        "{ai} reading high but lab result on spec, suspect analyzer.",
        "APC pushing reflux on {col} because of {ai}, lab disagrees.",
        "{ai} drifting, analyzer tech informed.",
        "{col} reflux high for no obvious reason, {ai} vs lab mismatch.",
    ],
    "turndown": [
        "{col} steam ratio left on auto at low rates, boilup only {pct}% of design.",
        "{col} at low rates, minimum boilup not set.",
    ],
}

RESOLUTION = {
    "turndown": ["{col} rates restored to normal, profile recovered.", "{col} feed back up to {flow} t/h, dP and temps normal again."],
    "steam_valve_stiction": ["{fv} repacked by instruments, steam flow steady now.", "{fic} oscillation gone after valve work.", "{fv} positioner swapped, control much better."],
    "reboiler_fouling": ["{reb} back in service after cleaning, steam valve now {op}%.", "{reb} cleaned, {col} boilup recovered."],
    "tray_damage": ["{col} trays repaired during short outage, separation back to normal."],
    "pressure_rise_moc": ["{col} pressure SP returned to {sp} barg, MOC closed.", "{tic} setpoint adjusted for the higher pressure, profile recovered."],
    "pressure_rise_condenser": ["{cond} backflushed, {col} pressure back to normal.", "{col} pressure down after {cond} clean."],
    "subcooled_feed": ["{pre} back in service, {col} feed temp {tf} C.", "Feed preheat restored on {col}."],
    "flooding": ["{col} rates trimmed back, dP normal."],
    "foaming": ["{col} contamination flushed through, dP steady."],
    "analyzer_fault": ["{ai} recalibrated, now agrees with lab.", "Analyzer tech fixed {ai} sample system, reading OK."],
}

MISATTRIBUTION = [
    "{col} odd behaviour, probably the analyzer playing up again.",
    "{col} possibly foaming, keep an eye on it.",
    "{col} might be the feed composition changing.",
    "{col} upstream unit sending rubbish feed, will ask them.",
]

SECTION_NAME = {"rect": "top", "strip": "bottom"}


@dataclass
class TextBundle:
    shift_logs: list[dict] = field(default_factory=list)
    work_orders: list[dict] = field(default_factory=list)
    lab_results: list[dict] = field(default_factory=list)
    gamma_scans: list[dict] = field(default_factory=list)
    mocs: list[dict] = field(default_factory=list)
    operator_actions: list[dict] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def _fmt(template: str, ctx: dict) -> str:
    try:
        return template.format(**ctx)
    except KeyError:
        out = template
        for k, v in ctx.items():
            out = out.replace("{" + k + "}", str(v))
        return out


class TextWriter:
    def __init__(self, settings, geoms, scenarios, hist: dict, truth: dict, index: pd.DatetimeIndex, rng):
        self.s = settings
        self.geoms = geoms
        self.scenarios = scenarios
        self.hist = hist
        self.truth = truth
        self.index = index
        self.rng = rng
        self.cfg = settings["text"]
        self.bundle = TextBundle()
        self.wo_counter = 410000
        self.lab_counter = 1
        self.oa_counter = 1
        self.step_min = settings["simulation"]["step_minutes"]

    # helpers -------------------------------------------------------------
    def ctx(self, column: str) -> dict:
        eq = equipment(column)
        r = self.rng
        u = column[1]
        return {
            "col": alias(column, r) if r.random() < 0.25 else column,
            "fv": alias(eq["steam_valve"], r),
            "fic": alias(eq["steam_controller"], r),
            "reb": alias(eq["reboiler"], r),
            "cond": alias(eq["condenser"], r),
            "pre": alias(eq["preheater"], r),
            "pic": alias(eq["pressure_controller"], r),
            "tic": alias(eq["temp_controller"], r),
            "ai": alias(eq["top_analyzer"], r),
            "psv": alias(eq["relief_valve"], r),
            "pdi": alias(f"PDI{u}{'10' if r.random() < 0.6 else '09'}", r),
            "pump": eq["reflux_pump_a"],
            "pumpb": eq["reflux_pump_b"],
            "drum": eq["reflux_drum"],
            "exch": eq["condenser"],
        }

    def idx_of(self, t: pd.Timestamp) -> int:
        i = int((t - self.index[0]) / pd.Timedelta(minutes=self.step_min))
        return min(max(i, 0), len(self.index) - 1)

    def window_stats(self, column: str, t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
        i0, i1 = self.idx_of(t0), max(self.idx_of(t1), self.idx_of(t0) + 1)
        h = self.hist[column]
        tr = self.truth[column]
        col = self.s.column(column)
        wf = np.maximum(tr["true_wf_rect"][i0:i1], tr["true_wf_strip"][i0:i1])
        sec = "strip" if tr["true_wf_strip"][i0:i1].mean() >= tr["true_wf_rect"][i0:i1].mean() else "rect"
        steam = h["steam_flow"][i0:i1]
        op = h["steam_valve_op"][i0:i1]
        return {
            "weep_frac": float(np.mean(wf > 0.03)),
            "section": sec,
            "feed_pct": float(np.mean(h["feed_flow"][i0:i1]) / col["feed_design_t_h"] * 100),
            "feed": float(np.mean(h["feed_flow"][i0:i1])),
            "steam_amp": float(np.percentile(steam, 95) - np.percentile(steam, 5)) / 2,
            "op_lo": float(np.percentile(op, 5)),
            "op_hi": float(np.percentile(op, 95)),
            "op_mean": float(np.mean(op)),
            "chest": float(np.mean(h["chest_pressure"][i0:i1])),
            "p": float(np.mean(h["top_pressure"][i0:i1])),
            "tf": float(np.mean(h["feed_temp"][i0:i1])),
        }

    def new_wo_id(self) -> str:
        self.wo_counter += int(self.rng.integers(1, 9))
        return f"WO{self.wo_counter}"

    def link(self, scenario_id: str | None, item_id: str) -> None:
        if scenario_id:
            self.bundle.evidence[scenario_id].append(item_id)

    # shift logs -----------------------------------------------------------
    def build_shift_logs(self) -> None:
        r = self.rng
        areas = self.s.plant["site"]["areas"]
        active_cols = set(self.s.columns)
        shift_starts = pd.date_range(self.index[0].normalize() + pd.Timedelta(hours=7), self.index[-1], freq="12h")
        by_col = defaultdict(list)
        for sc in self.scenarios:
            by_col[sc.column].append(sc)
        seen_trigger: set[str] = set()
        seen_resolution: set[str] = set()
        for t0 in shift_starts:
            t1 = t0 + pd.Timedelta(hours=12)
            shift = "Day" if t0.hour == 7 else "Night"
            for area, cols in areas.items():
                cols = [c for c in cols if c in active_cols]
                if not cols:
                    continue
                crew = crew_for(t0)
                author = pick(OPERATORS[area][crew], r)
                log_id = f"SL{t0.strftime('%Y%m%d')}{shift[0]}{area[-1]}"
                items: list[tuple[str, str | None]] = []
                for column in cols:
                    for sc in by_col[column]:
                        if t1 <= sc.start or t0 >= sc.end + pd.Timedelta(hours=10):
                            continue
                        items.extend(self._scenario_items(sc, t0, t1, seen_trigger, seen_resolution))
                n_bg = int(r.integers(self.cfg["background_items_per_log"][0], self.cfg["background_items_per_log"][1] + 1))
                for _ in range(n_bg):
                    column = pick(cols, r)
                    c = self.ctx(column)
                    c.update({"cw": f"{r.uniform(21, 27):.0f}", "lp": f"{r.uniform(3.4, 3.8):.1f}", "col": column})
                    tmpl = pick(DISTRACTORS, r) if r.random() < 0.18 else pick(ROUTINE, r)
                    items.append((_fmt(tmpl, c), None))
                order = r.permutation(len(items))
                rendered = []
                for k, j in enumerate(order, 1):
                    text, sid = items[j]
                    item_id = f"{log_id}.{k}"
                    text = typo(text, self.cfg["typo_rate"], r)
                    rendered.append({"item_id": item_id, "text": text})
                    self.link(sid, item_id)
                self.bundle.shift_logs.append({
                    "log_id": log_id, "area": area, "shift": shift, "crew": crew, "author": author,
                    "start": t0.isoformat(), "end": t1.isoformat(), "columns": cols, "items": rendered,
                    "text": "\n".join(f"{k}. {it['text']}" for k, it in enumerate(rendered, 1)),
                })

    def _scenario_items(self, sc, t0, t1, seen_trigger, seen_resolution) -> list[tuple[str, str]]:
        r = self.rng
        out = []
        c = self.ctx(sc.column)
        st = self.window_stats(sc.column, max(t0, sc.start), min(t1, sc.end + pd.Timedelta(hours=10)))
        p = sc.params
        c.update({
            "pct": f"{st['feed_pct']:.0f}", "flow": f"{st['feed']:.0f}", "section": SECTION_NAME[st["section"]],
            "lo": f"{st['op_lo']:.0f}", "hi": f"{st['op_hi']:.0f}", "amp": f"{st['steam_amp']:.1f}",
            "per": f"{r.integers(25, 70)}", "chest": f"{st['chest']:.1f}", "op": f"{st['op_mean']:.0f}",
            "p": f"{st['p']:.1f}", "tf": f"{st['tf']:.0f}", "time": (sc.start + pd.Timedelta(minutes=int(r.integers(0, 50)))).strftime("%H:%M"),
            "hrs": f"{p.get('trip_hours', 4):.0f}", "reason": p.get("reason", ""), "contaminant": p.get("contaminant", "contamination"),
        })
        cause_key = sc.cause
        if sc.cause == "pressure_rise":
            cause_key = f"pressure_rise_{p['variant']}"
            c["sp"] = f"{self.hist[sc.column]['pressure_sp'][self.idx_of(sc.start) + 24]:.1f}" if p["variant"] == "moc" else ""
            c["moc"] = p.get("moc_id", "the MOC")
        if sc.cause in ("turndown", "flooding"):
            design = self.s.column(sc.column)["feed_design_t_h"]
            c["pct"] = f"{p['level'] * 100:.0f}"
            c["flow"] = f"{p['level'] * design:.0f}"
        # trigger note on the first shift that sees the scenario
        if sc.scenario_id not in seen_trigger and t0 <= sc.start + pd.Timedelta(hours=2) < t1 + pd.Timedelta(hours=12):
            seen_trigger.add(sc.scenario_id)
            bank = TRIGGER.get(cause_key, [])
            if bank and r.random() < 0.93:
                out.append((_fmt(pick(bank, r), c), sc.scenario_id))
        active = t0 < sc.end and t1 > sc.start
        if active and r.random() < self.cfg["log_item_probability"]:
            if r.random() < self.cfg["misattribution_rate"]:
                out.append((_fmt(pick(MISATTRIBUTION, r), c), sc.scenario_id))
            bank = SYMPTOM_CAUSE.get(sc.cause, [])
            show_cause = bank and (sc.cause != "turndown" or r.random() < 0.35)
            if show_cause and r.random() < 0.75:
                out.append((_fmt(pick(bank, r), c), sc.scenario_id))
            if st["weep_frac"] > 0.05 and r.random() < 0.85:
                out.append((_fmt(pick(SYMPTOM_WEEP, r), c), sc.scenario_id))
        # resolution note
        end_note_time = sc.end
        if sc.scenario_id not in seen_resolution and t0 <= end_note_time < t1:
            seen_resolution.add(sc.scenario_id)
            res_key = cause_key if cause_key in RESOLUTION else sc.cause
            if sc.cause == "pressure_rise" and p["variant"] == "moc":
                c["sp"] = f"{self.s.column(sc.column)['pressure_design_barg']:.1f}"
            if res_key in RESOLUTION and r.random() < 0.85:
                out.append((_fmt(pick(RESOLUTION[res_key], r), c), sc.scenario_id))
        return out

    # operator actions ------------------------------------------------------
    def build_operator_actions(self) -> None:
        r = self.rng
        for sc in self.scenarios:
            eq = equipment(sc.column)
            u = sc.column[1]
            col = self.s.column(sc.column)
            p = sc.params
            acts = []
            if sc.cause in ("turndown", "flooding"):
                lvl = p["level"]
                acts.append((sc.start, f"FI{u}02", "SP", f"{col['feed_design_t_h'] * 0.93:.1f}", f"{col['feed_design_t_h'] * lvl:.1f}", "Feed rate change"))
                acts.append((sc.end, f"FI{u}02", "SP", f"{col['feed_design_t_h'] * lvl:.1f}", f"{col['feed_design_t_h'] * 0.93:.1f}", "Feed rate change"))
            elif sc.cause == "pressure_rise" and p["variant"] == "moc":
                new = col["pressure_design_barg"] * 0 + (col["pressure_design_barg"] + 1.013) * p["pabs_ratio"] - 1.013
                acts.append((sc.start, eq["pressure_controller"], "SP", f"{col['pressure_design_barg']:.1f}", f"{new:.1f}", "Pressure setpoint change"))
                acts.append((sc.end, eq["pressure_controller"], "SP", f"{new:.1f}", f"{col['pressure_design_barg']:.1f}", "Pressure setpoint change"))
            elif sc.cause == "subcooled_feed":
                acts.append((sc.start, eq["preheater"], "STATE", "IN SERVICE", "BYPASSED", "Exchanger bypass"))
                acts.append((sc.end, eq["preheater"], "STATE", "BYPASSED", "IN SERVICE", "Exchanger bypass"))
            elif sc.cause == "tray_damage":
                acts.append((sc.start, f"FI{u}02", "MODE", "AUTO", "TRIP", "Unit trip"))
                acts.append((sc.start + pd.Timedelta(hours=p["trip_hours"]), f"FI{u}02", "MODE", "TRIP", "AUTO", "Unit restart"))
            elif sc.cause == "foaming":
                acts.append((sc.start + pd.Timedelta(hours=float(r.uniform(3, 10))), f"XV{u}20", "STATE", "CLOSED", "OPEN", "Antifoam injection started"))
            elif sc.cause == "steam_valve_stiction":
                acts.append((sc.end, eq["steam_controller"], "MODE", "MAN", "AUTO", "Controller returned to auto after valve repair"))
                for _ in range(int(r.integers(1, 4))):
                    t = sc.start + (sc.end - sc.start) * float(r.uniform(0.15, 0.9))
                    acts.append((t, eq["steam_controller"], "MODE", "AUTO", "MAN", "Operator took steam controller to manual"))
            for t, tag, kind, old, new, desc in acts:
                aid = f"OA{self.oa_counter:06d}"
                self.oa_counter += 1
                self.bundle.operator_actions.append({
                    "action_id": aid, "timestamp": pd.Timestamp(t).floor("min").isoformat(), "column": sc.column,
                    "tag": tag, "kind": kind, "old_value": old, "new_value": new, "description": desc,
                })
                self.link(sc.scenario_id, aid)

    # work orders ----------------------------------------------------------
    def build_work_orders(self) -> None:
        r = self.rng
        family_seen: dict[str, int] = defaultdict(int)
        for sc in self.scenarios:
            eq = equipment(sc.column)
            p = sc.params
            fam_n = family_seen[sc.family_id]
            family_seen[sc.family_id] += 1
            dur = sc.end - sc.start
            items = []
            if sc.cause == "steam_valve_stiction":
                created = sc.start + dur * float(r.uniform(0.08, 0.35))
                findings = [
                    "Packing found overtightened after last steam leak repair. Packing adjusted and valve stroked, deadband now under 1%.",
                    "Positioner feedback arm worn, replaced. Valve signature test shows high friction remaining, recommend overhaul.",
                    "Valve plug and cage galled, stem friction high. Trim replaced, valve stroked OK.",
                    "Repeat failure. Actuator diaphragm leaking and packing hard. Diaphragm replaced, packing replaced with low friction set.",
                ]
                closure = findings[min(fam_n, len(findings) - 1)]
                if fam_n >= 1:
                    closure += f" Note: this valve has failed {fam_n + 1} times this year."
                items.append((created, eq["steam_valve"], "P2", "CM", f"{eq['steam_valve']} reboiler steam control valve hunting",
                               f"{eq['steam_controller']} steam flow cycling with sawtooth OP on {sc.column}. Valve appears to stick and jump. Request valve diagnostics and repair. Operators running in manual at times.",
                               "STICTION", sc.end, closure))
            elif sc.cause == "reboiler_fouling":
                i0, i1 = self.idx_of(sc.start), self.idx_of(sc.end)
                op = self.hist[sc.column]["steam_valve_op"][i0:i1]
                sat = np.where(op > 95)[0]
                created = self.index[i0 + int(sat[0])] if len(sat) else sc.start + dur * 0.7
                items.append((created, eq["reboiler"], "P2", "CM", f"{eq['reboiler']} reboiler duty limited on {sc.column}",
                              f"Steam valve {eq['steam_valve']} at 100% and {eq['steam_controller']} below setpoint. Chest pressure high. Suspect fouled bundle. Request cleaning plan.",
                              "FOULING", sc.end,
                              f"Bundle pulled and hydro jetted. Heavy {p['foulant']} deposits on tubes. Returned to service, duty restored."))
            elif sc.cause == "tray_damage":
                created = sc.start + pd.Timedelta(days=float(r.uniform(1.0, 4.0)))
                items.append((created, eq["column"], "P2", "INSP", f"{sc.column} suspected tray damage after trip",
                              f"Separation and dP on {sc.column} not recovered since high pressure trip and {eq['relief_valve']} lift. Request gamma scan of {sc.column}.",
                              "INSPECTION", created + pd.Timedelta(days=float(r.uniform(2, 6))), "Gamma scan completed, report issued. See scan report."))
                items.append((created + pd.Timedelta(days=7), eq["column"], "P2", "CM", f"{sc.column} tray repair",
                              f"Repair dislodged tray panels and valves on {sc.column} {SECTION_NAME[p['section']]} section identified by gamma scan. Requires column outage.",
                              "MECHANICAL", sc.end, "Column opened during short outage. Missing panels and loose valve units refitted, manways secured. Separation restored."))
            elif sc.cause == "pressure_rise" and p["variant"] == "condenser":
                created = sc.start + dur * float(r.uniform(0.2, 0.5))
                items.append((created, eq["condenser"], "P3", "CM", f"{eq['condenser']} overhead condenser poor performance",
                              f"{sc.column} overhead pressure running high. {eq['condenser']} cooling water approach poor. Request backflush or clean.",
                              "FOULING", sc.end, "Condenser backflushed, significant silt and debris removed. Pressure back to normal."))
            elif sc.cause == "subcooled_feed":
                items.append((sc.start + pd.Timedelta(hours=float(r.uniform(0.5, 4))), eq["preheater"], "P2", "CM", f"{eq['preheater']} feed preheater {p['reason']}",
                               f"{eq['preheater']} isolated and bypassed due to {p['reason']}. {sc.column} running on cold feed. Repair required.",
                               "LEAK" if p["reason"] == "tube leak" else "FOULING", sc.end,
                               f"{eq['preheater']} repaired ({p['reason']}) and returned to service."))
            elif sc.cause == "analyzer_fault":
                created = sc.start + dur * float(r.uniform(0.4, 0.75))
                items.append((created, eq["top_analyzer"], "P3", "CM", f"{eq['top_analyzer']} reading high versus lab",
                              f"{eq['top_analyzer']} reading above lab result by a growing margin. APC increasing reflux. Request calibration and sample system check.",
                              "INSTRUMENT", sc.end, "Sample conditioning filter blocked, replaced. Analyzer recalibrated against certified gas."))
            elif sc.cause == "foaming":
                items.append((sc.start + pd.Timedelta(hours=float(r.uniform(6, 30))), eq["column"], "P3", "ENG", f"{sc.column} foaming investigation",
                               f"Investigate source of {p['contaminant']} in {sc.column} feed. Review antifoam dosing.",
                               "PROCESS", sc.end + pd.Timedelta(days=2), "Upstream source identified and corrected. Antifoam dosing procedure updated."))
            elif sc.cause == "flooding" and r.random() < 0.5:
                items.append((sc.end, eq["column"], "P4", "ENG", f"{sc.column} review flood margin at max rates",
                               f"High dP alarms on {sc.column} during the max rate campaign. Review hydraulic limits before next campaign.",
                               "PROCESS", None, ""))
            elif sc.cause == "turndown" and fam_n >= 1 and r.random() < 0.6:
                items.append((sc.end + pd.Timedelta(hours=8), eq["column"], "P3", "ENG", f"{sc.column} poor separation at low rates",
                               f"{sc.column} lost separation again during the rate cut. Review minimum boilup and turndown procedure.",
                               "PROCESS", None, ""))
            for created, eqp, prio, wtype, short, long_, code, completed, closure in items:
                wid = self.new_wo_id()
                self.bundle.work_orders.append({
                    "wo_id": wid, "created": pd.Timestamp(created).floor("min").isoformat(), "equipment": eqp,
                    "column": sc.column, "priority": prio, "type": wtype, "short_text": short, "long_text": long_,
                    "failure_code": code,
                    "status": "CLOSED" if completed is not None and pd.Timestamp(completed) < self.index[-1] else "OPEN",
                    "completed": pd.Timestamp(completed).floor("min").isoformat() if completed is not None else None,
                    "closure_notes": closure,
                })
                self.link(sc.scenario_id, wid)
        self._background_work_orders()
        self.bundle.work_orders.sort(key=lambda w: w["created"])

    def _background_work_orders(self) -> None:
        r = self.rng
        templates = [
            ("{pump}", "P3", "CM", "{pump} mechanical seal leak", "Small seal leak on {pump}. Swap to standby and replace seal.", "LEAK", "Seal replaced, pump tested OK."),
            ("{pump}", "P4", "PM", "{pump} vibration route", "Routine vibration monitoring of {pump}.", "PM", "Readings within limits."),
            ("{tx}", "P4", "PM", "{tx} transmitter calibration", "Annual calibration of {tx}.", "PM", "As found within tolerance. No adjustment."),
            ("{exch}", "P4", "PM", "{exch} thermography survey", "Routine thermography survey of {exch} and associated piping.", "PM", "No hot spots found."),
            ("{col}", "P4", "INSP", "{col} external corrosion inspection", "Visual inspection of {col} insulation and cladding.", "INSPECTION", "Minor cladding damage, repaired."),
            ("{drum}", "P3", "CM", "{drum} level gauge glass leaking", "Gauge glass on {drum} weeping at bottom gland.", "LEAK", "Gland repacked."),
            ("{valve}", "P4", "PM", "{valve} valve signature test", "Planned diagnostic test of {valve} under routine program.", "PM", "Signature acceptable."),
            ("{psv}", "P3", "PM", "{psv} relief valve recertification", "Statutory recertification of {psv}.", "PM", "Valve bench tested and reset."),
            ("{col}", "P4", "CM", "{col} lagging damaged", "Lagging and cladding damaged near platform on {col}.", "MECHANICAL", "Lagging replaced."),
            ("{ai}", "P4", "PM", "{ai} routine validation", "Monthly validation of {ai} against certified sample.", "PM", "Validation passed."),
        ]
        cols = list(self.s.columns)
        n = int(self.cfg["background_work_orders"])
        span = (self.index[-1] - self.index[0]).total_seconds()
        for _ in range(n):
            column = pick(cols, r)
            eq = equipment(column)
            u = column[1]
            fill = {
                "pump": pick([eq["reflux_pump_a"], eq["reflux_pump_b"], eq["bottoms_pump_a"], eq["bottoms_pump_b"]], r),
                "tx": pick([f"PDI{u}09", f"PDI{u}10", f"TI{u}11", f"FI{u}02", f"LIC{u}14", f"TIC{u}12"], r),
                "exch": pick([eq["reboiler"], eq["condenser"], eq["preheater"]], r), "col": column, "drum": eq["reflux_drum"],
                "valve": pick([eq["steam_valve"], f"LV{u}14", f"FV{u}04"], r), "psv": eq["relief_valve"], "ai": pick([eq["top_analyzer"], eq["bottom_analyzer"]], r),
            }
            eqp, prio, wtype, short, long_, code, closure = templates[int(r.integers(0, len(templates)))]
            created = self.index[0] + pd.Timedelta(seconds=float(r.uniform(0, span)))
            done = created + pd.Timedelta(days=float(r.uniform(0.5, 25)))
            self.bundle.work_orders.append({
                "wo_id": self.new_wo_id(), "created": created.floor("min").isoformat(), "equipment": _fmt(eqp, fill), "column": column,
                "priority": prio, "type": wtype, "short_text": _fmt(short, fill), "long_text": _fmt(long_, fill), "failure_code": code,
                "status": "CLOSED" if done < self.index[-1] else "OPEN", "completed": done.floor("min").isoformat() if done < self.index[-1] else None,
                "closure_notes": closure if done < self.index[-1] else "",
            })

    # laboratory -----------------------------------------------------------
    def build_lab(self) -> None:
        r = self.rng
        times = pd.date_range(self.index[0].normalize() + pd.Timedelta(hours=6), self.index[-1], freq="8h")
        for column in self.s.columns:
            col = self.s.column(column)
            tr = self.truth[column]
            for t in times:
                i = self.idx_of(t)
                for point, key, spec, analysis in (
                    ("overhead", "true_xt", col["top_impurity_spec"], "heavy key in overhead"),
                    ("bottoms", "true_xb", col["bottom_impurity_spec"], "light key in bottoms"),
                ):
                    if r.random() < 0.03:
                        continue  # missed sample
                    val = float(tr[key][i] * (1 + 0.025 * r.normal()))
                    lid = f"LB{self.lab_counter:07d}"
                    self.lab_counter += 1
                    self.bundle.lab_results.append({
                        "sample_id": lid, "timestamp": (t + pd.Timedelta(minutes=int(r.integers(-20, 25)))).isoformat(), "column": column,
                        "sample_point": f"{column} {point}", "analysis": analysis, "value": round(val, 3), "unit": "mol%",
                        "spec_limit": spec, "flag": "HIGH" if val > spec else "OK",
                    })
        for sc in self.scenarios:
            if sc.cause == "foaming":
                for k in range(3):
                    t = sc.start + pd.Timedelta(hours=4 + 10 * k + float(r.uniform(0, 3)))
                    lid = f"LB{self.lab_counter:07d}"
                    self.lab_counter += 1
                    self.bundle.lab_results.append({
                        "sample_id": lid, "timestamp": t.isoformat(), "column": sc.column, "sample_point": f"{sc.column} feed",
                        "analysis": f"contaminant screen ({sc.params['contaminant']})", "value": round(float(r.uniform(25, 140)), 1), "unit": "ppmw",
                        "spec_limit": 10.0, "flag": "HIGH",
                    })
                    self.link(sc.scenario_id, lid)
            if sc.cause == "analyzer_fault":
                eq = equipment(sc.column)
                t = sc.start + (sc.end - sc.start) * float(r.uniform(0.45, 0.8))
                i = self.idx_of(t)
                lid = f"LB{self.lab_counter:07d}"
                self.lab_counter += 1
                lab_val = float(self.truth[sc.column]["true_xt"][i])
                ana = float(self.hist[sc.column]["top_analyzer"][i])
                self.bundle.lab_results.append({
                    "sample_id": lid, "timestamp": t.isoformat(), "column": sc.column, "sample_point": f"{sc.column} overhead",
                    "analysis": f"analyzer validation {eq['top_analyzer']} (lab {lab_val:.2f}, analyzer {ana:.2f})", "value": round(ana - lab_val, 3),
                    "unit": "mol% bias", "spec_limit": 0.1 * self.s.column(sc.column)["top_impurity_spec"], "flag": "HIGH",
                })
                self.link(sc.scenario_id, lid)
        self.bundle.lab_results.sort(key=lambda x: x["timestamp"])

    # gamma scans and MOCs ---------------------------------------------------
    def build_gamma_and_moc(self) -> None:
        r = self.rng
        gs_n = 1
        moc_n = 2601
        for sc in self.scenarios:
            col = self.s.column(sc.column)
            g = self.geoms[sc.column]
            if sc.cause == "tray_damage":
                sec = sc.params["section"]
                lo, hi = (1, g.n_rect) if sec == "rect" else (g.n_rect + 1, g.n_rect + g.n_strip)
                a = int(r.integers(lo, max(lo + 1, hi - 6)))
                bad = list(range(a, min(a + int(r.integers(4, 8)), hi + 1)))
                miss = bad[len(bad) // 2: len(bad) // 2 + 2]
                t = sc.start + pd.Timedelta(days=float(r.uniform(3, 9)))
                gid = f"GS{gs_n:03d}"
                gs_n += 1
                body = (
                    f"Gamma scan report {gid}, {sc.column} {col['name']}, scanned {t.strftime('%Y-%m-%d')} at {self.hist[sc.column]['feed_flow'][self.idx_of(t)]:.0f} t/h feed.\n"
                    f"Scope: full height, two scan lines at 90 degrees, trays 1 to {col['n_trays']}.\n"
                    f"Findings: trays {bad[0]} to {bad[-1]} in the {SECTION_NAME[sec]} section show low froth density and liquid build up on the tray below, "
                    f"consistent with liquid bypassing the active area. Trays {miss[0]} and {miss[-1]} show no froth signature on one scan line, "
                    f"indicating missing or displaced panels. Remaining trays show normal froth heights and clear vapour spaces. No flooding signature.\n"
                    f"Interpretation: mechanical damage consistent with an upward pressure surge. Expect reduced turndown and efficiency in the {SECTION_NAME[sec]} section.\n"
                    f"Recommendation: plan internal inspection and repair at the earliest outage. Until then avoid low vapour rates.")
                self.bundle.gamma_scans.append({"scan_id": gid, "timestamp": t.isoformat(), "column": sc.column, "damaged_trays": bad, "text": body})
                self.link(sc.scenario_id, gid)
            if sc.cause == "steam_valve_stiction" and r.random() < 0.5:
                t = sc.start + (sc.end - sc.start) * float(r.uniform(0.3, 0.7))
                gid = f"GS{gs_n:03d}"
                gs_n += 1
                body = (f"Gamma scan report {gid}, {sc.column} {col['name']}, scanned {t.strftime('%Y-%m-%d')}.\n"
                        f"Scope: trays 1 to {col['n_trays']}, single scan line.\n"
                        f"Findings: all trays show froth heights within the expected band for the current vapour rate. No missing tray signatures, no flooding. "
                        f"Some trays in the stripping section show periodically reduced froth density, consistent with intermittent low vapour rates rather than mechanical damage.\n"
                        f"Interpretation: tray hardware intact. Investigate vapour supply stability from the reboiler.")
                self.bundle.gamma_scans.append({"scan_id": gid, "timestamp": t.isoformat(), "column": sc.column, "damaged_trays": [], "text": body})
                self.link(sc.scenario_id, gid)
            if sc.cause == "pressure_rise" and sc.params["variant"] == "moc":
                mid = f"MOC{moc_n}"
                moc_n += 1
                sc.params["moc_id"] = mid
                eq = equipment(sc.column)
                new = (col["pressure_design_barg"] + 1.013) * sc.params["pabs_ratio"] - 1.013
                self.bundle.mocs.append({
                    "moc_id": mid, "timestamp": (sc.start - pd.Timedelta(days=float(r.uniform(2, 10)))).isoformat(), "column": sc.column,
                    "title": f"Raise {sc.column} operating pressure to {new:.1f} barg",
                    "text": (f"Temporary change: raise {eq['pressure_controller']} setpoint on {sc.column} from {col['pressure_design_barg']:.1f} to {new:.1f} barg "
                             f"to relieve overhead condenser limitation and reduce venting during warm weather. Risk review: relief capacity checked. "
                             f"Temperature controller {eq['temp_controller']} setpoint unchanged. Duration up to two weeks."),
                    "status": "CLOSED",
                })
                self.link(sc.scenario_id, mid)
        # background MOCs and baseline scans add realistic noise to the corpus
        span = (self.index[-1] - self.index[0]).total_seconds()
        bg = [
            "Change antifoam supplier on {col} feed injection skid",
            "Replace {pump} impeller with trimmed impeller",
            "Install temporary bypass around {valve} for maintenance",
            "Update alarm limits on {tx} after rationalisation review",
            "Revise sample frequency for {col} overhead product",
            "Add pressure compensated temperature calculation for {tic} (design study only)",
        ]
        for k in range(10):
            column = pick(list(self.s.columns), r)
            eq = equipment(column)
            u = column[1]
            fill = {"col": column, "pump": eq["bottoms_pump_a"], "valve": f"LV{u}14", "tx": f"PDI{u}10", "tic": eq["temp_controller"]}
            title = _fmt(bg[k % len(bg)], fill)
            self.bundle.mocs.append({
                "moc_id": f"MOC{moc_n}", "timestamp": (self.index[0] + pd.Timedelta(seconds=float(r.uniform(0, span)))).isoformat(),
                "column": column, "title": title, "text": f"{title}. Scope, risk review and approvals recorded. No change to column operating envelope.", "status": "CLOSED",
            })
            moc_n += 1
        for column in self.s.columns:
            if r.random() < 0.5:
                col = self.s.column(column)
                t = self.index[0] + pd.Timedelta(days=float(r.uniform(5, 20)))
                gid = f"GS{gs_n:03d}"
                gs_n += 1
                self.bundle.gamma_scans.append({"scan_id": gid, "timestamp": t.isoformat(), "column": column, "damaged_trays": [],
                                                "text": f"Gamma scan report {gid}, {column} {col['name']}, baseline scan {t.strftime('%Y-%m-%d')}. All trays show normal froth and vapour space signatures. Baseline for future comparison."})
        self.bundle.mocs.sort(key=lambda m: m["timestamp"])
        self.bundle.gamma_scans.sort(key=lambda g: g["timestamp"])

    def build_all(self) -> TextBundle:
        self.build_gamma_and_moc()  # first, so MOC ids exist for log text
        self.build_operator_actions()
        self.build_shift_logs()
        self.build_work_orders()
        self.build_lab()
        return self.bundle
