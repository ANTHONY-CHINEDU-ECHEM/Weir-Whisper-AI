"""Scenario catalogue and scheduler.

A scenario is one physical event on one column: a feed turndown, a sticking steam valve,
a fouling reboiler, damaged trays after a trip, a pressure rise, a subcooled feed, or one
of three confounders (flooding, foaming, analyzer fault) that look like efficiency loss
but are not weeping. Each scenario is converted into exogenous arrays that the plant
model consumes step by step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

WEEP_CAUSES = [
    "turndown",
    "steam_valve_stiction",
    "reboiler_fouling",
    "tray_damage",
    "pressure_rise",
    "subcooled_feed",
]
CONFOUNDERS = ["flooding", "foaming", "analyzer_fault"]
ALL_CAUSES = WEEP_CAUSES + CONFOUNDERS

CAUSE_LABELS = {
    "turndown": "Low vapour load during feed turndown",
    "steam_valve_stiction": "Reboiler steam control valve stiction",
    "reboiler_fouling": "Reboiler fouling limiting boilup",
    "tray_damage": "Mechanical tray damage after an upset",
    "pressure_rise": "Column pressure rise without temperature compensation",
    "subcooled_feed": "Subcooled feed after loss of feed preheat",
    "flooding": "Jet flooding at high throughput",
    "foaming": "Foaming from feed contamination",
    "analyzer_fault": "Online analyzer drift",
}

DURATION_DAYS = {
    "turndown": (1.0, 4.5),
    "steam_valve_stiction": (4.0, 12.0),
    "reboiler_fouling": (42.0, 70.0),
    "tray_damage": (45.0, 95.0),
    "pressure_rise": (3.0, 9.0),
    "subcooled_feed": (2.0, 7.0),
    "flooding": (0.8, 2.5),
    "foaming": (1.0, 3.5),
    "analyzer_fault": (3.0, 8.0),
}

SAMPLE_CLASSES = ["normal", "weeping", "flooding", "foaming", "analyzer_fault"]


@dataclass
class Scenario:
    scenario_id: str
    column: str
    cause: str
    start: pd.Timestamp
    end: pd.Timestamp
    start_idx: int
    end_idx: int
    family_id: str
    params: dict = field(default_factory=dict)

    @property
    def is_weeping_cause(self) -> bool:
        return self.cause in WEEP_CAUSES

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d


def family_for(column: str, cause: str) -> str:
    u = column[1]
    mapping = {
        "turndown": f"FAM{column}_TURNDOWN_PRACTICE",
        "steam_valve_stiction": f"FAM{column}_FV{u}07",
        "reboiler_fouling": f"FAM{column}_E{u}01",
        "tray_damage": f"FAM{column}_TRAYS",
        "pressure_rise": f"FAM{column}_PRESSURE",
        "subcooled_feed": f"FAM{column}_E{u}03",
        "flooding": f"FAM{column}_HIGHRATE",
        "foaming": f"FAM{column}_CONTAMINATION",
        "analyzer_fault": f"FAM{column}_AI{u}15",
    }
    return mapping[cause]


def sample_params(cause: str, col_cfg: dict, geom, rng: np.random.Generator) -> dict:
    if cause == "turndown":
        return {
            "level": float(rng.uniform(0.45, 0.58)),
            "ramp_hours": float(rng.uniform(1.0, 3.0)),
            "reason": str(rng.choice(["low downstream demand", "upstream unit outage", "tankage constraint", "planned rate cut by scheduling"])),
        }
    if cause == "steam_valve_stiction":
        return {"deadband_pct": float(rng.uniform(17.0, 24.0)), "slip_pct": float(rng.uniform(4.0, 7.0))}
    if cause == "reboiler_fouling":
        return {"final_ua_ratio": float(rng.uniform(0.58, 0.65)), "foulant": str(rng.choice(["polymer", "iron sulphide and gum", "salt and coke"]))}
    if cause == "tray_damage":
        return {
            "section": str(rng.choice(["strip", "strip", "rect"])),
            "ah_multiplier": float(rng.uniform(1.6, 2.1)),
            "eff_penalty": float(rng.uniform(0.04, 0.08)),
            "trip_hours": float(rng.uniform(3.0, 5.0)),
        }
    if cause == "pressure_rise":
        return {
            "pabs_ratio": float(rng.uniform(1.22, 1.34)),
            "variant": str(rng.choice(["condenser", "moc"])),
        }
    if cause == "subcooled_feed":
        frac = float(rng.uniform(0.34, 0.44))
        dq = frac * geom.v_rect_design / geom.feed_kg_h
        dT = dq * col_cfg["latent_heat_kj_kg"] / col_cfg["feed_cp_kj_kg_k"]
        return {"delta_t_c": float(dT), "reason": str(rng.choice(["tube leak", "fouled bundle", "heating medium outage"]))}
    if cause == "flooding":
        return {"level": float(rng.uniform(1.13, 1.22)), "ramp_hours": float(rng.uniform(1.0, 2.5))}
    if cause == "foaming":
        return {
            "dp_factor": float(rng.uniform(1.35, 1.8)),
            "eff_penalty": float(rng.uniform(0.10, 0.18)),
            "contaminant": str(rng.choice(["amine carryover", "corrosion inhibitor overdose", "heavy aromatic slug"])),
        }
    if cause == "analyzer_fault":
        return {"bias_fraction_of_spec": float(rng.uniform(0.6, 1.2)), "analyzer": "top"}
    raise KeyError(cause)


def schedule_scenarios(settings, geoms: dict, index: pd.DatetimeIndex, rng: np.random.Generator) -> list[Scenario]:
    sim = settings["simulation"]
    step = pd.Timedelta(minutes=sim["step_minutes"])
    ref_end = index[0] + pd.Timedelta(days=sim["reference_days"])
    last = index[-1] - pd.Timedelta(days=2)
    gap = pd.Timedelta(days=sim["min_gap_days"])
    scenarios: list[Scenario] = []
    counter = 0
    for column in settings.columns:
        plan = sim["schedule"].get(column, {})
        items = []
        for cause, count in plan.items():
            for _ in range(int(count)):
                lo, hi = DURATION_DAYS[cause]
                items.append((cause, float(rng.uniform(lo, hi))))
        items.sort(key=lambda x: -x[1])
        placed: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for cause, dur_days in items:
            dur = pd.Timedelta(days=dur_days)
            span = (last - dur) - ref_end
            if span <= pd.Timedelta(0):
                continue
            ok = False
            for _attempt in range(800):
                start = ref_end + pd.Timedelta(seconds=float(rng.uniform(0, span.total_seconds())))
                start = start.floor(f"{sim['step_minutes']}min")
                end = start + dur
                if all(end + gap <= s or start >= e + gap for s, e in placed):
                    ok = True
                    break
            if not ok:
                continue
            placed.append((start, end))
            counter += 1
            sid = f"SC{counter:03d}"
            params = sample_params(cause, settings.column(column), geoms[column], rng)
            scenarios.append(Scenario(
                scenario_id=sid, column=column, cause=cause, start=start, end=end,
                start_idx=int((start - index[0]) / step), end_idx=int((end - index[0]) / step),
                family_id=family_for(column, cause), params=params,
            ))
    scenarios.sort(key=lambda s: (s.start, s.column))
    # renumber chronologically so identifiers read naturally in reports
    for i, s in enumerate(scenarios, 1):
        s.scenario_id = f"SC{i:03d}"
    return scenarios


def _ramp(n: int, i0: int, i1: int, ramp: int) -> np.ndarray:
    """Trapezoid weight in [0,1]: rises over ramp samples after i0 and falls before i1."""
    w = np.zeros(n)
    i0 = max(i0, 0)
    i1 = min(i1, n)
    if i1 <= i0:
        return w
    idx = np.arange(i0, i1)
    up = np.clip((idx - i0) / max(ramp, 1), 0, 1)
    down = np.clip((i1 - idx) / max(ramp, 1), 0, 1)
    w[i0:i1] = np.minimum(up, down)
    return w


@dataclass
class Exogenous:
    feed_mult: np.ndarray
    feed_temp_offset: np.ndarray
    pressure_ratio: np.ndarray
    pressure_sp_ratio: np.ndarray
    stiction_deadband: np.ndarray
    stiction_slip: np.ndarray
    ua_ratio: np.ndarray
    ah_mult_rect: np.ndarray
    ah_mult_strip: np.ndarray
    eff_pen_rect: np.ndarray
    eff_pen_strip: np.ndarray
    foam_dp: np.ndarray
    foam_eff: np.ndarray
    analyzer_bias_top: np.ndarray
    active_id: np.ndarray       # scenario index active at each step, or -1
    pressure_spike: np.ndarray
    latent_factor: np.ndarray   # slow feed composition drift in latent heat, unknown to the soft sensor
    steam_meter_bias: np.ndarray
    dp_drift_rect: np.ndarray   # differential pressure transmitter drift (impulse line fouling, zero shift)
    dp_drift_strip: np.ndarray


def build_exogenous(column: str, col_cfg: dict, scenarios: list[Scenario], n: int, step_min: int,
                    index: pd.DatetimeIndex, rng: np.random.Generator, keep_causes: set[str] | None = None) -> Exogenous:
    """Convert scenarios into step arrays.

    keep_causes restricts which scenario effects are applied. The healthy counterfactual
    keeps only the business driven feed changes (turndown and high rate campaigns) and
    drops every equipment, control and instrument fault.
    """
    per_h = 60 // step_min
    t_days = np.arange(n) * step_min / 1440.0
    # base feed: slow bounded random walk, daily demand cycle and sensor scale noise
    walk = np.cumsum(rng.normal(0, 0.0025, n))
    walk = walk - pd.Series(walk).rolling(per_h * 24 * 10, min_periods=1, center=True).mean().to_numpy()
    walk = np.clip(walk, -0.07, 0.07)
    daily = 0.022 * np.sin(2 * np.pi * (t_days - 0.3))
    feed = 0.925 + walk + daily
    # short harmless feed dips that should not be mistaken for weeping
    for _ in range(int(n / (per_h * 24 * 9))):
        i0 = int(rng.integers(0, n - per_h * 8))
        feed -= 0.12 * _ramp(n, i0, i0 + int(rng.integers(per_h * 2, per_h * 6)), per_h)
    ambient = 1.0 + 0.010 * np.sin(2 * np.pi * (t_days - 0.4)) + 0.012 * np.sin(2 * np.pi * (index.dayofyear.to_numpy() - 110) / 365.0)

    ex = Exogenous(
        feed_mult=feed, feed_temp_offset=rng.normal(0, 0.4, n).cumsum() * 0.0 + np.zeros(n),
        pressure_ratio=ambient.astype(float), pressure_sp_ratio=np.ones(n), stiction_deadband=np.zeros(n), stiction_slip=np.zeros(n),
        ua_ratio=np.ones(n), ah_mult_rect=np.ones(n), ah_mult_strip=np.ones(n),
        eff_pen_rect=np.zeros(n), eff_pen_strip=np.zeros(n), foam_dp=np.ones(n), foam_eff=np.zeros(n),
        analyzer_bias_top=np.zeros(n), active_id=np.full(n, -1, dtype=int), pressure_spike=np.zeros(n),
        latent_factor=np.ones(n), steam_meter_bias=np.zeros(n), dp_drift_rect=np.zeros(n), dp_drift_strip=np.zeros(n),
    )
    for name in ("dp_drift_rect", "dp_drift_strip"):
        walk = pd.Series(rng.normal(0, 1, n)).cumsum().to_numpy()
        walk = walk / (np.abs(walk).max() + 1e-9) * float(rng.uniform(0.03, 0.07))
        # occasional step changes when an impulse line is blown down or a transmitter is re zeroed
        for _ in range(int(rng.integers(1, 4))):
            j = int(rng.integers(0, n))
            walk[j:] += float(rng.normal(0, 0.025))
        setattr(ex, name, walk)
    # feed composition drifts slowly, so the true latent heat wanders a few percent
    lw = pd.Series(rng.normal(0, 1, n)).rolling(per_h * 24 * 3, min_periods=1).mean().to_numpy()
    lw = lw / (np.std(lw) + 1e-9)
    ex.latent_factor = 1.0 + 0.03 * np.clip(lw, -2.5, 2.5)
    # the steam orifice meter carries a fixed bias plus a slow seasonal drift
    ex.steam_meter_bias = float(rng.uniform(-0.035, 0.035)) + 0.012 * np.sin(2 * np.pi * t_days / float(rng.uniform(90, 200)) + float(rng.uniform(0, 6)))
    # slow feed temperature wander of a degree or two
    ex.feed_temp_offset = np.clip(pd.Series(rng.normal(0, 0.15, n)).cumsum().to_numpy() * 0.05, -2.5, 2.5)

    for k, sc in enumerate(scenarios):
        if sc.column != column:
            continue
        if keep_causes is not None and sc.cause not in keep_causes:
            continue
        i0, i1 = sc.start_idx, min(sc.end_idx, n)
        ex.active_id[i0:i1] = k
        p = sc.params
        if sc.cause in ("turndown", "flooding"):
            ramp = int(p["ramp_hours"] * per_h)
            w = _ramp(n, i0, i1, ramp)
            base = ex.feed_mult.copy()
            ex.feed_mult = base * (1 - w) + (p["level"] + 0.4 * (base - 0.925)) * w
        elif sc.cause == "steam_valve_stiction":
            # stiction develops over the first day, then persists until repair
            w = _ramp(n, i0, i1, per_h * 18)
            w[i0 + per_h * 18:i1] = 1.0
            ex.stiction_deadband[i0:i1] = p["deadband_pct"] * w[i0:i1]
            ex.stiction_slip[i0:i1] = p["slip_pct"] * w[i0:i1]
        elif sc.cause == "reboiler_fouling":
            L = i1 - i0
            x = np.linspace(0, 1, L)
            ex.ua_ratio[i0:i1] = 1.0 - (1.0 - p["final_ua_ratio"]) * x ** 1.35
        elif sc.cause == "tray_damage":
            trip = int(p["trip_hours"] * per_h)
            ex.feed_mult[i0:i0 + trip] = 0.18
            ex.pressure_spike[i0:i0 + per_h // 2] = 0.28
            arr_ah = ex.ah_mult_strip if p["section"] == "strip" else ex.ah_mult_rect
            arr_pen = ex.eff_pen_strip if p["section"] == "strip" else ex.eff_pen_rect
            arr_ah[i0 + per_h:i1] = p["ah_multiplier"]
            arr_pen[i0 + per_h:i1] = p["eff_penalty"]
        elif sc.cause == "pressure_rise":
            ramp = per_h * (20 if p["variant"] == "condenser" else 1)
            w = _ramp(n, i0, i1, ramp)
            ex.pressure_ratio = ex.pressure_ratio * (1 + (p["pabs_ratio"] - 1) * w)
            if p["variant"] == "moc":
                ex.pressure_sp_ratio = ex.pressure_sp_ratio * (1 + (p["pabs_ratio"] - 1) * w)
        elif sc.cause == "subcooled_feed":
            w = _ramp(n, i0, i1, per_h)
            ex.feed_temp_offset = ex.feed_temp_offset - p["delta_t_c"] * w
        elif sc.cause == "foaming":
            w = _ramp(n, i0, i1, per_h * 2)
            ex.foam_dp = 1 + (p["dp_factor"] - 1) * w
            ex.foam_eff = p["eff_penalty"] * w
        elif sc.cause == "analyzer_fault":
            L = i1 - i0
            bias = p["bias_fraction_of_spec"] * col_cfg["top_impurity_spec"]
            ex.analyzer_bias_top[i0:i1] = bias * np.linspace(0, 1, L) ** 0.8
    ex.feed_mult = np.clip(ex.feed_mult, 0.1, 1.3)
    return ex
