"""Signal analyst agent.

Turns the episode signature (medians of the dimensionless features inside the episode and
their shift from the column reference window) into named observations. Each observation
carries log likelihood ratio contributions over the nine cause hypotheses. The weights
encode the symptom matrix in KB002 and the troubleshooting guides; they are engineering
judgement, not fitted, which keeps the reasoning auditable and lets the system work on a
column that has never had a labelled incident.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..simulator.scenarios import ALL_CAUSES, WEEP_CAUSES
from .base import EvidenceIndex, FactRegistry, fmt_num

LOOKALIKES = ["flooding", "foaming", "analyzer_fault"]


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi, linear between (works for lo greater than hi too)."""
    if hi == lo:
        return float(x >= hi)
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class Observation:
    eid: str
    statement: str
    contributions: dict[str, float] = field(default_factory=dict)
    value: float | None = None
    strength: float = 0.0

    def support(self) -> dict[str, float]:
        return {k: v for k, v in self.contributions.items() if abs(v) > 1e-9}


class SignalAnalyst:
    name = "signal_analyst"

    def analyse(self, ep: dict, design: dict, evidence: EvidenceIndex, facts: FactRegistry) -> list[Observation]:
        s = ep["signature"]
        mix = ep["class_mix"]
        obs: list[Observation] = []

        def add(eid, statement, contrib, value=None, strength=0.0):
            o = Observation(eid, statement, {k: float(v) for k, v in contrib.items()}, value, strength)
            obs.append(o)
            evidence.add(eid, "signal", statement, ep["start"], weight=strength, method="historian soft sensor", value=value)
            return o

        feed = facts.add("feed_pct_of_design", s["feed_ratio"] * 100, "percent", "median feed flow over design feed")
        st = ramp(s["feed_ratio"], 0.78, 0.55)
        hi = ramp(s["feed_ratio"], 1.05, 1.15)
        add("SIG_FEED", f"Median feed was {fmt_num(feed)} percent of design during the episode.",
            {"turndown": 3.0 * st - (1.5 if s["feed_ratio"] > 0.88 else 0.0), "flooding": 1.5 * hi}, s["feed_ratio"], max(st, hi))

        wm = facts.add("weep_margin_est", s["wm_min"], "ratio", "soft sensor minimum section weep margin, design hole area")
        wm_r = facts.add("weep_margin_rect_est", s["wm_rect"], "ratio", "soft sensor rectifying weep margin")
        wm_s = facts.add("weep_margin_strip_est", s["wm_strip"], "ratio", "soft sensor stripping weep margin")
        sw = ramp(s["wm_min"], 1.0, 0.85)
        add("SIG_WEEP_MARGIN", f"Estimated weep margin fell to {wm:.2f} (rectifying {wm_r:.2f}, stripping {wm_s:.2f}); below 1.00 the trays cannot hold their liquid.",
            {**{c: 0.6 * sw for c in WEEP_CAUSES}, **{c: -1.0 * sw for c in LOOKALIKES}}, s["wm_min"], sw)

        dpr = facts.add("dp_rect_resid_pct", s["dp_rect_resid"] * 100, "percent", "rectifying dP versus expected at measured load")
        dps = facts.add("dp_strip_resid_pct", s["dp_strip_resid"] * 100, "percent", "stripping dP versus expected at measured load")
        low_dp = ramp(-s["dp_min_resid"], 0.15, 0.35)
        healthy_margin = ramp(s["wm_min"], 1.05, 1.20)
        add("SIG_DP", f"Section dP versus the expected value at the measured load: rectifying {dpr:+.0f} percent, stripping {dps:+.0f} percent.",
            {"tray_damage": 3.0 * low_dp * healthy_margin - (1.0 if s["dp_min_resid"] > -0.08 else 0.0),
             "foaming": 2.0 * ramp(s["dp_max_resid"], 0.25, 0.6) - (1.5 if s["dp_max_resid"] < 0.15 else 0.0),
             "flooding": 0.8 * ramp(s["dp_max_resid"], 0.05, 0.25)},
            s["dp_min_resid"], max(low_dp, ramp(s["dp_max_resid"], 0.25, 0.6)))

        cv = facts.add("steam_cv_pct", s["steam_cv_1h"] * 100, "percent", "one hour coefficient of variation of reboiler steam flow")
        sr = facts.add("stick_ratio_pct", s["stick_ratio_3h"] * 100, "percent", "share of samples where valve output moved but flow did not")
        osc = ramp(s["steam_cv_1h"], 0.04, 0.12)
        stk = ramp(s["stick_ratio_3h"], 0.04, 0.12)
        add("SIG_STEAM_CYCLING", f"Reboiler steam flow variability was {cv:.1f} percent (one hour coefficient of variation) and the valve output moved without flow response in {sr:.0f} percent of samples.",
            {"steam_valve_stiction": 3.5 * osc + 1.5 * stk - (2.0 if s["steam_cv_1h"] < 0.025 else 0.0)}, s["steam_cv_1h"], max(osc, stk))

        sat = facts.add("valve_saturation_pct", s["op_sat_frac_max"] * 100, "percent", "maximum three hour share of samples with steam valve above 95 percent")
        ua = facts.add("reboiler_ua_shift_pct", s["chest_ua_index_shift"] * 100, "percent", "change in duty to chest pressure index versus reference")
        sv = ramp(s["op_sat_frac_max"], 0.3, 0.8)
        fu = ramp(-s["chest_ua_index_shift"], 0.05, 0.15)
        add("SIG_REBOILER", f"Steam valve above 95 percent output for up to {sat:.0f} percent of the time; reboiler heat transfer index moved {ua:+.0f} percent from reference.",
            {"reboiler_fouling": 3.0 * sv + 2.0 * fu - (2.0 if s["op_sat_frac_max"] < 0.05 and s["chest_ua_index_shift"] > -0.03 else 0.0)},
            s["op_sat_frac_max"], max(sv, fu))

        pr = facts.add("pressure_shift_pct", s["pressure_ratio_shift"] * 100, "percent", "absolute pressure change versus reference")
        ps = ramp(s["pressure_ratio_shift"], 0.06, 0.18)
        add("SIG_PRESSURE", f"Column absolute pressure was {pr:+.0f} percent versus the reference period.",
            {"pressure_rise": 3.5 * ps - (2.0 if s["pressure_ratio_shift"] < 0.03 else 0.0)}, s["pressure_ratio_shift"], ps)

        q = facts.add("feed_q_shift", s["q_feed_shift"], "ratio", "change in feed thermal condition q versus reference")
        tf = facts.add("feed_temp_shift_c", s["feed_temp_dev_shift"] * (design["bottom_temp_design_c"] - design["top_temp_design_c"]), "degC", "feed temperature change versus reference")
        cq = ramp(s["q_feed_shift"], 0.06, 0.2)
        asym = ramp(s["wm_strip"] - s["wm_rect"], 0.1, 0.3)
        add("SIG_FEED_TEMP", f"Feed temperature moved {tf:+.0f} C versus reference, raising feed q by {q:.2f}.",
            {"subcooled_feed": 3.5 * cq + 1.0 * asym * cq - (2.0 if s["q_feed_shift"] < 0.02 else 0.0)}, s["q_feed_shift"], cq)

        trip = 1.0 if (s["pre_feed_min"] < 0.35 and s["pre_pressure_max"] > 1.12) else 0.0
        add("SIG_TRIP", "A feed trip with a pressure spike occurred in the 72 hours before the episode." if trip else
            "No feed trip or pressure spike in the 72 hours before the episode.",
            {"tray_damage": 2.5 if trip else -0.5}, trip, trip)

        fl = facts.add("flood_fraction_est", s["flood_max"], "ratio", "soft sensor flood fraction")
        fs = ramp(s["flood_max"], 0.80, 0.88)
        add("SIG_FLOOD", f"Estimated flood fraction peaked at {fl:.2f}.", {"flooding": 3.0 * fs - (1.5 if s["flood_max"] < 0.75 else 0.0)}, s["flood_max"], fs)

        lvl = facts.add("level_std_pct", s["level_std_1h"], "percent", "one hour standard deviation of sump level")
        ls = ramp(s["level_std_1h"], 2.5, 5.0)
        add("SIG_LEVEL", f"Sump level noise was {lvl:.1f} percent (one hour standard deviation).", {"foaming": 1.5 * ls}, s["level_std_1h"], ls)

        al = facts.add("analyzer_lab_bias_pct", (np.exp(s["ai_lab_logratio"]) - 1) * 100, "percent", "overhead analyzer relative to latest lab result")
        aa = ramp(abs(s["ai_lab_logratio"]), 0.15, 0.45)
        add("SIG_ANALYZER", f"Overhead analyzer read {al:+.0f} percent versus the laboratory.",
            {"analyzer_fault": 4.0 * aa - (2.0 if abs(s["ai_lab_logratio"]) < 0.08 else 0.0)}, s["ai_lab_logratio"], aa)

        eff = facts.add("efficiency_est_pct", min(s["eff_rect_est"], s["eff_strip_est"]) * 100, "percent", "apparent tray efficiency over design, inverse shortcut model")
        add("SIG_EFFICIENCY", f"Apparent tray efficiency fell to {eff:.0f} percent of design in the worse section.", {}, eff, 0.0)

        dm = {c: 0.0 for c in ALL_CAUSES}
        for c in WEEP_CAUSES:
            dm[c] = 1.5 * mix.get("weeping", 0.0)
        for c in LOOKALIKES:
            dm[c] = 2.5 * mix.get(c, 0.0)
        add("SIG_DETECTOR", "Detector class probabilities over the episode: " + ", ".join(f"{k} {v:.2f}" for k, v in mix.items()) + ".", dm,
            mix.get("weeping", 0.0), max(mix.values()))
        return obs
