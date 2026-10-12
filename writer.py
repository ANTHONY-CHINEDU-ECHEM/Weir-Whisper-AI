"""Writer agent: turns the analysis into cited claims.

The offline writer composes every sentence from templates and the fact registry, which
makes reports reproducible and keeps the evaluation deterministic. When the Claude
backend is enabled the writer asks the model for the summary, the root cause explanation
and the recommendations as JSON claims; the verifier then filters them and any section
that comes back empty falls back to the offline text.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..llm.backends import parse_json_block
from ..llm.prompts import INVESTIGATION_TEMPLATE, SYSTEM_INVESTIGATOR
from ..simulator.scenarios import CAUSE_LABELS
from ..simulator.tags import equipment
from .base import Claim, EvidenceIndex, FactRegistry, fmt_num, gbp

LOG = logging.getLogger("weir_whisper.writer")

SECTION_WORD = {"rect": "rectifying (top)", "strip": "stripping (bottom)"}

RECOMMENDATIONS = {
    "turndown": [
        ("Apply the low rate procedure: hold reboiler steam at or above 65 percent of design whenever feed is below 70 percent.", ["KB003"]),
        ("Add a minimum steam limit to the steam to feed ratio block so the procedure does not depend on the panel operator.", ["KB003", "KB015"]),
        ("Alarm on the estimated weep margin falling below 1.00 rather than waiting for a dP alarm.", ["KB001"]),
    ],
    "steam_valve_stiction": [
        ("Overhaul or replace the {steam_valve} trim and packing; packing adjustment alone has not held on this valve.", ["KB004", "KB014"]),
        ("Until the valve is repaired, run {steam_controller} with reduced integral action or in manual during upsets.", ["KB004"]),
        ("Add a valve stiction index for {steam_valve} to the daily control loop performance report.", ["KB004"]),
    ],
    "reboiler_fouling": [
        ("Plan cleaning of {reboiler} before the steam valve saturates, using the duty to chest pressure index as the trigger.", ["KB005"]),
        ("Review the foulant source with the upstream unit and set a fouling index alarm.", ["KB005"]),
    ],
    "tray_damage": [
        ("Repair the damaged tray section at the earliest outage and inspect the remaining trays.", ["KB009"]),
        ("Make a gamma scan mandatory after any trip with a relief valve lift where dP and separation do not recover within two days.", ["KB016", "KB008"]),
        ("Until the repair, keep vapour rates above the original turndown limit and accept the energy penalty.", ["KB009"]),
    ],
    "pressure_rise": [
        ("Replace the {temp_controller} temperature setpoint with a pressure compensated temperature.", ["KB006"]),
        ("Require any MOC that moves {pressure_controller} to move the temperature setpoint in the same change.", ["KB006"]),
        ("Restore overhead condenser {condenser} capacity so pressure can return to design.", ["KB006"]),
    ],
    "subcooled_feed": [
        ("While {preheater} is out of service, raise boilup by the vapour condensed by the cold feed.", ["KB007"]),
        ("Add feed temperature feedforward to the boilup controller.", ["KB007"]),
    ],
    "flooding": [("Treat the estimated flood fraction as a hard constraint during high rate campaigns.", ["KB002", "KB001"])],
    "foaming": [("Trace and remove the contamination source and keep antifoam dosing to procedure.", ["KB010"])],
    "analyzer_fault": [("Recalibrate {top_analyzer} and add an automatic analyzer versus lab bias check before the APC acts on it.", ["KB011"])],
}


class Writer:
    name = "writer"

    def __init__(self, backend):
        self.backend = backend

    def offline_claims(self, ctx: dict) -> list[Claim]:
        ep, imp, dx, rec = ctx["episode"], ctx["impact"], ctx["diagnosis"], ctx["recurrence"]
        col, name = ep["column"], ctx["column_name"]
        eq = equipment(col)
        claims: list[Claim] = []
        start = pd.Timestamp(ep["start"]).strftime("%Y-%m-%d %H:%M")
        end = pd.Timestamp(ep["end"]).strftime("%Y-%m-%d %H:%M")
        dur = ctx["facts"].add("duration_h", ep["duration_h"], "hours", "episode duration from detector segmentation")
        kind_word = "tray weeping" if ep["kind"] == "weeping" else ep["kind"].replace("_", " ")
        sec = SECTION_WORD[imp["worst_section"]]
        claims.append(Claim(f"{col} {name} showed {kind_word} from {start} to {end} ({fmt_num(dur, 1)} hours), worst in the {sec} section.",
                            ["SIG_DETECTOR", "SIG_DP", "SIG_EFFICIENCY"], "observation", "summary", ["duration_h"]))
        p = ctx["facts"].add("posterior_top", dx.posterior[dx.cause], "probability", "evidence weighted Bayesian score")
        top_ids = [e for e, _, _ in dx.support[:3]]
        claims.append(Claim(f"The most probable cause is {dx.label.lower()} (posterior {p:.2f}).", ctx["cite"](top_ids), "inference", "summary", ["posterior_top"]))
        claims.append(Claim(
            f"Estimated cost {gbp(imp['total_cost_gbp'])}: {fmt_num(imp['excess_steam_t'])} t of steam against the normal model, "
            f"{fmt_num(imp['offspec_t'])} t of off specification product and {fmt_num(imp['co2_t'])} t CO2e.", ["SIG_IMPACT"], "quantity", "summary",
            ["total_cost_gbp", "excess_steam_t", "offspec_t", "co2_t"]))
        if rec["earlier_episodes"]:
            claims.append(Claim(
                f"This is occurrence {rec['occurrence_number']} of this cause on {rec['asset']} ({rec['asset_description']}); earlier episodes cost {gbp(rec['earlier_cost_gbp'])}.",
                rec["earlier_episodes"][-3:] + rec["prior_work_orders"][-2:], "history", "summary", ["occurrence_number", "earlier_cost_gbp"]))
        # root cause reasoning
        for e, w, statement in dx.support[:5]:
            claims.append(Claim(statement, ctx["cite"]([e]), "observation", "root_cause"))
        for alt, items in dx.against.items():
            pa = dx.posterior[alt]
            if not items:
                continue
            e, w, statement = items[0]
            if e:
                claims.append(Claim(f"{CAUSE_LABELS[alt]} was considered and rejected (posterior {pa:.2f}): {statement}", ctx["cite"]([e]), "inference", "alternatives"))
            else:
                best = dx.support[0][0] if dx.support else "SIG_DETECTOR"
                claims.append(Claim(f"{CAUSE_LABELS[alt]} was considered (posterior {pa:.2f}) but no signal or record supports it as specifically as the evidence for {dx.label.lower()}.",
                                    ctx["cite"]([best]), "inference", "alternatives"))
        # quantities
        claims.append(Claim(f"Excess reboiler steam {fmt_num(imp['excess_steam_t'])} t, costing {gbp(imp['steam_cost_gbp'])} plus {gbp(imp['carbon_cost_gbp'])} of carbon.",
                            ["SIG_IMPACT"], "quantity", "impact", ["excess_steam_t", "steam_cost_gbp", "carbon_cost_gbp"]))
        claims.append(Claim(f"Off specification product {fmt_num(imp['offspec_t'])} t ({fmt_num(imp['offspec_top_t'])} t overhead, {fmt_num(imp['offspec_bottom_t'])} t bottoms) judged on the {imp['quality_source']}, costing {gbp(imp['quality_cost_gbp'])}.",
                            ["SIG_IMPACT"], "quantity", "impact", ["offspec_t", "offspec_top_t", "offspec_bottom_t", "quality_cost_gbp"]))
        claims.append(Claim(f"Apparent tray efficiency loss {fmt_num(imp['efficiency_loss_pct'], 1)} percent of design in the {sec} section.",
                            ["SIG_EFFICIENCY"], "quantity", "impact", ["efficiency_loss_pct"]))
        # recurrence
        if rec["historical_investigations"]:
            claims.append(Claim(f"The library holds earlier investigations of the same asset or practice: {', '.join(rec['historical_investigations'])}.",
                                rec["historical_investigations"], "history", "recurrence"))
        if rec["prior_work_orders"]:
            claims.append(Claim(f"Earlier work orders on {rec['asset']}: {', '.join(rec['prior_work_orders'][-4:])}.", rec["prior_work_orders"][-4:], "history", "recurrence"))
        # recommendations
        for text, docs in RECOMMENDATIONS.get(dx.cause, []):
            claims.append(Claim(text.format(**eq), docs, "recommendation", "recommendations"))
        return claims

    def llm_claims(self, ctx: dict, evidence: EvidenceIndex, facts: FactRegistry) -> tuple[list[Claim] | None, dict | None]:
        if self.backend.name == "offline":
            return None, None
        ep, dx = ctx["episode"], ctx["diagnosis"]
        ev_lines = []
        for eid, it in list(evidence.items.items())[:90]:
            ev_lines.append(f"{eid} | {it.kind} | {it.time or ''} | {it.text[:220]}")
        fact_lines = [f"{k}: {round(v.value, 3)} {v.unit}" for k, v in facts.facts.items()]
        alts = ", ".join(f"{CAUSE_LABELS[c]} {p:.2f}" for c, p in dx.ranked()[1:4])
        prompt = INVESTIGATION_TEMPLATE.format(
            episode_id=ep["episode_id"], column=ep["column"], column_name=ctx["column_name"], start=ep["start"], end=ep["end"],
            cause_label=dx.label, posterior=dx.posterior[dx.cause], alternatives=alts, facts="\n".join(fact_lines), evidence="\n".join(ev_lines))
        res = self.backend.complete(SYSTEM_INVESTIGATOR, prompt)
        if res is None:
            return None, None
        data = parse_json_block(res.text)
        if not data:
            LOG.warning("model response was not valid JSON, using offline text")
            return None, {"backend": res.backend, "model": res.model, "usage": res.usage, "parsed": False}
        claims = []
        for key, section, kind in (("summary", "summary", "inference"), ("root_cause_explanation", "root_cause", "inference"),
                                   ("recommendations", "recommendations", "recommendation")):
            for c in data.get(key, []):
                if isinstance(c, dict) and c.get("text"):
                    claims.append(Claim(str(c["text"]), [str(x).strip("[]") for x in c.get("evidence", [])], kind, section))
        return claims, {"backend": res.backend, "model": res.model, "usage": res.usage, "parsed": True}
