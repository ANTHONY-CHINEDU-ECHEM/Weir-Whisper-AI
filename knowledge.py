"""Engineering knowledge library written into the dataset.

The library is what the retrieval layer grounds its explanations in: tray hydraulics
fundamentals, a symptom matrix, troubleshooting guides per cause, site operating
procedures, column datasheets generated from the design basis, and three historical
investigation reports from before the simulated year.
"""
from __future__ import annotations

from pathlib import Path

from .hydraulics import TrayGeometry
from .tags import EQUIPMENT_DESCRIPTIONS, equipment

ARTICLES: dict[str, tuple[str, str]] = {}

ARTICLES["KB001_tray_weeping_fundamentals"] = ("Tray weeping fundamentals", """
# Tray weeping fundamentals

## What weeping is
Weeping is the flow of liquid down through the perforations or valve openings of a tray
instead of across the active area and over the outlet weir. It happens when the vapour
velocity through the holes is too low to hold up the liquid head on the tray. A tray that
weeps lightly still works, because most liquid crosses the deck and contacts vapour. As the
weep fraction grows the bypassed liquid skips a stage of contact, the Murphree efficiency
falls, and at the extreme the tray dumps and stops behaving as a stage at all.

## The weep point
The design check compares the actual hole velocity with the minimum hole velocity at the
weep point. In the Coulson and Richardson method the minimum velocity is
u_min = (K2 minus 0.90 times (25.4 minus hole diameter in mm)) divided by the square root
of vapour density, where K2 rises with the clear liquid height (weir height plus weir crest).
The ratio of actual to minimum hole velocity is the weep margin. Above 1.0 the tray holds
its liquid. Between 1.0 and about 0.85 weeping is light. Below about 0.7 it is severe.

## What moves the weep margin
1. Vapour mass rate. Lower boilup gives lower hole velocity. This is why turndown, reboiler
   limitation and an oscillating steam supply all cause weeping.
2. Vapour density. At higher column pressure the same mass of vapour occupies less volume.
   Hole velocity falls in proportion to density while the minimum velocity falls only with
   its square root, so raising pressure erodes the weep margin.
3. Open area. Dislodged valves or missing panels raise the effective hole area, so the same
   vapour passes at a lower velocity. Damaged trays weep at normal rates.
4. Liquid load. A higher weir crest raises the liquid head the vapour must support.
5. Feed thermal condition. A subcooled feed condenses vapour at the feed tray and starves
   the section above it.

## Signatures in plant data
Section differential pressure falls because the liquid that weeps no longer contributes
clear liquid head. The temperature profile flattens as stages are lost. Product impurities
rise and the composition controllers respond by increasing reflux and boilup, which costs
energy. Sump level becomes noisier as dumped liquid arrives in surges.

## Distinguishing weeping from look alikes
Flooding raises dP sharply and is a high load problem. Foaming raises and destabilises dP
at normal load and is linked to contamination. An analyzer drift shows a quality problem
on the analyzer but not in the lab and with no hydraulic change. Weeping is the only one of
these that lowers dP at the same time as separation gets worse.
""")

ARTICLES["KB002_symptom_matrix"] = ("Efficiency loss symptom matrix", """
# Efficiency loss symptom matrix for tray columns

Use this matrix to discriminate causes of separation loss. Each row is a cause and each
column a symptom family. Strong means the symptom is expected and diagnostic. Weak means
it may be present. Against means its presence argues against the cause.

| Cause | Section dP | Weep margin estimate | Steam flow pattern | Steam valve | Pressure | Feed temp | Lab vs analyzer | Text cues |
|---|---|---|---|---|---|---|---|---|
| Turndown low load | Low (strong) | Below 1 (strong) | Low, steady | Normal range | Normal | Normal | Agree | Rate cut, minimum rates |
| Steam valve stiction | Low in troughs | Dips below 1 periodically | Oscillating, square wave | Sawtooth output | Normal | Normal | Agree | Hunting, cycling, sticky valve |
| Reboiler fouling | Low (strong) | Below 1 late in cycle | Low, below setpoint | Pinned near 100 percent | Normal | Normal | Agree | Valve wide open, chest pressure high |
| Tray damage | Low at normal rates | Above 1 (estimate misleads) | Normal | Normal | Spike at trip | Normal | Agree | Trip, relief valve lift, gamma scan |
| Pressure rise | Slightly low | Lower than normal | Lower than normal for rate | Normal | High (strong) | Normal | Agree | Pressure setpoint change, condenser |
| Subcooled feed | Low in top section | Low in top section | Normal or high | Normal | Normal | Low (strong) | Agree | Preheater isolated, cold feed |
| Flooding (not weeping) | High (strong) | Well above 1 | High | High | Normal | Normal | Agree | Max rates, carryover |
| Foaming (not weeping) | High and erratic | Above 1 | Normal | Normal | Normal | Normal | Agree, contaminant found | Antifoam, contamination |
| Analyzer fault (not weeping) | Normal | Above 1 | Slightly high | Normal | Normal | Normal | Disagree (strong) | Analyzer versus lab mismatch |

## Notes
1. Tray damage is the cause most often missed because the weep margin calculated from the
   design hole area looks healthy. The tell is low dP and poor separation at normal load
   after a pressure surge. A gamma scan confirms it.
2. Pressure driven weeping is usually caused by the temperature controller rather than the
   pressure itself. When pressure rises with an uncompensated temperature setpoint the
   controller sees a hotter tray and cuts boilup.
3. Stiction produces intermittent weeping that appears and disappears within an hour. Look
   for the square wave in flow against a sawtooth in valve output.
""")

ARTICLES["KB003_turndown_procedure"] = ("Low rate operation procedure", """
# Site procedure: operating fractionators at reduced throughput

## Purpose
Prevent loss of separation and off specification product when column feed is reduced
below 70 percent of design.

## Requirements
1. Before reducing feed below 70 percent, the panel operator shall place the boilup on a
   minimum steam flow limit equal to 65 percent of design steam, irrespective of the feed
   ratio. The steam to feed ratio must not be left in charge at low rates.
2. Reflux shall be held at or above 70 percent of design reflux.
3. If section dP falls below its low alarm, increase boilup first and reflux second.
4. Record the start and end of low rate operation in the shift log.

## Background
Our trays are designed for a turndown of about 60 percent of design vapour. Below this
the trays weep and lose efficiency. Running extra boilup at low rates costs steam but
prevents off specification product, which costs more. Several investigations have found
that this procedure was not followed during scheduling driven rate cuts.
""")

ARTICLES["KB004_control_valve_stiction"] = ("Control valve stiction guide", """
# Diagnosing control valve stiction on reboiler steam

## Mechanism
Stiction is static friction in the valve stem packing, guides or trim. The controller output
must move by more than the deadband before the stem breaks free. When it does, it jumps
past the desired position (slip jump). Under integral control this produces a limit cycle:
the output ramps (sawtooth) while flow stays still, then flow jumps (square wave).

## Effect on a column
On a reboiler, stiction makes boilup oscillate. In each trough the vapour rate can fall
below the weep point, so trays weep intermittently, separation degrades and the composition
controllers chase the disturbance. At the peaks the column may approach its flood limit.

## Data signature
1. Flow against valve output shows a parallelogram shaped plot.
2. Flow is constant for several samples and then steps, while output ramps.
3. Oscillation period is usually 20 to 90 minutes on steam loops.
4. Oscillation stops when the controller is put in manual.

## Fix
Short term: put the loop in manual or reduce integral action. Long term: repair packing,
replace positioner or trim. Recurrence after packing adjustment alone is common; a valve
that fails three times should be overhauled or replaced at the next opportunity.
""")

ARTICLES["KB005_reboiler_fouling"] = ("Reboiler fouling guide", """
# Reboiler fouling and boilup limitation

## Mechanism
Deposits on the tube side or shell side reduce the overall heat transfer coefficient. To
deliver the same duty the reboiler needs a higher temperature difference, which shows up
as a rising steam chest pressure. Once the chest pressure approaches the steam supply
pressure, the control valve opens fully and duty becomes capped.

## Data signature
1. Steam valve output rises slowly over weeks at constant duty.
2. Steam chest pressure trends upward.
3. Eventually the valve pins at 100 percent and steam flow falls below setpoint.
4. Boilup becomes insufficient, the trays weep and bottoms purity is lost.

## Action
Plan a clean before the valve saturates. Track the ratio of duty to chest pressure as a
fouling index. A clean typically restores full capacity.
""")

ARTICLES["KB006_pressure_effects"] = ("Column pressure and the weep point", """
# How column pressure moves the weep point

Raising pressure increases vapour density. For a fixed vapour mass rate the volumetric flow
and the hole velocity fall in proportion to density, while the minimum hole velocity at the
weep point falls only with the square root of density. The net effect is that the weep
margin falls with the square root of the pressure ratio (absolute).

The larger and less obvious effect is through temperature control. Most columns control
boilup on a sensitive tray temperature that is not pressure compensated. When pressure
rises the bubble point of the tray liquid rises, the controller sees a hotter tray and cuts
steam. The vapour rate can fall by twenty percent or more, which on a column already near
its turndown limit causes weeping.

Remedies: use a pressure compensated temperature, move the temperature setpoint when the
pressure setpoint is changed through MOC, or fix condenser capacity so the pressure can
return to design.
""")

ARTICLES["KB007_feed_thermal_condition"] = ("Feed thermal condition", """
# Feed thermal condition and section vapour traffic

The feed quality q describes how much of the feed enters as liquid. A saturated liquid
feed has q equal to one. A subcooled feed has q above one, because the cold liquid
condenses some of the rising vapour at the feed tray. The vapour rate in the rectifying
section is then lower than in the stripping section by (q minus 1) times the feed rate.

When a feed preheater is lost, q can rise to 1.3 to 1.6 on naphtha and condensate service.
If the boilup is controlled on a stripping section temperature it does not respond, so the
rectifying section is starved of vapour and weeps. Symptoms are low rectifying section dP,
a rising overhead impurity and a cold feed temperature reading.

Remedy: restore preheat, or increase boilup by the vapour lost to condensation while the
preheater is out of service.
""")

ARTICLES["KB008_gamma_scan_interpretation"] = ("Gamma scan interpretation", """
# Interpreting gamma scans of trayed columns

A gamma scan passes a sealed source and detector down opposite sides of the column and
records transmitted intensity. Liquid and froth absorb radiation, vapour space does not.

1. Normal tray: a sharp density step at the tray deck with a froth zone above and clear
   vapour space below the next tray.
2. Weeping tray: low froth density on the tray and liquid accumulating on the tray below.
3. Missing or displaced panels: no froth signature at the tray location on at least one
   scan line.
4. Flooding: froth fills the tray spacing and the vapour space disappears.
5. Foaming: elevated froth height and diffuse froth interface on many trays.

A scan taken at the operating rate when the problem appears is the most useful. A healthy
scan rules out mechanical damage and points the investigation toward vapour supply or
control problems.
""")

ARTICLES["KB009_tray_damage"] = ("Tray damage after upsets", """
# Tray damage after pressure surges

Rapid depressurisation, high pressure trips, water slugs into hot columns and relief valve
lifts create upward forces large enough to lift valve units or tray panels from their
supports. The column restarts and runs, but the damaged section has a larger effective open
area. Because the design calculation still assumes the original hole area, soft sensors
report a healthy weep margin while the real trays weep.

Signatures: section dP lower than before the upset at the same load, efficiency loss that
does not respond to the usual load changes, and the event in the alarm and shift logs.
Confirm with a gamma scan. Repair requires an outage. Until then, operate at higher vapour
rates where possible and accept higher energy use.
""")

ARTICLES["KB010_foaming"] = ("Foaming in hydrocarbon columns", """
# Foaming

Foaming is caused by surface active contaminants such as amine carryover, corrosion
inhibitors or heavy aromatic slugs. Foam fills the tray spacing, raises and destabilises the
pressure drop, and carries liquid upward. Separation falls and the sump level becomes
erratic. Foaming is not weeping: dP goes up, not down. Confirm with a contaminant screen of
the feed. Antifoam injection and removal of the source are the remedies.
""")

ARTICLES["KB011_analyzer_validation"] = ("Online analyzer validation", """
# Online analyzer validation

Online chromatographs drift when sample conditioning filters block, carrier gas pressure
changes or columns degrade. A positive bias makes the advanced controller believe the
product is off specification and it increases reflux and boilup. The result is an energy
penalty with no real quality problem.

Signature: analyzer reading rises while laboratory results stay on specification, no change
in dP or temperatures beyond what the extra reflux causes. Always compare analyzer and lab
before concluding that separation has been lost.
""")

ARTICLES["KB012_energy_accounting"] = ("Energy and cost accounting for efficiency loss", """
# Accounting for the cost of efficiency loss

The cost of an efficiency loss event has two parts.
1. Energy: additional reboiler steam used to hold product quality, measured against the
   steam a healthy column would have needed at the same feed rate, pressure and feed
   temperature. A normal behaviour model trained on fault free operation provides this
   counterfactual.
2. Quality: product made outside specification multiplied by the downgrade or reprocessing
   cost per tonne.
Carbon cost is the steam penalty multiplied by the emission factor and the carbon price.
Report all three separately and do not net quality losses against energy savings.
""")

ARTICLES["KB013_investigation_method"] = ("Incident investigation evidence standard", """
# Evidence standard for process efficiency investigations

1. Every finding must reference at least one piece of evidence: historian trend, alarm,
   shift log entry, work order, laboratory result, scan report or MOC.
2. Distinguish observation (what the data shows) from inference (why it happened).
3. Quantities must come from calculation, not estimation, and the method must be stated.
4. Consider and explicitly reject alternative causes using the symptom matrix.
5. Check whether the same cause has occurred before on the same equipment. Recurring causes
   need a systemic action, not a repeat of the same repair.
""")

ARTICLES["KB014_hist_investigation_2023_c201"] = ("Historical investigation 2023: C201 steam valve", """
# Historical investigation INV2023-07: C201 depropaniser off specification propane

Summary: In 2023 C201 produced off specification propane for nine days. The cause was
stiction in the reboiler steam control valve FV207 after the gland was overtightened to stop
a steam leak. The valve cycled with a period of about 40 minutes and the stripping trays
wept in each trough. The packing was adjusted, but the problem returned twice within four
months. Action closed: replace FV207 trim and packing with a low friction design. Lesson:
repeated packing adjustment does not cure stiction on this valve.
""")

ARTICLES["KB015_hist_investigation_2024_c401"] = ("Historical investigation 2024: C401 turndown", """
# Historical investigation INV2024-03: C401 C3 splitter propylene downgrade at low rates

Summary: During a cracker outage C401 feed was reduced to 50 percent of design. The steam to
feed ratio controller remained in charge and boilup fell with feed. The trays wept, propylene
purity was lost and 900 tonnes were downgraded to chemical grade. The low rate procedure
requiring a minimum boilup was not applied. Action: add a minimum steam limit to the ratio
block. Status: action open at the time of this report.
""")

ARTICLES["KB016_hist_investigation_2024_c301"] = ("Historical investigation 2024: C301 trip", """
# Historical investigation INV2024-11: C301 deethaniser after a high pressure trip

Summary: After a high pressure trip with relief valve lift, C301 separation never recovered.
Engineers spent three weeks looking at feed composition and analyzers. A gamma scan finally
showed missing panels in the stripping section. Lesson: after any trip with a relief valve
lift, request a gamma scan if dP and separation do not return to pre trip values within two
days.
""")


def datasheet(column: str, col: dict, g: TrayGeometry) -> str:
    eq = equipment(column)
    eq_lines = "\n".join(f"| {tag} | {EQUIPMENT_DESCRIPTIONS[k]} |" for k, tag in eq.items())
    turndown = 1.0 / min(g.weep_margin_rect_design, g.weep_margin_strip_design)
    return f"""
# Column datasheet {column} {col['name']}

Service: {col['service']}.

## Mechanical design
| Item | Value |
|---|---|
| Tray type | {col['tray_type']} |
| Number of trays | {col['n_trays']} (feed on tray {col['feed_tray']} from the top) |
| Rectifying section trays | {g.n_rect} |
| Stripping section trays | {g.n_strip} |
| Internal diameter | {g.diameter_m:.2f} m |
| Tray spacing | {col['tray_spacing_m']:.2f} m |
| Active area | {g.active_area_m2:.2f} m2 |
| Hole (open) area | {g.hole_area_m2:.3f} m2 ({g.hole_area_fraction * 100:.1f} percent of active area) |
| Weir height | {g.weir_height_mm:.0f} mm |
| Weir length | {g.weir_length_m:.2f} m |

## Process design
| Item | Value |
|---|---|
| Design feed | {col['feed_design_t_h']:.1f} t/h |
| Design pressure (operating) | {col['pressure_design_barg']:.1f} barg |
| Design reflux ratio | {col['reflux_ratio_design']:.2f} |
| Design reboiler steam | {g.steam_design_t_h:.1f} t/h |
| Reboiler capacity | {g.steam_capacity_t_h:.1f} t/h |
| Design weep margin (rectifying / stripping) | {g.weep_margin_rect_design:.2f} / {g.weep_margin_strip_design:.2f} |
| Design flood fraction (rectifying / stripping) | {g.flood_rect_design:.2f} / {g.flood_strip_design:.2f} |
| Approximate vapour turndown to weep point | {turndown * 100:.0f} percent of design vapour |
| Overhead specification | {col['top_impurity_spec']:.2f} mol% heavy key |
| Bottoms specification | {col['bottom_impurity_spec']:.2f} mol% light key |
| Off specification cost | {col['downgrade_price_per_t']:.0f} GBP per tonne |

## Equipment
| Tag | Description |
|---|---|
{eq_lines}
"""


def write_library(root: Path, columns: dict, geoms: dict) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for doc_id, (title, body) in ARTICLES.items():
        path = root / f"{doc_id}.md"
        path.write_text(body.strip() + "\n", encoding="utf8")
        manifest.append({"doc_id": doc_id.split("_")[0], "file": path.name, "title": title, "kind": "article"})
    for column, col in columns.items():
        if column not in geoms:
            continue
        path = root / f"DS_{column}.md"
        path.write_text(datasheet(column, col, geoms[column]).strip() + "\n", encoding="utf8")
        manifest.append({"doc_id": f"DS{column}", "file": path.name, "title": f"Column datasheet {column} {col['name']}", "kind": "datasheet", "column": column})
    return manifest
