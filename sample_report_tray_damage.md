# Investigation DE0043: C301 Deethaniser

**C301 Deethaniser: mechanical tray damage after an upset (weeping), GBP 720,956**

Window 2026-03-20 20:55 to 2026-06-22 19:25 (2254.5 h). Detector class: weeping. Writer backend: offline.

## Summary

* C301 Deethaniser showed tray weeping from 2026-03-20 20:55 to 2026-06-22 19:25 (2,254.5 hours), worst in the rectifying (top) section. [SIG_DETECTOR] [SIG_DP] [SIG_EFFICIENCY]
* The most probable cause is mechanical tray damage after an upset (posterior 1.00). [GS003] [SIG_DP] [SIG_TRIP]
* Estimated cost GBP 720,956: -116 t of steam against the normal model, 15,453 t of off specification product and -24 t CO2e. [SIG_IMPACT]

## Why we believe this

* Gamma scan GS003 found damaged trays 4 to 8. [GS003]
* Section dP versus the expected value at the measured load: rectifying -47 percent, stripping +5 percent. [SIG_DP]
* A feed trip with a pressure spike occurred in the 72 hours before the episode. [SIG_TRIP]
* Work order WO410066 raises suspected tray damage on C301. [WO410066]
* Work order WO410070 raises suspected tray damage on C301. [WO410070]

| Hypothesis | Posterior |
|---|---|
| Mechanical tray damage after an upset | 1.000 |
| Low vapour load during feed turndown | 0.000 |
| Subcooled feed after loss of feed preheat | 0.000 |
| Foaming from feed contamination | 0.000 |
| Reboiler steam control valve stiction | 0.000 |

## Alternatives considered

* Low vapour load during feed turndown was considered and rejected (posterior 0.00): Median feed was 93 percent of design during the episode. [SIG_FEED]
* Subcooled feed after loss of feed preheat was considered and rejected (posterior 0.00): Feed temperature moved -1 C versus reference, raising feed q by 0.01. [SIG_FEED_TEMP]
* Foaming from feed contamination was considered and rejected (posterior 0.00): Section dP versus the expected value at the measured load: rectifying -47 percent, stripping +5 percent. [SIG_DP]

## Impact

* Excess reboiler steam -116 t, costing GBP -3,641 plus GBP -1,706 of carbon. [SIG_IMPACT]
* Off specification product 15,453 t (15,355 t overhead, 99 t bottoms) judged on the online analyzer composite, costing GBP 726,303. [SIG_IMPACT]
* Apparent tray efficiency loss 27.4 percent of design in the rectifying (top) section. [SIG_EFFICIENCY]

## Recurrence

* The library holds earlier investigations of the same asset or practice: KB016. [KB016]

## Recommendations

* Repair the damaged tray section at the earliest outage and inspect the remaining trays. [KB009]
* Make a gamma scan mandatory after any trip with a relief valve lift where dP and separation do not recover within two days. [KB016] [KB008]
* Until the repair, keep vapour rates above the original turndown limit and accept the energy penalty. [KB009]

## Evidence timeline

| Time | Source | Record | Detail |
|---|---|---|---|
| 2026-02-24 15:52 | work order | WO410054 | C301 poor separation at low rates. C301 lost separation again during the rate cut. Review minimum boilup and turndown procedure.  |
| 2026-03-20 19:00 | shift log | SL20260320N2.1 | Big upset on C301 aronud 21:15: pressure spike, psv301 popped, feed tripped. Back on line after 5 hrs. |
| 2026-03-20 20:50 | operator action | OA000037 | Unit trip: FI302 MODE AUTO to TRIP |
| 2026-03-20 20:55 | detector | SIG_DETECTOR | Episode DE0043 starts (weeping) |
| 2026-03-20 21:46 | lab | LB0005235 | C301 bottoms light key in bottoms 1.038 mol% (limit 0.3) |
| 2026-03-20 22:02 | lab | LB0005234 | C301 overhead heavy key in overhead 4.209 mol% (limit 1.5) |
| 2026-03-21 01:38 | operator action | OA000038 | Unit restart: FI302 MODE TRIP to AUTO |
| 2026-03-21 06:09 | lab | LB0005236 | C301 overhead heavy key in overhead 1.958 mol% (limit 1.5) |
| 2026-03-21 13:41 | lab | LB0005238 | C301 overhead heavy key in overhead 2.965 mol% (limit 1.5) |
| 2026-03-21 19:00 | shift log | SL20260321N2.4 | C301 still not right after the upset, profile not recovered. |
| 2026-03-21 19:00 | shift log | SL20260321N2.5 | C301 top section dP way below normal. |
| 2026-03-21 22:04 | lab | LB0005240 | C301 overhead heavy key in overhead 3.479 mol% (limit 1.5) |
| 2026-03-22 06:08 | lab | LB0005242 | C301 overhead heavy key in overhead 3.238 mol% (limit 1.5) |
| 2026-03-22 07:00 | shift log | SL20260322D2.1 | C301 still not right after the upset, profile not recovered. |
| 2026-03-22 07:00 | shift log | SL20260322D2.2 | Separation poor on C301, AI315 trending up while rates are low. |
| 2026-03-22 14:24 | lab | LB0005244 | C301 overhead heavy key in overhead 2.8 mol% (limit 1.5) |
| 2026-03-22 15:29 | work order | WO410066 | C301 suspected tray damage after trip. Separation and dP on C301 not recovered since high pressure trip and PSV301 lift. Request gamma scan of C301. Gamma scan completed, report issued. See scan report. |
| 2026-03-22 19:00 | shift log | SL20260322N2.5 | C301 still not right after the upset, profile not recovered. |
| 2026-03-22 22:20 | lab | LB0005246 | C301 overhead heavy key in overhead 3.254 mol% (limit 1.5) |
| 2026-03-23 05:55 | lab | LB0005248 | C301 overhead heavy key in overhead 2.613 mol% (limit 1.5) |
| 2026-03-23 07:00 | shift log | SL20260323D2.1 | C301 temperature profile flattening, TIC 312 wandering. |
| 2026-03-23 14:15 | lab | LB0005250 | C301 overhead heavy key in overhead 2.688 mol% (limit 1.5) |
| 2026-03-23 21:59 | lab | LB0005252 | C301 overhead heavy key in overhead 3.36 mol% (limit 1.5) |
| 2026-03-24 05:52 | lab | LB0005254 | C301 overhead heavy key in overhead 2.809 mol% (limit 1.5) |
| 2026-03-24 07:00 | shift log | SL20260324D2.1 | C301 sounds like the trays are weeping, dP low, temps sagging. |
| 2026-03-24 07:00 | shift log | SL20260324D2.3 | C301 top dP lower than bfeore the trip at same feed. |
| 2026-03-24 14:08 | lab | LB0005256 | C301 overhead heavy key in overhead 3.016 mol% (limit 1.5) |
| 2026-03-24 19:00 | shift log | SL20260324N2.1 | C301 dP across top section dropped off, pdi309 reading low for the rate. |
| 2026-03-24 22:05 | lab | LB0005258 | C301 overhead heavy key in overhead 3.189 mol% (limit 1.5) |
| 2026-03-25 05:40 | lab | LB0005260 | C301 overhead heavy key in overhead 3.394 mol% (limit 1.5) |
| 2026-03-28 14:14 | gamma scan | GS003 | Gamma scan report GS003, C301 Deethaniser, scanned 2026-03-28 at 37 t/h feed. Scope: full height, two scan lines at 90 degrees, trays 1 to 40. Findings: trays 4 to 8 in the top section show low froth density and liquid build up on the tray  |
| 2026-03-29 15:29 | work order | WO410070 | C301 tray repair. Repair dislodged tray panels and valves on C301 top section identified by gamma scan. Requires column outage. Column opened during short outage. Missing panels and loose valve units refitted, manways secured. Separation re |
| 2026-04-23 21:26 | work order | WO410156 | PSV301 relief valve recertification. Statutory recertification of PSV301. Valve bench tested and reset. |
| 2026-06-22 19:25 | detector | SIG_DETECTOR | Episode DE0043 ends |

## Verification

18 of 18 claims passed the verifier (citation coverage 100%); 27 numbers checked against the fact registry; 720 evidence items indexed.
