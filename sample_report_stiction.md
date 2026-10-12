# Investigation DE0034: C201 Depropaniser

**C201 Depropaniser: reboiler steam control valve stiction (weeping), GBP 317,334**

Window 2026-02-07 15:20 to 2026-02-19 00:05 (272.8 h). Detector class: weeping. Writer backend: offline.

## Summary

* C201 Depropaniser showed tray weeping from 2026-02-07 15:20 to 2026-02-19 00:05 (272.8 hours), worst in the stripping (bottom) section. [SIG_DETECTOR] [SIG_DP] [SIG_EFFICIENCY]
* The most probable cause is reboiler steam control valve stiction (posterior 1.00). [SIG_STEAM_CYCLING] [WO410039] [SL20260209D2.1] [SL20260209N2.2] [SL20260210N2.5] [SL20260211D2.4]
* Estimated cost GBP 317,334: 92 t of steam against the normal model, 5,693 t of off specification product and 19 t CO2e. [SIG_IMPACT]
* This is occurrence 3 of this cause on FV207 (reboiler steam control valve); earlier episodes cost GBP 417,558. [DE0011] [DE0021] [WO410007] [WO410031]

## Why we believe this

* Reboiler steam flow variability was 20.1 percent (one hour coefficient of variation) and the valve output moved without flow response in 14 percent of samples. [SIG_STEAM_CYCLING]
* Work order WO410039 reports FV207 steam valve hunting or sticking. [WO410039]
* 11 record(s) describe cycling: SL20260209D2.1, SL20260209N2.2, SL20260210N2.5, SL20260211D2.4. [SL20260209D2.1] [SL20260209N2.2] [SL20260210N2.5] [SL20260211D2.4]
* Operator action OA000022: steam controller taken to manual. [OA000022]
* Operator action OA000021: steam controller taken to manual. [OA000021]

| Hypothesis | Posterior |
|---|---|
| Reboiler steam control valve stiction | 1.000 |
| Jet flooding at high throughput | 0.000 |
| Low vapour load during feed turndown | 0.000 |
| Reboiler fouling limiting boilup | 0.000 |
| Column pressure rise without temperature compensation | 0.000 |

## Alternatives considered

* Jet flooding at high throughput was considered (posterior 0.00) but no signal or record supports it as specifically as the evidence for reboiler steam control valve stiction. [SIG_STEAM_CYCLING]
* Low vapour load during feed turndown was considered and rejected (posterior 0.00): Median feed was 92 percent of design during the episode. [SIG_FEED]
* Reboiler fouling limiting boilup was considered and rejected (posterior 0.00): Steam valve above 95 percent output for up to 0 percent of the time; reboiler heat transfer index moved +1 percent from reference. [SIG_REBOILER]

## Impact

* Excess reboiler steam 92 t, costing GBP 2,887 plus GBP 1,353 of carbon. [SIG_IMPACT]
* Off specification product 5,693 t (2,268 t overhead, 3,425 t bottoms) judged on the online analyzer composite, costing GBP 313,095. [SIG_IMPACT]
* Apparent tray efficiency loss 5.6 percent of design in the stripping (bottom) section. [SIG_EFFICIENCY]

## Recurrence

* The library holds earlier investigations of the same asset or practice: KB014. [KB014]
* Earlier work orders on FV207: WO410007, WO410031. [WO410007] [WO410031]

## Recommendations

* Overhaul or replace the FV207 trim and packing; packing adjustment alone has not held on this valve. [KB004] [KB014]
* Until the valve is repaired, run FIC207 with reduced integral action or in manual during upsets. [KB004]
* Add a valve stiction index for FV207 to the daily control loop performance report. [KB004]

## Evidence timeline

| Time | Source | Record | Detail |
|---|---|---|---|
| 2025-12-06 17:27 | work order | WO410007 | FV207 reboiler steam control valve hunting. FIC207 steam flow cycling with sawtooth OP on C201. Valve appears to stick and jump. Request valve diagnostics and repair. Operators running in manual at times. Packing found overtightened after l |
| 2026-01-13 08:47 | work order | WO410031 | FV207 reboiler steam control valve hunting. FIC207 steam flow cycling with sawtooth OP on C201. Valve appears to stick and jump. Request valve diagnostics and repair. Operators running in manual at times. Positioner feedback arm worn, repla |
| 2026-02-07 15:20 | detector | SIG_DETECTOR | Episode DE0034 starts (weeping) |
| 2026-02-08 09:39 | work order | WO410039 | FV207 reboiler steam control valve hunting. FIC207 steam flow cycling with sawtooth OP on C201. Valve appears to stick and jump. Request valve diagnostics and repair. Operators running in manual at times. Valve plug and cage galled, stem fr |
| 2026-02-08 19:00 | shift log | SL20260208N2.2 | C201 PDI 210 low and noisy, bottoms level swinging. |
| 2026-02-09 05:41 | lab | LB0002889 | C201 bottoms light key in bottoms 1.126 mol% (limit 0.9) |
| 2026-02-09 05:57 | lab | LB0002888 | C201 overhead heavy key in overhead 1.724 mol% (limit 1.2) |
| 2026-02-09 07:00 | shift log | SL20260209D2.1 | FV207 hunting, steam to E201 swinging by about 3.2 t/h. |
| 2026-02-09 13:57 | lab | LB0002891 | C201 bottoms light key in bottoms 0.902 mol% (limit 0.9) |
| 2026-02-09 14:10 | lab | LB0002890 | C201 overhead heavy key in overhead 1.27 mol% (limit 1.2) |
| 2026-02-09 19:00 | shift log | SL20260209N2.2 | C201 reboiler steam flow oscillating, period rouglhy 26 min. |
| 2026-02-09 19:00 | shift log | SL20260209N2.4 | C201 PDI-209 low and noisy, bottoms level swinging. |
| 2026-02-09 21:55 | lab | LB0002892 | C201 overhead heavy key in overhead 1.515 mol% (limit 1.2) |
| 2026-02-09 22:22 | lab | LB0002893 | C201 bottoms light key in bottoms 1.118 mol% (limit 0.9) |
| 2026-02-10 05:52 | lab | LB0002895 | C201 bottoms light key in bottoms 1.156 mol% (limit 0.9) |
| 2026-02-10 06:02 | lab | LB0002894 | C201 overhead heavy key in overhead 1.665 mol% (limit 1.2) |
| 2026-02-10 13:41 | lab | LB0002897 | C201 bottoms light key in bottoms 1.047 mol% (limit 0.9) |
| 2026-02-10 14:22 | lab | LB0002896 | C201 overhead heavy key in overhead 1.531 mol% (limit 1.2) |
| 2026-02-10 19:00 | shift log | SL20260210N2.3 | C201 pdi209 low and noisy, bottoms level swinging. |
| 2026-02-10 19:00 | shift log | SL20260210N2.5 | FIC207 in auto but flow will not settle, looks like a sticky valve. |
| 2026-02-10 20:15 | operator action | OA000022 | Operator took steam controller to manual: FIC207 MODE AUTO to MAN |
| 2026-02-11 07:00 | shift log | SL20260211D2.3 | C201 sonuds like the trays are weeping, dP low, temps sagging. |
| 2026-02-11 07:00 | shift log | SL20260211D2.4 | FIC207 steam flow cycling again, OP swatoothing between 38 and 68%. |
| 2026-02-11 13:43 | lab | LB0002902 | C201 overhead heavy key in overhead 1.538 mol% (limit 1.2) |
| 2026-02-11 14:17 | lab | LB0002903 | C201 bottoms light key in bottoms 1.051 mol% (limit 0.9) |
| 2026-02-11 19:00 | shift log | SL20260211N2.1 | C201 bottom section dP way below normal. |
| 2026-02-11 19:00 | shift log | SL20260211N2.6 | Steam conrtol on C201 poor, FV 207 sticking then jumping. |
| 2026-02-12 07:00 | shift log | SL20260212D2.2 | C201 looks like trays dumping, lost the profile in the bottom section. |
| 2026-02-12 22:02 | lab | LB0002910 | C201 overhead heavy key in overhead 1.304 mol% (limit 1.2) |
| 2026-02-12 22:07 | lab | LB0002911 | C201 bottoms light key in bottoms 0.962 mol% (limit 0.9) |
| 2026-02-13 17:20 | operator action | OA000021 | Operator took steam controller to manual: FIC207 MODE AUTO to MAN |
| 2026-02-15 01:58 | gamma scan | GS002 | Gamma scan report GS002, C201 Depropaniser, scanned 2026-02-15. Scope: trays 1 to 45, single scan line. Findings: all trays show froth heights within the expected band for the current vapour rate. No missing tray signatures, no flooding. So |
| 2026-02-18 20:30 | operator action | OA000020 | Controller returned to auto after valve repair: FIC207 MODE MAN to AUTO |
| 2026-02-19 00:05 | detector | SIG_DETECTOR | Episode DE0034 ends |

## Verification

20 of 20 claims passed the verifier (citation coverage 100%); 26 numbers checked against the fact registry; 127 evidence items indexed.
