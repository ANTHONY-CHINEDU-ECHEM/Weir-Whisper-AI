"""Dynamic column model.

The model steps every column at the historian resolution (five minutes by default) and
closes the three control loops that matter for weeping in practice:

1. Reboiler steam flow control (FIC) acting through a steam valve that can stick.
2. Sensitive tray temperature control (TIC) trimming boilup. It is not pressure
   compensated, which is exactly why a pressure rise quietly cuts boilup.
3. Overhead composition control (APC) trimming reflux from the online analyzer, which is
   why a drifting analyzer burns steam with no real quality problem.

Tray hydraulics (weep margin, weep fraction, flooding, tray pressure drop) are evaluated
for the rectifying and stripping sections at every step and feed back into tray
efficiency and therefore product composition. A second pass of the same model with the
weeping to efficiency link switched off gives the exact counterfactual used as ground
truth for the cost of weeping.
"""
from __future__ import annotations

import math

import numpy as np

from .hydraulics import TrayGeometry
from .scenarios import Exogenous


def simulate_column(col: dict, geom: TrayGeometry, ex: Exogenous, seed: int, suppress_weep: bool = False,
                    hidden: dict | None = None) -> dict[str, np.ndarray]:
    """Simulate one column.

    hidden holds the as built deviations from the datasheet that a real column always has:
    the true hole area, the true weep coefficient and the reflux meter bias. The soft
    sensor never sees them, which is what makes physics alone imperfect.
    """
    hidden = hidden or {}
    n = len(ex.feed_mult)
    rng = np.random.default_rng(seed)
    # pre drawn noise so the counterfactual pass sees identical disturbances
    nz = {k: rng.normal(0, 1, n) for k in (
        "flow", "flow2", "flow3", "temp1", "temp2", "temp3", "dp1", "dp2", "press", "ai1", "ai2", "lvl", "steam", "chest", "foam")}

    e0 = col["murphree_eff"]
    n_r, n_s = geom.n_rect, geom.n_strip
    k_r = math.log(col["sep_sensitivity_top"]) / (0.20 * n_r)
    k_s = math.log(col["sep_sensitivity_bottom"]) / (0.20 * n_s)
    ln_xt_d = math.log(col["top_impurity_design"])
    ln_xb_d = math.log(col["bottom_impurity_design"])
    span = col["bottom_temp_design_c"] - col["top_temp_design_c"]
    a_p = 0.25 * span
    c_t = 0.12 * span
    r_d = col["reflux_ratio_design"]
    lam_p = col["latent_heat_kj_kg"]
    lam_s = col["steam_latent_kj_kg"]
    cp = col["feed_cp_kj_kg_k"]
    tb_feed = col["feed_bubble_temp_c"]
    tf_d = col["feed_temp_design_c"]
    p_abs_d = geom.pressure_abs_design
    rho_vd = geom.rho_vapour_design
    rho_l = geom.rho_liquid
    lw = geom.weir_length_m
    hw = geom.weir_height_mm
    dh = geom.hole_diameter_mm
    k2_off = geom.k2_offset + hidden.get("k2_offset_delta", 0.0)
    ah0 = geom.hole_area_m2 * hidden.get("hole_area_factor", 1.0)
    reflux_bias = hidden.get("reflux_meter_bias", 0.0)
    an = geom.net_area_m2
    csb = geom.csb
    hr = 12.5e3 / rho_l
    root_l = rho_l
    f_d = geom.feed_kg_h
    dfrac = col["distillate_fraction"]
    q_d = geom.q_design
    vs_over_b_d = geom.v_strip_design / geom.bottoms_kg_h
    s_d = geom.steam_design_t_h
    chest_d = col["chest_pressure_design_barg"]
    supply = 1.45 * chest_d
    chest_max = 1.17 * chest_d
    cv = s_d / (0.62 * math.sqrt(supply - chest_d))
    ki = 31.0 / s_d
    kp = 18.6 / s_d
    alpha = 1.0 - math.exp(-5.0 / 30.0)

    out = {k: np.zeros(n) for k in (
        "feed_flow", "feed_temp", "reflux_flow", "distillate_flow", "bottoms_flow", "steam_flow", "steam_sp",
        "steam_valve_op", "top_pressure", "pressure_sp", "dp_rect", "dp_strip", "top_temp", "sens_temp",
        "sens_temp_sp", "bottom_temp", "sump_level", "top_analyzer", "bottom_analyzer", "chest_pressure",
        # ground truth channels (never shown to the analytics layer)
        "true_wf_rect", "true_wf_strip", "true_eff_rect", "true_eff_strip", "true_margin_rect", "true_margin_strip",
        "true_flood_rect", "true_flood_strip", "true_xt", "true_xb", "true_offspec_t", "true_vs", "true_vr", "true_steam")}

    u_t = 0.0
    u_b = 0.0
    integ = 62.0
    s_prev = s_d * 0.93
    op_stuck = 62.0
    lnxt = ln_xt_d
    lnxb = ln_xb_d
    lvl = 50.0
    ai_top_hold = math.exp(lnxt)
    ai_bot_hold = math.exp(lnxb)
    t_sp = col["sens_temp_design_c"]
    dt_h = 5.0 / 60.0
    comp_alpha = 1.0 - math.exp(-5.0 / 60.0)
    xt_comp = math.exp(lnxt)
    xb_comp = math.exp(lnxb)

    def valve_flow(opening: float, guess: float, kc: float) -> float:
        """Solve flow = Cv x sqrt(supply minus chest(flow)) by Newton iteration."""
        a = cv * min(max(opening, 0.0), 100.0) / 100.0
        if a <= 0.0:
            return 0.0
        x = max(guess, 0.05 * s_d)
        for _ in range(5):
            head = max(supply - kc * x ** 0.6, 1e-4)
            g = x - a * math.sqrt(head)
            dg = 1.0 + a * 0.5 / math.sqrt(head) * kc * 0.6 * x ** -0.4
            x = max(x - g / dg, 0.02 * s_d)
        return x

    for i in range(n):
        F = f_d * ex.feed_mult[i]
        tf = tf_d + ex.feed_temp_offset[i]
        q = 1.0 + cp * max(tb_feed - tf, 0.0) / lam_p
        pr = ex.pressure_ratio[i] * (1.0 + ex.pressure_spike[i])
        p_abs = p_abs_d * pr
        rho_v = rho_vd * pr
        D = F * dfrac
        B = F - D

        # setpoint generation: reflux target from APC, boilup from ratio plus TIC trim
        r_target = r_d * math.exp(u_t)
        vr_req = (r_target + 1.0) * D
        vs_req = (vr_req + (q_d - 1.0) * F) * math.exp(u_b)
        steam_sp = vs_req * lam_p / lam_s / 1000.0

        # steam valve and flow controller
        ua = ex.ua_ratio[i]
        s_max = s_d * (chest_max / chest_d * ua) ** (1.0 / 0.6)
        kc = chest_d / (s_d ** 0.6 * ua)

        err = steam_sp - s_prev
        integ = min(max(integ + ki * err, 0.0), 100.0)
        op = min(max(integ + kp * err, 0.0), 100.0)
        deadband = ex.stiction_deadband[i]
        if deadband > 0.0:
            if abs(op - op_stuck) > deadband:
                slip = ex.stiction_slip[i] * (1.0 if op > op_stuck else -1.0)
                s = valve_flow(op + slip, s_prev, kc)
                op_stuck = op
            else:
                s = s_prev
        else:
            s = valve_flow(op, s_prev, kc)
            op_stuck = op
        if s > s_max:
            s = s_max
            integ = min(integ, op)  # anti windup while duty limited
        s = max(s, 0.02 * s_d)
        s_prev = s
        chest = min(chest_d * (s / s_d) ** 0.6 / ua, chest_max)

        # internal flows
        vs = s * 1000.0 * lam_s / (lam_p * ex.latent_factor[i])
        vr = max(vs - (q - 1.0) * F, 0.10 * geom.v_rect_design)
        lr = max(vr - D, 0.02 * D)
        ls = lr + q * F
        r_act = lr / D

        # hydraulics, rectifying then stripping
        res = []
        for v, l, ah in ((vr, lr, ah0 * ex.ah_mult_rect[i]), (vs, ls, ah0 * ex.ah_mult_strip[i])):
            qv = v / 3600.0 / rho_v
            uh = qv / ah
            how = 750.0 * (l / 3600.0 / (root_l * lw)) ** (2.0 / 3.0)
            hc = hw + how
            hcc = min(max(hc, 0.0), 120.0)
            k2 = 28.0 + 0.10 * hcc - 0.0004 * hcc * hcc + k2_off
            uh_min = (k2 - 0.90 * (25.4 - dh)) / math.sqrt(rho_v)
            margin = uh / uh_min
            deficit = 1.0 - margin
            wf = min(1.55 * deficit ** 1.25, 0.60) if deficit > 0 else 0.0
            u_flood = csb * math.sqrt((rho_l - rho_v) / rho_v)
            flood = qv / an / u_flood
            hd = 51.0 * (uh / 0.80) ** 2 * rho_v / rho_l
            ht = hd + (1.0 - 0.45 * wf) * hc + hr
            if flood > 0.85:
                ht *= 1.0 + 5.0 * (flood - 0.85) ** 1.2
            res.append((margin, wf, flood, ht))
        (m_r, wf_r, fl_r, ht_r), (m_s, wf_s, fl_s, ht_s) = res

        ent_r = min(max(2.2 * (fl_r - 0.86), 0.0), 0.45)
        ent_s = min(max(2.2 * (fl_s - 0.86), 0.0), 0.45)
        wk_r = 0.0 if suppress_weep else wf_r
        wk_s = 0.0 if suppress_weep else wf_s
        e_r = e0 * (1 - 0.75 * wk_r) * (1 - ent_r) * (1 - ex.foam_eff[i]) * (1 - ex.eff_pen_rect[i])
        e_s = e0 * (1 - 0.75 * wk_s) * (1 - ent_s) * (1 - ex.foam_eff[i]) * (1 - ex.eff_pen_strip[i])

        # compositions with a thirty minute first order response
        # relative volatility falls as pressure rises, which makes both products harder to purify
        alpha_pen = 1.2 * math.log(pr)
        lnxt_ss = ln_xt_d - k_r * n_r * (e_r - e0) - 1.4 * math.log(max(r_act, 1e-3) / r_d) + alpha_pen
        lnxb_ss = ln_xb_d - k_s * n_s * (e_s - e0) - 1.4 * math.log(max(vs / B, 1e-3) / vs_over_b_d) + alpha_pen
        lnxt += (min(lnxt_ss, math.log(30.0)) - lnxt) * alpha
        lnxb += (min(lnxb_ss, math.log(30.0)) - lnxb) * alpha
        xt = math.exp(lnxt)
        xb = math.exp(lnxb)

        lnpr = math.log(pr)
        t_top = col["top_temp_design_c"] + a_p * lnpr + 0.04 * span * (lnxt - ln_xt_d)
        t_sens = col["sens_temp_design_c"] + a_p * lnpr - c_t * (lnxb - ln_xb_d)
        t_bot = col["bottom_temp_design_c"] + a_p * lnpr - 0.5 * c_t * (lnxb - ln_xb_d)

        # analyzers update every ten minutes and the top one may carry a drifting bias
        if i % 2 == 0:
            ai_top_hold = max(xt * (1 + 0.03 * nz["ai1"][i]) + ex.analyzer_bias_top[i], 0.005)
            ai_bot_hold = max(xb * (1 + 0.03 * nz["ai2"][i]), 0.005)

        t_sens_meas = t_sens + 0.15 * nz["temp2"][i]
        # TIC trims boilup, APC trims reflux; both have bounded authority
        if max(fl_r, fl_s) > 0.88:
            # panel operators and the APC flood constraint back boilup off high dP
            u_b = max(u_b - 0.02, -0.30)
            u_t = max(u_t - 0.01, -0.15)
        else:
            u_b = min(max(u_b + 0.03 * (t_sp - t_sens_meas) / c_t, -0.30), 0.25)
            u_t = min(max(u_t + 0.010 * (math.log(ai_top_hold) - ln_xt_d), -0.15), 0.25)

        foam = ex.foam_dp[i]
        foam_noise = 1.0 + (foam - 1.0) * 0.35 * nz["foam"][i]
        lvl_sd = 1.1 + 9.0 * wf_s + 14.0 * (foam - 1.0)
        lvl = 0.75 * lvl + 0.25 * (50.0 + lvl_sd * nz["lvl"][i] * 2.0)

        # rundown quality is judged on a one hour composite, as the product tanks see it
        xt_comp += (xt - xt_comp) * comp_alpha
        xb_comp += (xb - xb_comp) * comp_alpha
        offspec = 0.0
        if xt_comp > col["top_impurity_spec"]:
            offspec += D / 1000.0 * dt_h
        if xb_comp > col["bottom_impurity_spec"]:
            offspec += B / 1000.0 * dt_h

        out["feed_flow"][i] = F / 1000.0 * (1 + 0.008 * nz["flow"][i])
        out["feed_temp"][i] = tf + 0.3 * nz["temp1"][i]
        out["reflux_flow"][i] = lr / 1000.0 * (1 + reflux_bias + 0.008 * nz["flow2"][i])
        out["distillate_flow"][i] = D / 1000.0 * (1 + 0.010 * nz["flow3"][i])
        out["bottoms_flow"][i] = B / 1000.0 * (1 + 0.010 * nz["flow"][(i + 7) % n])
        out["steam_flow"][i] = s * (1 + ex.steam_meter_bias[i] + 0.012 * nz["steam"][i])
        out["steam_sp"][i] = steam_sp
        out["steam_valve_op"][i] = op
        out["top_pressure"][i] = p_abs - 1.013 + 0.02 * (p_abs_d - 1.013) * 0.15 * nz["press"][i]
        out["pressure_sp"][i] = p_abs_d * ex.pressure_sp_ratio[i] - 1.013
        out["dp_rect"][i] = ht_r / 1000.0 * rho_l * 9.81 * n_r / 100.0 * foam * foam_noise * (1 + ex.dp_drift_rect[i] + 0.025 * nz["dp1"][i])
        out["dp_strip"][i] = ht_s / 1000.0 * rho_l * 9.81 * n_s / 100.0 * foam * foam_noise * (1 + ex.dp_drift_strip[i] + 0.025 * nz["dp2"][i])
        out["top_temp"][i] = t_top + 0.15 * nz["temp1"][(i + 3) % n]
        out["sens_temp"][i] = t_sens_meas
        out["sens_temp_sp"][i] = t_sp
        out["bottom_temp"][i] = t_bot + 0.15 * nz["temp3"][i]
        out["sump_level"][i] = lvl
        out["top_analyzer"][i] = ai_top_hold
        out["bottom_analyzer"][i] = ai_bot_hold
        out["chest_pressure"][i] = chest * (1 + 0.006 * nz["chest"][i])

        out["true_wf_rect"][i] = wf_r
        out["true_wf_strip"][i] = wf_s
        out["true_eff_rect"][i] = e_r
        out["true_eff_strip"][i] = e_s
        out["true_margin_rect"][i] = m_r
        out["true_margin_strip"][i] = m_s
        out["true_flood_rect"][i] = fl_r
        out["true_flood_strip"][i] = fl_s
        out["true_xt"][i] = xt
        out["true_xb"][i] = xb
        out["true_offspec_t"][i] = offspec
        out["true_vs"][i] = vs
        out["true_vr"][i] = vr
        out["true_steam"][i] = s
    return out
