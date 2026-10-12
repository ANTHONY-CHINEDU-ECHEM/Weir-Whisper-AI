"""Hydraulic and separation soft sensors.

The soft sensor turns raw historian tags into the physical quantities an engineer reasons
with: section vapour and liquid traffic, weep margin, flood fraction, expected tray
pressure drop, and an inverse estimate of tray efficiency. It uses only the design basis
(datasheet values) and measured tags, never the simulator state. That means it inherits
the blind spots a real soft sensor has: it assumes the design hole area, so damaged trays
look healthy to it, and it assumes steam flow equals boilup, so a fouled reboiler is only
visible through the valve and chest pressure.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..simulator.hydraulics import (TrayGeometry, feed_quality, flood_fraction, head_to_mbar, section_state,
                                    tray_head_mm)


def hydraulic_state(df: pd.DataFrame, col: dict, g: TrayGeometry) -> pd.DataFrame:
    f = df["feed_flow"].to_numpy() * 1000.0
    d = np.maximum(df["distillate_flow"].to_numpy() * 1000.0, 1.0)
    b = np.maximum(df["bottoms_flow"].to_numpy() * 1000.0, 1.0)
    lr = np.maximum(df["reflux_flow"].to_numpy() * 1000.0, 1.0)
    steam = df["steam_flow"].to_numpy()
    p_abs = df["top_pressure"].to_numpy() + 1.013
    pr = p_abs / g.pressure_abs_design
    rho_v = g.rho_vapour_design * pr
    q = feed_quality(df["feed_temp"].to_numpy(), col["feed_bubble_temp_c"], col["feed_cp_kj_kg_k"], col["latent_heat_kj_kg"])
    vs = steam * 1000.0 * col["steam_latent_kj_kg"] / col["latent_heat_kj_kg"]
    vr_balance = lr + d                         # overhead material balance
    vr_reboil = np.maximum(vs - (q - 1.0) * f, 0.1 * g.v_rect_design)
    vr = 0.5 * (vr_balance + vr_reboil)
    ls = lr + q * f
    _, _, wm_r, hd_r, hc_r = section_state(vr, lr, rho_v, g)
    _, _, wm_s, hd_s, hc_s = section_state(vs, ls, rho_v, g)
    fl_r = flood_fraction(vr, rho_v, g.rho_liquid, g.net_area_m2, g.csb)
    fl_s = flood_fraction(vs, rho_v, g.rho_liquid, g.net_area_m2, g.csb)
    dp_exp_r = head_to_mbar(tray_head_mm(hd_r, hc_r, 0.0, g.rho_liquid), g.n_rect, g.rho_liquid)
    dp_exp_s = head_to_mbar(tray_head_mm(hd_s, hc_s, 0.0, g.rho_liquid), g.n_strip, g.rho_liquid)
    fac_r = vr / 3600.0 / rho_v / g.active_area_m2 * np.sqrt(rho_v)
    fac_s = vs / 3600.0 / rho_v / g.active_area_m2 * np.sqrt(rho_v)

    # inverse shortcut separation model gives an apparent tray efficiency per section
    e0 = col["murphree_eff"]
    k_r = math.log(col["sep_sensitivity_top"]) / (0.20 * g.n_rect)
    k_s = math.log(col["sep_sensitivity_bottom"]) / (0.20 * g.n_strip)
    xt = pd.Series(np.maximum(df["top_analyzer"].to_numpy(), 1e-3), index=df.index).rolling(12, min_periods=1).mean().to_numpy()
    xb = pd.Series(np.maximum(df["bottom_analyzer"].to_numpy(), 1e-3), index=df.index).rolling(12, min_periods=1).mean().to_numpy()
    r_act = lr / d
    vs_b_d = g.v_strip_design / g.bottoms_kg_h
    lnpr = np.log(pr)
    e_r = e0 - (np.log(xt / col["top_impurity_design"]) + 1.4 * np.log(r_act / col["reflux_ratio_design"]) - 1.2 * lnpr) / (k_r * g.n_rect)
    e_s = e0 - (np.log(xb / col["bottom_impurity_design"]) + 1.4 * np.log((vs / b) / vs_b_d) - 1.2 * lnpr) / (k_s * g.n_strip)

    return pd.DataFrame({
        "feed_ratio": f / g.feed_kg_h,
        "pressure_ratio": pr,
        "q_feed": q,
        "v_rect": vr, "v_strip": vs, "l_rect": lr, "l_strip": ls,
        "reflux_ratio": r_act,
        "wm_rect": wm_r, "wm_strip": wm_s,
        "flood_rect": fl_r, "flood_strip": fl_s,
        "dp_exp_rect": dp_exp_r, "dp_exp_strip": dp_exp_s,
        "ffactor_rect": fac_r, "ffactor_strip": fac_s,
        "eff_rect_est": np.clip(e_r, 0.0, 1.2), "eff_strip_est": np.clip(e_s, 0.0, 1.2),
    }, index=df.index)
