"""Tray hydraulics correlations used by both the plant simulator and the soft sensor.

The forms follow the classical sieve tray design method found in Coulson and Richardson
Volume 6 (weep point, weir crest, dry plate and residual head) together with a Souders
Brown style flooding capacity. The K2 weep coefficient is a smooth polynomial fitted to
the published chart rather than a digitised curve, so the numbers are representative of
industrial behaviour but this module is not a substitute for a vendor rating.

All functions accept numpy arrays so the soft sensor can evaluate a whole year of
historian data in one vectorised call.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

GRAVITY = 9.81


def k2_coefficient(h_liquid_mm):
    """Weep point coefficient K2 as a function of clear liquid height (weir plus crest)."""
    h = np.clip(np.asarray(h_liquid_mm, dtype=float), 0.0, 120.0)
    return 28.0 + 0.10 * h - 0.0004 * h ** 2


def weir_crest_mm(liquid_kg_h, rho_liquid, weir_length_m):
    """Francis weir crest height in millimetres of clear liquid."""
    lw = np.maximum(np.asarray(liquid_kg_h, dtype=float), 0.0) / 3600.0
    return 750.0 * (lw / (rho_liquid * weir_length_m)) ** (2.0 / 3.0)


def min_hole_velocity(h_liquid_mm, hole_diameter_mm, rho_vapour, k2_offset=0.0):
    """Minimum vapour velocity through the holes before liquid starts to weep (m/s)."""
    k2 = k2_coefficient(h_liquid_mm) + k2_offset
    return (k2 - 0.90 * (25.4 - hole_diameter_mm)) / np.sqrt(np.asarray(rho_vapour, dtype=float))


def dry_plate_drop_mm(hole_velocity, rho_vapour, rho_liquid, orifice_coeff=0.80):
    return 51.0 * (np.asarray(hole_velocity) / orifice_coeff) ** 2 * np.asarray(rho_vapour) / rho_liquid


def residual_head_mm(rho_liquid):
    return 12.5e3 / rho_liquid


def weep_fraction(weep_margin):
    """Fraction of tray liquid that bypasses the active area through the holes.

    Zero above the weep point, rising steeply below it and saturating at sixty percent,
    which is roughly where a sieve tray dumps and stops behaving as a stage at all.
    """
    m = np.asarray(weep_margin, dtype=float)
    deficit = np.clip(1.0 - m, 0.0, None)
    return np.clip(1.55 * deficit ** 1.25, 0.0, 0.60)


def flood_fraction(vapour_kg_h, rho_vapour, rho_liquid, net_area_m2, csb):
    qv = np.asarray(vapour_kg_h, dtype=float) / 3600.0 / np.asarray(rho_vapour, dtype=float)
    u_net = qv / net_area_m2
    u_flood = csb * np.sqrt((rho_liquid - np.asarray(rho_vapour)) / np.asarray(rho_vapour))
    return u_net / u_flood


def entrainment_penalty(flood):
    return np.clip(2.2 * (np.asarray(flood, dtype=float) - 0.86), 0.0, 0.45)


@dataclass
class TrayGeometry:
    """Derived geometry and design loads for one column."""

    tag: str
    diameter_m: float
    total_area_m2: float
    net_area_m2: float
    active_area_m2: float
    hole_area_m2: float
    hole_area_fraction: float
    weir_length_m: float
    weir_height_mm: float
    hole_diameter_mm: float
    k2_offset: float
    csb: float
    n_rect: int
    n_strip: int
    rho_liquid: float
    rho_vapour_design: float
    pressure_abs_design: float
    feed_kg_h: float
    distillate_kg_h: float
    bottoms_kg_h: float
    q_design: float
    v_rect_design: float
    v_strip_design: float
    l_rect_design: float
    l_strip_design: float
    weep_margin_rect_design: float
    weep_margin_strip_design: float
    flood_rect_design: float
    flood_strip_design: float
    steam_design_t_h: float
    steam_capacity_t_h: float

    def to_dict(self) -> dict:
        return asdict(self)


def feed_quality(feed_temp_c, bubble_temp_c, cp, latent):
    """Thermal condition q of the feed. Values above one mean subcooled liquid."""
    sub = np.clip(np.asarray(bubble_temp_c) - np.asarray(feed_temp_c), 0.0, None)
    return 1.0 + cp * sub / latent


def section_state(vapour_kg_h, liquid_kg_h, rho_vapour, geom: TrayGeometry, hole_area_m2=None):
    """Return hole velocity, minimum hole velocity, weep margin and tray head (mm)."""
    ah = geom.hole_area_m2 if hole_area_m2 is None else hole_area_m2
    qv = np.asarray(vapour_kg_h, dtype=float) / 3600.0 / rho_vapour
    uh = qv / ah
    how = weir_crest_mm(liquid_kg_h, geom.rho_liquid, geom.weir_length_m)
    h_clear = geom.weir_height_mm + how
    uh_min = min_hole_velocity(h_clear, geom.hole_diameter_mm, rho_vapour, geom.k2_offset)
    margin = uh / uh_min
    hd = dry_plate_drop_mm(uh, rho_vapour, geom.rho_liquid)
    return uh, uh_min, margin, hd, h_clear


def tray_head_mm(hd, h_clear, weep_frac, rho_liquid):
    """Total wet tray pressure drop in mm of liquid; weeping liquid stops contributing head."""
    return hd + (1.0 - 0.45 * np.asarray(weep_frac)) * h_clear + residual_head_mm(rho_liquid)


def head_to_mbar(head_mm, n_trays, rho_liquid):
    return np.asarray(head_mm) / 1000.0 * rho_liquid * GRAVITY * n_trays / 100.0


def derive_geometry(tag: str, c: dict) -> TrayGeometry:
    """Derive hole area and capacity factor from the design basis in plant.yaml."""
    d = c["diameter_m"]
    at = math.pi * d * d / 4.0
    an = 0.88 * at
    aa = 0.76 * at
    lw = 0.73 * d
    p_abs = c["pressure_design_barg"] + 1.013
    feed = c["feed_design_t_h"] * 1000.0
    dist = feed * c["distillate_fraction"]
    bot = feed - dist
    q = float(feed_quality(c["feed_temp_design_c"], c["feed_bubble_temp_c"], c["feed_cp_kj_kg_k"], c["latent_heat_kj_kg"]))
    r = c["reflux_ratio_design"]
    l_r = r * dist
    v_r = (r + 1.0) * dist
    v_s = v_r + (q - 1.0) * feed
    l_s = l_r + q * feed
    rho_v = c["rho_vapour_design"]
    rho_l = c["rho_liquid"]
    k2_off = c.get("weep_k2_offset", 0.0)
    dh = c["hole_diameter_mm"]
    hw = c["weir_height_mm"]

    def needed_area(v, l):
        how = float(weir_crest_mm(l, rho_l, lw))
        uh_min = float(min_hole_velocity(hw + how, dh, rho_v, k2_off))
        qv = v / 3600.0 / rho_v
        return qv / (c["design_weep_margin"] * uh_min)

    ah = min(needed_area(v_r, l_r), needed_area(v_s, l_s))
    root = math.sqrt((rho_l - rho_v) / rho_v)
    u_net_max = max(v_r, v_s) / 3600.0 / rho_v / an
    csb = u_net_max / (root * c["design_flood"])

    n_rect = max(int(c["feed_tray"]) - 1, 1)
    n_strip = int(c["n_trays"]) - n_rect
    geom = TrayGeometry(
        tag=tag, diameter_m=d, total_area_m2=at, net_area_m2=an, active_area_m2=aa,
        hole_area_m2=ah, hole_area_fraction=ah / aa, weir_length_m=lw, weir_height_mm=hw,
        hole_diameter_mm=dh, k2_offset=k2_off, csb=csb, n_rect=n_rect, n_strip=n_strip,
        rho_liquid=rho_l, rho_vapour_design=rho_v, pressure_abs_design=p_abs,
        feed_kg_h=feed, distillate_kg_h=dist, bottoms_kg_h=bot, q_design=q,
        v_rect_design=v_r, v_strip_design=v_s, l_rect_design=l_r, l_strip_design=l_s,
        weep_margin_rect_design=0.0, weep_margin_strip_design=0.0,
        flood_rect_design=0.0, flood_strip_design=0.0,
        steam_design_t_h=v_s * c["latent_heat_kj_kg"] / c["steam_latent_kj_kg"] / 1000.0,
        steam_capacity_t_h=0.0,
    )
    geom.steam_capacity_t_h = geom.steam_design_t_h * c["reboiler_capacity_factor"]
    geom.weep_margin_rect_design = float(section_state(v_r, l_r, rho_v, geom)[2])
    geom.weep_margin_strip_design = float(section_state(v_s, l_s, rho_v, geom)[2])
    geom.flood_rect_design = float(flood_fraction(v_r, rho_v, rho_l, an, csb))
    geom.flood_strip_design = float(flood_fraction(v_s, rho_v, rho_l, an, csb))
    return geom
