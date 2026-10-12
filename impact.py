"""Quantify what an episode cost.

Energy: measured reboiler steam minus the steam the normal behaviour model expects at the
same feed rate, pressure and feed temperature, integrated over the episode plus a six hour
recovery tail.

Quality: product made off specification, judged on a one hour composite of the online
analyzer as the rundown tanks would see it. When the analyzer disagrees with the lab by
more than a quarter in log terms it is treated as untrusted and the lab results are
interpolated instead, so a drifting analyzer cannot create phantom quality losses.

Efficiency: apparent tray efficiency from the inverse shortcut separation model, expressed
as a percentage loss relative to the design Murphree efficiency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _composite(x: pd.Series, minutes: float = 60.0, step: float = 5.0) -> np.ndarray:
    alpha = 1.0 - np.exp(-step / minutes)
    return x.ewm(alpha=alpha, adjust=False).mean().to_numpy()


def lab_series(lab: pd.DataFrame, column: str, point: str, index: pd.DatetimeIndex) -> pd.Series:
    l = lab[(lab["column"] == column) & (lab["sample_point"] == f"{column} {point}") & (lab["unit"] == "mol%")]
    if l.empty:
        return pd.Series(np.nan, index=index)
    s = l.set_index("timestamp")["value"].sort_index()
    s = s[~s.index.duplicated()]
    return s.reindex(s.index.union(index)).interpolate("time").reindex(index).ffill().bfill()


def episode_impact(ep, hist: pd.DataFrame, feat: pd.DataFrame, lab: pd.DataFrame, col: dict, econ: dict, step_min: int) -> dict:
    per_h = 60 // step_min
    a = ep.start_idx
    b = min(ep.end_idx + 6 * per_h, len(hist))
    dt_h = step_min / 60.0
    h = hist.iloc[a:b]
    f = feat.iloc[a:b]
    steam_excess = float(((h["steam_flow"] - f["steam_expected"]) * dt_h).sum())
    trust = float(np.median(np.abs(f["ai_lab_logratio"]))) < 0.25
    if trust:
        xt = pd.Series(_composite(hist["top_analyzer"]), index=hist.index).iloc[a:b]
        xb = pd.Series(_composite(hist["bottom_analyzer"]), index=hist.index).iloc[a:b]
        source = "online analyzer composite"
    else:
        xt = lab_series(lab, ep.column, "overhead", hist.index).iloc[a:b]
        xb = lab_series(lab, ep.column, "bottoms", hist.index).iloc[a:b]
        source = "laboratory interpolation (analyzer untrusted)"
    off_top = (xt.to_numpy() > col["top_impurity_spec"]) * h["distillate_flow"].to_numpy() * dt_h
    off_bot = (xb.to_numpy() > col["bottom_impurity_spec"]) * h["bottoms_flow"].to_numpy() * dt_h
    offspec = float(np.nansum(off_top) + np.nansum(off_bot))
    steam_cost = steam_excess * econ["lp_steam_price_per_t"]
    co2 = steam_excess * econ["steam_co2_t_per_t"]
    carbon = co2 * econ["carbon_price_per_t"]
    quality = offspec * col["downgrade_price_per_t"]
    core = feat.iloc[ep.start_idx:ep.end_idx]
    eff_loss = float(np.clip(1.0 - core["eff_min_est"].median(), 0.0, 1.0) * 100.0)
    section = "strip" if core["eff_strip_est"].median() <= core["eff_rect_est"].median() else "rect"
    return {
        "episode_id": ep.episode_id,
        "column": ep.column,
        "window_start": hist.index[a].isoformat(),
        "window_end": hist.index[b - 1].isoformat(),
        "excess_steam_t": round(steam_excess, 2),
        "offspec_t": round(offspec, 2),
        "offspec_top_t": round(float(np.nansum(off_top)), 2),
        "offspec_bottom_t": round(float(np.nansum(off_bot)), 2),
        "quality_source": source,
        "analyzer_trusted": bool(trust),
        "steam_cost_gbp": round(steam_cost, 0),
        "quality_cost_gbp": round(quality, 0),
        "co2_t": round(co2, 2),
        "carbon_cost_gbp": round(carbon, 0),
        "total_cost_gbp": round(steam_cost + quality + carbon, 0),
        "efficiency_loss_pct": round(eff_loss, 2),
        "worst_section": section,
        "min_weep_margin_est": round(float(core["wm_min"].quantile(0.05)), 3),
        "min_dp_resid": round(float(core["dp_min_resid"].quantile(0.05)), 3),
    }
