"""Feature engineering for weeping detection and discrimination.

All features are dimensionless ratios or residuals against a normal behaviour model fitted
on each column's own fault free commissioning window. That makes a feature mean the same
thing on a 1.2 metre deethaniser and a 3.4 metre C3 splitter, which is what lets one model
generalise across columns (tested with leave one column out evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..simulator.hydraulics import TrayGeometry
from .softsensor import hydraulic_state

FEATURES = [
    "feed_ratio", "pressure_ratio", "q_feed", "feed_temp_dev",
    "wm_min", "wm_rect", "wm_strip", "flood_max",
    "dp_rect_resid", "dp_strip_resid", "dp_min_resid", "dp_max_resid",
    "dp_rect_resid_3h", "dp_strip_resid_3h", "dp_resid_std_1h",
    "steam_resid", "steam_cv_1h", "steam_osc_3h", "steam_sp_err_1h", "stick_ratio_3h",
    "op_mean_3h", "op_sat_frac_3h", "chest_ua_index",
    "eff_rect_est", "eff_strip_est", "eff_min_est", "eff_min_est_3h",
    "top_x_spec", "bottom_x_spec", "ai_lab_logratio", "sens_temp_dev", "spread_resid",
    "level_std_1h", "reflux_ratio_rel", "pressure_sp_dev",
]


@dataclass
class NormalModel:
    """Ridge regressions fitted on the reference window of one column."""

    steam: Ridge
    spread: Ridge
    dp_rect_calib: float
    dp_strip_calib: float
    chest_coef: float

    @staticmethod
    def design_matrix(feed_ratio, pr, tf_dev, q) -> np.ndarray:
        return np.column_stack([feed_ratio, feed_ratio ** 2, np.log(pr), tf_dev, q])


def fit_normal_model(df: pd.DataFrame, hyd: pd.DataFrame, col: dict, g: TrayGeometry, ref: slice) -> NormalModel:
    tf_dev = df["feed_temp"].to_numpy() - col["feed_temp_design_c"]
    X = NormalModel.design_matrix(hyd["feed_ratio"].to_numpy(), hyd["pressure_ratio"].to_numpy(), tf_dev, hyd["q_feed"].to_numpy())
    # steam is modelled on feed alone: boilup follows feed through the ratio controller, and the
    # reference window does not excite pressure or feed temperature enough to extrapolate safely
    steam = Ridge(alpha=1e-3).fit(hyd["feed_ratio"].to_numpy()[ref].reshape(-1, 1), df["steam_flow"].to_numpy()[ref])
    spread_y = (df["bottom_temp"] - df["top_temp"]).to_numpy()
    spread = Ridge(alpha=1e-3).fit(X[ref], spread_y[ref])
    dpr = float(np.median(df["dp_rect"].to_numpy()[ref] / hyd["dp_exp_rect"].to_numpy()[ref]))
    dps = float(np.median(df["dp_strip"].to_numpy()[ref] / hyd["dp_exp_strip"].to_numpy()[ref]))
    s = np.maximum(df["steam_flow"].to_numpy()[ref], 1e-3)
    chest_coef = float(np.median(df["chest_pressure"].to_numpy()[ref] / (s / g.steam_design_t_h) ** 0.6))
    return NormalModel(steam=steam, spread=spread, dp_rect_calib=dpr, dp_strip_calib=dps, chest_coef=chest_coef)


def _roll(x: pd.Series, w: int, fn: str) -> pd.Series:
    r = x.rolling(w, min_periods=max(2, w // 3))
    return getattr(r, fn)()


def _oscillation(x: pd.Series, w: int) -> pd.Series:
    """Zero crossings per hour of the detrended signal, a robust cycling indicator."""
    detr = x - x.rolling(w, min_periods=3, center=False).mean()
    sign = np.sign(detr.fillna(0).to_numpy())
    cross = np.abs(np.diff(sign, prepend=sign[0])) > 0
    return pd.Series(cross.astype(float), index=x.index).rolling(w, min_periods=3).sum() * (12.0 / w)


def lab_alignment(df: pd.DataFrame, lab: pd.DataFrame, column: str) -> pd.Series:
    """Log ratio of the overhead analyzer to the most recent overhead lab result within 12 hours."""
    l = lab[(lab["column"] == column) & (lab["sample_point"] == f"{column} overhead") & (lab["unit"] == "mol%")]
    l = l[["timestamp", "value"]].sort_values("timestamp").rename(columns={"value": "lab_top"})
    if l.empty:
        return pd.Series(0.0, index=df.index)
    left = pd.DataFrame({"timestamp": df.index, "ai": df["top_analyzer"].to_numpy()})
    # compare the analyzer value at the time the lab sample was drawn, then hold until the next sample
    ai_at = pd.merge_asof(l, left, on="timestamp", direction="nearest")
    ai_at["ratio"] = np.log(np.maximum(ai_at["ai"], 1e-3) / np.maximum(ai_at["lab_top"], 1e-3))
    held = pd.merge_asof(left[["timestamp"]], ai_at[["timestamp", "ratio"]], on="timestamp", direction="backward", tolerance=pd.Timedelta(hours=12))
    return pd.Series(held["ratio"].fillna(0.0).to_numpy(), index=df.index).rolling(36, min_periods=1).median()


def build_features(df: pd.DataFrame, lab: pd.DataFrame, column: str, col: dict, g: TrayGeometry,
                   ref_days: float, w_short: int = 12, w_long: int = 36) -> tuple[pd.DataFrame, NormalModel]:
    hyd = hydraulic_state(df, col, g)
    step = (df.index[1] - df.index[0]).total_seconds() / 60.0
    ref = slice(0, int(ref_days * 1440 / step))
    nbm = fit_normal_model(df, hyd, col, g, ref)
    tf_dev = df["feed_temp"].to_numpy() - col["feed_temp_design_c"]
    X = NormalModel.design_matrix(hyd["feed_ratio"].to_numpy(), hyd["pressure_ratio"].to_numpy(), tf_dev, hyd["q_feed"].to_numpy())
    steam_exp = np.maximum(nbm.steam.predict(hyd["feed_ratio"].to_numpy().reshape(-1, 1)), 0.05 * g.steam_design_t_h)
    spread_exp = nbm.spread.predict(X)
    span = col["bottom_temp_design_c"] - col["top_temp_design_c"]

    feat = pd.DataFrame(index=df.index)
    feat["feed_ratio"] = hyd["feed_ratio"]
    feat["pressure_ratio"] = hyd["pressure_ratio"]
    feat["q_feed"] = hyd["q_feed"]
    feat["feed_temp_dev"] = tf_dev / max(span, 1.0)
    feat["wm_rect"] = hyd["wm_rect"]
    feat["wm_strip"] = hyd["wm_strip"]
    feat["wm_min"] = np.minimum(hyd["wm_rect"], hyd["wm_strip"])
    feat["flood_max"] = np.maximum(hyd["flood_rect"], hyd["flood_strip"])
    dpr = df["dp_rect"] / (hyd["dp_exp_rect"] * nbm.dp_rect_calib) - 1.0
    dps = df["dp_strip"] / (hyd["dp_exp_strip"] * nbm.dp_strip_calib) - 1.0
    feat["dp_rect_resid"] = _roll(dpr, 6, "mean")
    feat["dp_strip_resid"] = _roll(dps, 6, "mean")
    feat["dp_min_resid"] = np.minimum(feat["dp_rect_resid"], feat["dp_strip_resid"])
    feat["dp_max_resid"] = np.maximum(feat["dp_rect_resid"], feat["dp_strip_resid"])
    feat["dp_rect_resid_3h"] = _roll(dpr, w_long, "mean")
    feat["dp_strip_resid_3h"] = _roll(dps, w_long, "mean")
    feat["dp_resid_std_1h"] = np.maximum(_roll(dpr, w_short, "std"), _roll(dps, w_short, "std"))
    steam = df["steam_flow"]
    feat["steam_resid"] = _roll(steam / steam_exp - 1.0, w_short, "mean")
    feat["steam_cv_1h"] = _roll(steam, w_short, "std") / _roll(steam, w_short, "mean")
    feat["steam_osc_3h"] = _oscillation(steam, w_long)
    feat["steam_sp_err_1h"] = _roll((steam - df["steam_sp"]).abs() / df["steam_sp"].clip(lower=0.1), w_short, "mean")
    d_op = df["steam_valve_op"].diff().abs()
    d_fl = steam.diff().abs() / steam.rolling(w_short, min_periods=2).mean()
    stuck = ((d_op > 0.8) & (d_fl < 0.004)).astype(float)
    feat["stick_ratio_3h"] = _roll(stuck, w_long, "mean")
    op = df["steam_valve_op"]
    feat["op_mean_3h"] = _roll(op, w_long, "mean") / 100.0
    feat["op_sat_frac_3h"] = _roll((op > 95).astype(float), w_long, "mean")
    feat["chest_ua_index"] = _roll(nbm.chest_coef * (steam.clip(lower=1e-3) / g.steam_design_t_h) ** 0.6 / df["chest_pressure"], w_short, "mean")
    feat["eff_rect_est"] = _roll(hyd["eff_rect_est"] / col["murphree_eff"], 6, "mean")
    feat["eff_strip_est"] = _roll(hyd["eff_strip_est"] / col["murphree_eff"], 6, "mean")
    feat["eff_min_est"] = np.minimum(feat["eff_rect_est"], feat["eff_strip_est"])
    feat["eff_min_est_3h"] = _roll(feat["eff_min_est"], w_long, "mean")
    feat["top_x_spec"] = _roll(df["top_analyzer"] / col["top_impurity_spec"], 6, "mean")
    feat["bottom_x_spec"] = _roll(df["bottom_analyzer"] / col["bottom_impurity_spec"], 6, "mean")
    feat["ai_lab_logratio"] = lab_alignment(df, lab, column)
    feat["sens_temp_dev"] = _roll((df["sens_temp"] - df["sens_temp_sp"]) / max(span, 1.0), 6, "mean")
    feat["spread_resid"] = _roll(((df["bottom_temp"] - df["top_temp"]) - spread_exp) / max(span, 1.0), 6, "mean")
    feat["level_std_1h"] = _roll(df["sump_level"], w_short, "std")
    feat["reflux_ratio_rel"] = hyd["reflux_ratio"] / col["reflux_ratio_design"]
    feat["pressure_sp_dev"] = (df["pressure_sp"] + 1.013) / g.pressure_abs_design - 1.0
    feat = feat[FEATURES].replace([np.inf, -np.inf], np.nan).bfill().fillna(0.0)
    feat["steam_expected"] = steam_exp
    return feat, nbm
