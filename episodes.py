"""Turn per sample class probabilities into discrete episodes.

Weeping episodes come from the weeping probability. Look alike episodes (flooding,
foaming, analyzer fault) come from their combined probability and are labelled with the
dominant class. Both use the same smoothing, hysteresis, minimum duration and gap
merging so short flickers do not become investigations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..simulator.scenarios import SAMPLE_CLASSES


@dataclass
class DetectedEpisode:
    episode_id: str
    column: str
    kind: str
    start: str
    end: str
    duration_h: float
    start_idx: int
    end_idx: int
    mean_prob: float
    peak_prob: float
    class_mix: dict
    signature: dict

    def to_dict(self) -> dict:
        return asdict(self)


def hysteresis_segments(p: np.ndarray, enter: float, exit_: float, smooth: int, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    ps = pd.Series(p).rolling(smooth, min_periods=1, center=True).mean().to_numpy()
    segs = []
    on = False
    start = 0
    for i, v in enumerate(ps):
        if not on and v >= enter:
            on, start = True, i
        elif on and v < exit_:
            segs.append((start, i))
            on = False
    if on:
        segs.append((start, len(ps)))
    merged: list[list[int]] = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_len]


SIGNATURE_FEATURES = [
    "feed_ratio", "pressure_ratio", "pressure_sp_dev", "q_feed", "feed_temp_dev", "wm_min", "wm_rect", "wm_strip",
    "flood_max", "dp_rect_resid", "dp_strip_resid", "dp_min_resid", "dp_max_resid", "dp_resid_std_1h", "steam_resid",
    "steam_cv_1h", "steam_osc_3h", "stick_ratio_3h", "op_mean_3h", "op_sat_frac_3h", "chest_ua_index",
    "eff_rect_est", "eff_strip_est", "eff_min_est", "top_x_spec", "bottom_x_spec", "ai_lab_logratio",
    "sens_temp_dev", "level_std_1h", "reflux_ratio_rel",
]


def episode_signature(feat: pd.DataFrame, i0: int, i1: int, ref_len: int) -> dict:
    """Median of each signature feature inside the episode and its shift from the column reference."""
    win = feat.iloc[i0:i1]
    ref = feat.iloc[:ref_len]
    sig = {}
    for f in SIGNATURE_FEATURES:
        sig[f] = float(win[f].median())
        sig[f + "_shift"] = float(win[f].median() - ref[f].median())
    sig["dp_min_resid_p10"] = float(win["dp_min_resid"].quantile(0.10))
    sig["wm_min_p10"] = float(win["wm_min"].quantile(0.10))
    sig["steam_osc_max"] = float(win["steam_osc_3h"].max())
    sig["op_sat_frac_max"] = float(win["op_sat_frac_3h"].max())
    # trend of the reboiler fouling index across the 30 days before the episode
    pre = feat["chest_ua_index"].iloc[max(0, i0 - 30 * 288):i0 + 1]
    sig["chest_ua_trend_30d"] = float(pre.iloc[-288:].median() - pre.iloc[:288].median()) if len(pre) > 600 else 0.0
    # did a pressure spike or feed trip happen in the 72 hours before
    pre72 = feat.iloc[max(0, i0 - 72 * 12):i0 + 1]
    sig["pre_feed_min"] = float(pre72["feed_ratio"].min()) if len(pre72) else 1.0
    sig["pre_pressure_max"] = float(pre72["pressure_ratio"].max()) if len(pre72) else 1.0
    return sig


def detect_episodes(probs: dict[str, np.ndarray], feats: dict[str, pd.DataFrame], cfg: dict, step_min: int,
                    ref_days: float) -> list[DetectedEpisode]:
    """Segment both tracks. Weeping segments closer than incident_gap_hours are consolidated into
    one incident, because intermittent weeping (stiction troughs, load swings on damaged trays)
    belongs to one investigation, not dozens."""
    smooth = int(cfg["smoothing_samples"])
    min_len = int(cfg["min_episode_minutes"] / step_min)
    gap = int(cfg["merge_gap_minutes"] / step_min)
    incident_gap = int(float(cfg.get("incident_gap_hours", 36)) * 60 / step_min)
    ref_len = int(ref_days * 1440 / step_min)
    found = []
    for column, p in probs.items():
        idx = feats[column].index
        tracks = {"weeping": p[:, SAMPLE_CLASSES.index("weeping")],
                  "lookalike": p[:, [SAMPLE_CLASSES.index(k) for k in ("flooding", "foaming", "analyzer_fault")]].sum(axis=1)}
        for track, pt in tracks.items():
            g = incident_gap if track == "weeping" else gap
            for a, b in hysteresis_segments(pt, cfg["enter_threshold"], cfg["exit_threshold"], smooth, min_len, g):
                mix = {k: float(p[a:b, i].mean()) for i, k in enumerate(SAMPLE_CLASSES)}
                kind = "weeping" if track == "weeping" else max(("flooding", "foaming", "analyzer_fault"), key=lambda k: mix[k])
                found.append(dict(column=column, kind=kind, a=a, b=b, start=idx[a], end=idx[min(b, len(idx) - 1)],
                                  mean_prob=float(pt[a:b].mean()), peak_prob=float(pt[a:b].max()), mix=mix,
                                  sig=episode_signature(feats[column], a, b, ref_len)))
    found.sort(key=lambda e: (e["start"], e["column"]))
    out = []
    for k, e in enumerate(found, 1):
        out.append(DetectedEpisode(
            episode_id=f"DE{k:04d}", column=e["column"], kind=e["kind"], start=e["start"].isoformat(), end=e["end"].isoformat(),
            duration_h=round((e["end"] - e["start"]).total_seconds() / 3600.0, 2), start_idx=int(e["a"]), end_idx=int(e["b"]),
            mean_prob=round(e["mean_prob"], 4), peak_prob=round(e["peak_prob"], 4), class_mix=e["mix"], signature=e["sig"],
        ))
    return out
