"""Supervised weeping detector with look alike discrimination.

A histogram gradient boosting classifier predicts five states per sample: normal,
weeping, flooding, foaming and analyzer fault. Training labels come from historical
investigations (in this repository, the simulator ground truth stands in for them).

Three evaluation protocols are produced:

1. Forward split: train before train_end, test after. This is the headline number.
2. Blocked temporal cross fitting: every month is scored by a model that never saw it,
   which gives honest probabilities for the whole year so investigations can run on it.
3. Leave one column out: train on five columns, score the sixth, to test whether the
   dimensionless features transfer to an unseen column.

Two baselines are scored on the same protocols: a pure physics rule (estimated weep
margin below one or section dP far below expected) and an unsupervised isolation forest.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from ..simulator.scenarios import SAMPLE_CLASSES
from .features import FEATURES

LOG = logging.getLogger("weir_whisper.detector")


def class_weights(y: np.ndarray, cap: float = 25.0) -> np.ndarray:
    counts = np.bincount(y, minlength=len(SAMPLE_CLASSES)).astype(float)
    w = np.where(counts > 0, len(y) / (len(SAMPLE_CLASSES) * np.maximum(counts, 1)), 0.0)
    w = np.clip(w, 0.0, cap)
    return w[y]


def make_model(cfg: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=int(cfg["max_iter"]), learning_rate=float(cfg["learning_rate"]), max_leaf_nodes=int(cfg["max_leaf_nodes"]),
        l2_regularization=1.0, min_samples_leaf=60, random_state=seed, early_stopping=False,
    )


def downsample(X: np.ndarray, y: np.ndarray, seed: int, max_rows: int = 160000) -> tuple[np.ndarray, np.ndarray]:
    """Keep every abnormal sample and a random share of normal ones so training stays fast.

    Class weights are computed after sampling, so the model still sees a balanced problem.
    """
    if len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    abn = np.where(y != 0)[0]
    nor = np.where(y == 0)[0]
    keep_n = max(max_rows - len(abn), len(abn))
    keep = np.concatenate([abn, rng.choice(nor, min(keep_n, len(nor)), replace=False)])
    keep.sort()
    return X[keep], y[keep]


def fit(X: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> HistGradientBoostingClassifier:
    X, y = downsample(X, y, seed, int(cfg.get("max_train_rows", 160000)))
    model = make_model(cfg, seed)
    model.fit(X, y, sample_weight=class_weights(y))
    return model


def predict_full(model, X: np.ndarray) -> np.ndarray:
    """Probabilities for all five classes even if a class was absent in training."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(SAMPLE_CLASSES)))
    out[:, model.classes_] = p
    return out


def physics_rule_score(feat: pd.DataFrame) -> np.ndarray:
    """Baseline: margin below one or dP at least fifteen percent below expected."""
    wm = feat["wm_min"].to_numpy()
    dp = feat["dp_min_resid"].to_numpy()
    s1 = np.clip((1.05 - wm) / 0.25, 0, 1)
    s2 = np.clip((-dp - 0.08) / 0.15, 0, 1)
    return np.maximum(s1, s2)


def isolation_score(train_X: np.ndarray, X: np.ndarray, seed: int) -> np.ndarray:
    iso = IsolationForest(n_estimators=200, max_samples=4096, contamination="auto", random_state=seed).fit(train_X)
    s = -iso.score_samples(X)
    lo, hi = np.percentile(s, 50), np.percentile(s, 99.5)
    return np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)


def stack(frames: dict[str, pd.DataFrame], labels: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    parts, ys = [], []
    for c, f in frames.items():
        g = f[FEATURES].copy()
        g["column"] = c
        parts.append(g)
        ys.append(labels[c])
    return pd.concat(parts), np.concatenate(ys)


def temporal_folds(index: pd.DatetimeIndex, k: int) -> list[np.ndarray]:
    edges = pd.date_range(index[0], index[-1] + pd.Timedelta(minutes=5), periods=k + 1)
    return [np.where((index >= edges[i]) & (index < edges[i + 1]))[0] for i in range(k)]


def run_detection(frames: dict[str, pd.DataFrame], labels: dict[str, np.ndarray], cfg: dict, seed: int,
                  ref_days: float) -> dict:
    """Train all protocols. Returns out of fold probabilities, forward split scores and the production model."""
    columns = list(frames)
    index = frames[columns[0]].index
    k = int(cfg.get("cv_folds", 4))
    folds = temporal_folds(index, k)
    gap = 288  # one day embargo either side of a test fold to stop rolling windows leaking
    oof = {c: np.zeros((len(index), len(SAMPLE_CLASSES))) for c in columns}
    for fi, test_idx in enumerate(folds):
        lo, hi = test_idx[0] - gap, test_idx[-1] + gap
        train_mask = np.ones(len(index), dtype=bool)
        train_mask[max(lo, 0):min(hi + 1, len(index))] = False
        X = np.concatenate([frames[c][FEATURES].to_numpy()[train_mask] for c in columns])
        y = np.concatenate([labels[c][train_mask] for c in columns])
        model = fit(X, y, cfg, seed + fi)
        for c in columns:
            oof[c][test_idx] = predict_full(model, frames[c][FEATURES].to_numpy()[test_idx])
        LOG.info("temporal fold %d of %d scored", fi + 1, k)

    # forward split
    t_end = pd.Timestamp(cfg["train_end"])
    tr = index < t_end - pd.Timedelta(days=1)
    te = index >= t_end
    X_tr = np.concatenate([frames[c][FEATURES].to_numpy()[tr] for c in columns])
    y_tr = np.concatenate([labels[c][tr] for c in columns])
    fwd_model = fit(X_tr, y_tr, cfg, seed)
    forward = {c: predict_full(fwd_model, frames[c][FEATURES].to_numpy()) for c in columns}

    # baselines on the forward split
    ref = int(ref_days * 288)
    iso_train = np.concatenate([frames[c][FEATURES].to_numpy()[:ref] for c in columns])
    baselines = {c: {"physics_rule": physics_rule_score(frames[c]),
                     "isolation_forest": isolation_score(iso_train, frames[c][FEATURES].to_numpy(), seed)} for c in columns}

    # leave one column out
    loco = {}
    if len(columns) > 2:
        for c in columns:
            others = [o for o in columns if o != c]
            X = np.concatenate([frames[o][FEATURES].to_numpy() for o in others])
            y = np.concatenate([labels[o] for o in others])
            m = fit(X, y, cfg, seed)
            loco[c] = predict_full(m, frames[c][FEATURES].to_numpy())
            LOG.info("leave one column out: %s scored", c)

    # production model on everything
    X_all = np.concatenate([frames[c][FEATURES].to_numpy() for c in columns])
    y_all = np.concatenate([labels[c] for c in columns])
    production = fit(X_all, y_all, cfg, seed)
    importance = feature_importance(production, X_all, y_all, seed)
    return {"oof": oof, "forward": forward, "train_mask": tr, "test_mask": te, "baselines": baselines,
            "loco": loco, "model": production, "importance": importance}


def feature_importance(model, X: np.ndarray, y: np.ndarray, seed: int, n: int = 40000) -> dict[str, float]:
    """Permutation importance on the weeping class log loss, on a stratified subsample."""
    rng = np.random.default_rng(seed)
    idx_w = np.where(y == 1)[0]
    idx_o = np.where(y != 1)[0]
    take = np.concatenate([rng.choice(idx_w, min(len(idx_w), n // 2), replace=False),
                           rng.choice(idx_o, min(len(idx_o), n // 2), replace=False)])
    Xs, ys = X[take], (y[take] == 1).astype(float)

    def loss(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p))

    base = loss(predict_full(model, Xs)[:, 1])
    out = {}
    for j, name in enumerate(FEATURES):
        Xp = Xs.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        out[name] = float(loss(predict_full(model, Xp)[:, 1]) - base)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
