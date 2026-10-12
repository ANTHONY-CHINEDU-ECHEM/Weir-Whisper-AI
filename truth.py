"""Ground truth access.

This is the only module that reads data/raw/ground_truth. Two consumers use it:
the detector, which needs historical labels to learn from (on a real site these labels
come from past investigation records), and the evaluation suite, which scores every
stage. Nothing in the investigation path (graph, retrieval, agents, reports) imports it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..simulator.scenarios import SAMPLE_CLASSES
from ..utils.io import read_json


def sample_labels(settings, column: str) -> np.ndarray:
    gt = pd.read_parquet(settings.path("raw") / "ground_truth" / f"signals_{column}.parquet", columns=["class"])
    lut = {k: i for i, k in enumerate(SAMPLE_CLASSES)}
    return gt["class"].map(lut).to_numpy().astype(int)


def signals(settings, column: str) -> pd.DataFrame:
    return pd.read_parquet(settings.path("raw") / "ground_truth" / f"signals_{column}.parquet").set_index("timestamp")


def episodes(settings) -> list[dict]:
    return read_json(settings.path("raw") / "ground_truth" / "episodes.json")


def evidence_links(settings) -> dict[str, list[str]]:
    return read_json(settings.path("raw") / "ground_truth" / "evidence_links.json")
