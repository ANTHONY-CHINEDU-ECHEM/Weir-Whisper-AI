"""Small IO helpers shared by every stage of the pipeline."""
from __future__ import annotations

import json
import logging
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

LOG = logging.getLogger("weir_whisper")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, cls=_Encoder, indent=indent, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf8") as fh:
        return json.load(fh)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf8") as fh:
        for row in rows:
            fh.write(json.dumps(row, cls=_Encoder, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


@contextmanager
def timed(label: str) -> Iterator[None]:
    t0 = time.perf_counter()
    LOG.info("%s started", label)
    yield
    LOG.info("%s finished in %.1f s", label, time.perf_counter() - t0)


def ts(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x)


def iso(x: Any) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d %H:%M")
