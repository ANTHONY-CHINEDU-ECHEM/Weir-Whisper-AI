import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def settings():
    from weir_whisper.config import load_settings
    return load_settings("default")


@pytest.fixture(scope="session")
def tiny_run():
    """Run the whole pipeline once on the tiny profile and hand the settings to the tests."""
    os.environ["WEIR_WORKERS"] = "1"
    from weir_whisper import pipeline
    from weir_whisper.config import load_settings
    s = load_settings("tiny")
    pipeline.stage_generate(s)
    pipeline.stage_analyse(s)
    pipeline.stage_graph(s)
    pipeline.stage_index(s)
    pipeline.stage_investigate(s, [])
    pipeline.stage_evaluate(s)
    return s
