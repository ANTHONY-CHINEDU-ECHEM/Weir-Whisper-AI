"""Weir Whisper AI command line entry point.

Usage (no flags, so commands read cleanly in any shell):

    python run.py generate          build the synthetic plant dataset
    python run.py analyse           soft sensors, detector training, episodes, impact
    python run.py graph             build the knowledge graph
    python run.py index             build the hybrid retrieval index
    python run.py investigate       investigate every detected episode and write reports
    python run.py investigate DE0007   investigate a single detected episode
    python run.py ask "why did C201 lose separation in March"
    python run.py evaluate          run the full evaluation suite
    python run.py figures           render the README figures
    python run.py serve             start the REST API
    python run.py all               run the whole pipeline end to end

Set WEIR_PROFILE=quick to run on the small smoke profile.
"""
from __future__ import annotations

import sys

from weir_whisper.pipeline import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
