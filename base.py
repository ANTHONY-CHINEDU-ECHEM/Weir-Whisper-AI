"""Shared structures for the investigation agents.

EvidenceIndex   every item a report may cite: graph records, knowledge chunks and computed
                signal observations (ids starting SIG_). Each has a type, a time and text.
FactRegistry    every number a report may state, with its unit and the method that produced
                it. The verifier refuses any number that is not registered here or quoted in
                the cited evidence.
Claim           one sentence of a report with the evidence ids that support it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class EvidenceItem:
    eid: str
    kind: str
    text: str
    time: str | None = None
    weight: float = 1.0
    meta: dict = field(default_factory=dict)


class EvidenceIndex:
    def __init__(self):
        self.items: dict[str, EvidenceItem] = {}

    def add(self, eid: str, kind: str, text: str, time=None, weight: float = 1.0, **meta) -> str:
        t = pd.Timestamp(time).isoformat() if time is not None and not pd.isna(time) else None
        if eid not in self.items:
            self.items[eid] = EvidenceItem(eid, kind, text, t, weight, meta)
        return eid

    def __contains__(self, eid: str) -> bool:
        return eid in self.items

    def get(self, eid: str) -> EvidenceItem | None:
        return self.items.get(eid)

    def to_dict(self) -> dict:
        return {k: asdict(v) for k, v in self.items.items()}


@dataclass
class Fact:
    name: str
    value: float
    unit: str
    method: str


class FactRegistry:
    def __init__(self):
        self.facts: dict[str, Fact] = {}

    def add(self, name: str, value: float, unit: str, method: str) -> float:
        self.facts[name] = Fact(name, float(value), unit, method)
        return float(value)

    def values(self) -> list[float]:
        return [f.value for f in self.facts.values()]

    def to_dict(self) -> dict:
        return {k: asdict(v) for k, v in self.facts.items()}


@dataclass
class Claim:
    text: str
    evidence: list[str]
    kind: str = "observation"   # observation, inference, quantity, recommendation, history
    section: str = "summary"
    facts: list[str] = field(default_factory=list)  # registry names the sentence's numbers must come from (strict mode)

    def to_dict(self) -> dict:
        return asdict(self)


def fmt_num(x: float, digits: int = 0) -> str:
    """Format numbers the same way everywhere so the verifier can match them."""
    if digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def gbp(x: float) -> str:
    return f"GBP {x:,.0f}"
