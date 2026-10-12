"""Verifier agent: the guardrail between analysis and the written report.

Every claim is checked on four rules:

1. Citation present: at least one evidence id.
2. Citation real: every id exists in the evidence index (no invented work orders or logs).
3. Citation in time: time stamped evidence for an episode claim must sit inside the
   investigation window (history and recommendation claims are exempt).
4. Numbers grounded: every number in the sentence must equal a value in the fact registry
   at the displayed precision, or appear in the text of a cited record.

Claims that fail are removed from the published report and listed in the verification
block, so a reader can see exactly what was suppressed and why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .base import Claim, EvidenceIndex, FactRegistry

ID_PATTERNS = [
    r"\[[^\]]*\]",                          # citations
    r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b",  # dates and times
    r"\b\d{1,2}:\d{2}\b",
    r"\b[A-Z]{1,4}_?[A-Z]*\d+[A-Z0-9_.#]*\b",   # tags and record ids such as FV207, WO410233, SL20260314D2.3, KB004
    r"\bSIG_[A-Z_]+\b", r"\bTXT_[A-Z_]+\b", r"\bCUE_\S+\b",
]
NUM_RE = re.compile(r"(?<![A-Za-z_])[+-]?\d[\d,]*(?:\.\d+)?")
TIME_BOUND = {"shift_log", "work_order", "alarm", "operator_action", "lab", "gamma_scan", "moc", "alarm_burst"}
EXEMPT_KINDS = {"history", "recommendation", "method"}


@dataclass
class ClaimCheck:
    claim: Claim
    ok: bool
    reasons: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)


def strip_ids(text: str) -> str:
    out = text
    for p in ID_PATTERNS:
        out = re.sub(p, " ", out)
    return out


def numbers_in(text: str) -> list[str]:
    return [m.group(0) for m in NUM_RE.finditer(strip_ids(text))]


def _matches(num: str, allowed: list[float]) -> bool:
    raw = num.replace(",", "").lstrip("+")
    try:
        x = abs(float(raw))
    except ValueError:
        return True
    dec = len(raw.split(".")[1]) if "." in raw else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return any(abs(x - abs(v)) <= tol for v in allowed)


class Verifier:
    name = "verifier"

    def __init__(self, lookback_h: float, lookahead_h: float, constants: list[float] | None = None):
        self.lookback_h = lookback_h
        self.lookahead_h = lookahead_h
        self.constants = constants or []

    def check(self, claim: Claim, evidence: EvidenceIndex, facts: FactRegistry, t0: pd.Timestamp, t1: pd.Timestamp) -> ClaimCheck:
        reasons = []
        if not claim.evidence:
            reasons.append("no citation")
        missing = [e for e in claim.evidence if e not in evidence]
        if missing:
            reasons.append("unknown evidence " + ", ".join(missing))
        if claim.kind not in EXEMPT_KINDS:
            lo = t0 - pd.Timedelta(hours=self.lookback_h)
            hi = t1 + pd.Timedelta(hours=self.lookahead_h)
            for e in claim.evidence:
                it = evidence.get(e)
                if it is None or it.kind not in TIME_BOUND or it.time is None:
                    continue
                t = pd.Timestamp(it.time)
                t_end = pd.Timestamp(it.meta["time_end"]) if it.meta.get("time_end") else t
                if t_end < lo or t > hi:
                    reasons.append(f"evidence {e} outside the investigation window")
        if claim.facts:
            # strict mode: the sentence declared which registry entries it states
            unknown = [f for f in claim.facts if f not in facts.facts]
            if unknown:
                reasons.append("unknown facts " + ", ".join(unknown))
            allowed = [facts.facts[f].value for f in claim.facts if f in facts.facts]
            nums = numbers_in(claim.text)
            bad = [n for n in nums if not _matches(n, allowed)]
            if bad:
                reasons.append("numbers do not match declared facts " + ", ".join(bad))
            return ClaimCheck(claim, not reasons, reasons, nums)
        allowed = facts.values() + self.constants
        cited_numbers = []
        for e in claim.evidence:
            it = evidence.get(e)
            if it is not None:
                for n in numbers_in(it.text):
                    try:
                        cited_numbers.append(float(n.replace(",", "").lstrip("+")))
                    except ValueError:
                        pass
        nums = numbers_in(claim.text)
        bad = [n for n in nums if not _matches(n, allowed + cited_numbers)]
        if bad:
            reasons.append("ungrounded numbers " + ", ".join(bad))
        return ClaimCheck(claim, not reasons, reasons, nums)

    def run(self, claims: list[Claim], evidence: EvidenceIndex, facts: FactRegistry, t0, t1) -> tuple[list[Claim], dict]:
        checks = [self.check(c, evidence, facts, pd.Timestamp(t0), pd.Timestamp(t1)) for c in claims]
        kept = [c.claim for c in checks if c.ok]
        n_nums = sum(len(c.numbers) for c in checks)
        summary = {
            "claims_total": len(checks),
            "claims_supported": sum(c.ok for c in checks),
            "claims_removed": [{"text": c.claim.text, "reasons": c.reasons} for c in checks if not c.ok],
            "citation_coverage": round(sum(c.ok for c in checks) / max(len(checks), 1), 4),
            "numbers_checked": n_nums,
            "evidence_items": len(evidence.items),
        }
        return kept, summary
