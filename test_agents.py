import json

import pandas as pd

from weir_whisper.agents.base import Claim, EvidenceIndex, FactRegistry
from weir_whisper.agents.root_cause import RootCauseAgent
from weir_whisper.agents.signal_analyst import Observation
from weir_whisper.agents.verifier import Verifier
from weir_whisper.analytics.episodes import hysteresis_segments
from weir_whisper.llm.backends import LLMResult, parse_json_block
from weir_whisper.simulator.scenarios import ALL_CAUSES


def test_posterior_is_a_distribution_and_follows_evidence():
    agent = RootCauseAgent()
    dx = agent.diagnose([Observation("SIG_X", "steam cycling", {"steam_valve_stiction": 4.0, "turndown": -1.0})], [])
    assert abs(sum(dx.posterior.values()) - 1.0) < 1e-9
    assert dx.cause == "steam_valve_stiction"
    assert set(dx.posterior) == set(ALL_CAUSES)


def test_temperature_softens_confidence():
    obs = [Observation("SIG_X", "s", {"turndown": 3.0})]
    sharp = RootCauseAgent().diagnose(obs, []).posterior["turndown"]
    soft = RootCauseAgent(temperature=2.0).diagnose(obs, []).posterior["turndown"]
    assert soft < sharp


def _evidence():
    ev = EvidenceIndex()
    ev.add("SL20260101D2.1", "shift_log", "FV207 hunting, swinging by 3.4 t/h", "2026-01-01 08:00")
    ev.add("SL20250601D2.1", "shift_log", "old note", "2025-06-01 08:00")
    ev.add("SIG_IMPACT", "signal", "method note", "2026-01-01 00:00")
    fr = FactRegistry()
    fr.add("total_cost_gbp", 12345.0, "GBP", "test")
    return ev, fr


def test_verifier_accepts_grounded_claims_and_rejects_bad_ones():
    ev, fr = _evidence()
    v = Verifier(72, 48, [72])
    t0, t1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")
    assert v.check(Claim("Cost GBP 12,345.", ["SIG_IMPACT"], "quantity", facts=["total_cost_gbp"]), ev, fr, t0, t1).ok
    assert v.check(Claim("Steam swung by 3.4 t/h.", ["SL20260101D2.1"]), ev, fr, t0, t1).ok
    assert not v.check(Claim("Cost GBP 15,000.", ["SIG_IMPACT"], "quantity", facts=["total_cost_gbp"]), ev, fr, t0, t1).ok
    assert not v.check(Claim("Valve was hunting.", ["WO999999"]), ev, fr, t0, t1).ok
    assert not v.check(Claim("Valve was hunting.", ["SL20250601D2.1"]), ev, fr, t0, t1).ok
    assert not v.check(Claim("Valve was hunting.", []), ev, fr, t0, t1).ok
    assert not v.check(Claim("Steam swung by 9.9 t/h.", ["SL20260101D2.1"]), ev, fr, t0, t1).ok


def test_verifier_ignores_identifiers_and_dates():
    ev, fr = _evidence()
    v = Verifier(72, 48, [])
    c = Claim("On 2026-01-01 08:00 the note SL20260101D2.1 mentioned FV207.", ["SL20260101D2.1"])
    assert v.check(c, ev, fr, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")).ok


class FakeBackend:
    name = "fake"
    model = "fake-model"

    def complete(self, system, prompt):
        data = {"summary": [{"text": "Steam swung by 3.4 t/h.", "evidence": ["SL20260101D2.1"]},
                            {"text": "The valve was replaced in 2019.", "evidence": ["WO123456"]},
                            {"text": "Cost was GBP 99,999.", "evidence": ["SIG_IMPACT"]}]}
        return LLMResult(text="Here is the JSON: " + json.dumps(data), backend="fake", model=self.model)


def test_llm_claims_are_filtered_by_the_verifier():
    ev, fr = _evidence()
    res = FakeBackend().complete("", "")
    data = parse_json_block(res.text)
    claims = [Claim(c["text"], c["evidence"], "inference", "summary") for c in data["summary"]]
    kept, summary = Verifier(72, 48, []).run(claims, ev, fr, "2026-01-01", "2026-01-02")
    assert [c.text for c in kept] == ["Steam swung by 3.4 t/h."]
    assert summary["claims_total"] == 3 and len(summary["claims_removed"]) == 2


def test_hysteresis_segmentation():
    import numpy as np
    p = np.zeros(200)
    p[20:40] = 0.9
    p[44:60] = 0.9      # short gap, merged
    p[150:152] = 0.9    # too short after smoothing
    segs = hysteresis_segments(p, 0.55, 0.3, smooth=3, min_len=6, merge_gap=6)
    assert len(segs) == 1
    assert segs[0][0] <= 21 and segs[0][1] >= 59
