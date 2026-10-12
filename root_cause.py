"""Root cause agent: evidence weighted Bayesian scoring over nine hypotheses.

posterior(cause) is proportional to prior(cause) times exp(sum of log likelihood ratio
contributions from every signal observation and record cue). Signals and records are
kept as separate channels so the report can show which kind of evidence carried the
decision, and so the evaluation can ablate them. The agent also explains the decision:
the strongest supporting observations for the winner and the strongest evidence against
each rejected alternative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulator.scenarios import ALL_CAUSES, CAUSE_LABELS
from .signal_analyst import Observation


@dataclass
class Diagnosis:
    cause: str
    label: str
    posterior: dict[str, float]
    score: dict[str, float]
    support: list[tuple[str, float, str]]
    against: dict[str, list[tuple[str, float, str]]]
    channels: dict[str, dict[str, float]]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.posterior.items(), key=lambda kv: -kv[1])

    def to_dict(self) -> dict:
        return {
            "cause": self.cause, "label": self.label,
            "posterior": {k: round(v, 4) for k, v in self.ranked()},
            "support": [{"evidence": e, "llr": round(w, 3), "statement": s} for e, w, s in self.support],
            "against": {c: [{"evidence": e, "llr": round(w, 3), "statement": s} for e, w, s in v] for c, v in self.against.items()},
            "channels": {k: {c: round(x, 3) for c, x in v.items()} for k, v in self.channels.items()},
        }


class RootCauseAgent:
    name = "root_cause"

    def __init__(self, priors: dict[str, float] | None = None, text_weight: float = 1.0, signal_weight: float = 1.0,
                 temperature: float = 1.0):
        self.priors = priors or {c: 1.0 / len(ALL_CAUSES) for c in ALL_CAUSES}
        self.text_weight = text_weight
        self.signal_weight = signal_weight
        self.temperature = temperature  # above one softens the posterior; fitted on held out episodes

    def diagnose(self, signal_obs: list[Observation], record_obs: list[Observation], use: tuple[str, ...] = ("signal", "records")) -> Diagnosis:
        chan = {"signal": {c: 0.0 for c in ALL_CAUSES}, "records": {c: 0.0 for c in ALL_CAUSES}}
        contrib: list[tuple[str, str, float, str]] = []
        for name, obs_list, w in (("signal", signal_obs, self.signal_weight), ("records", record_obs, self.text_weight)):
            if name not in use:
                continue
            for o in obs_list:
                for c, v in o.contributions.items():
                    if c in chan[name]:
                        chan[name][c] += w * v
                        contrib.append((c, o.eid, w * v, o.statement))
        score = {c: np.log(self.priors[c]) + chan["signal"][c] + chan["records"][c] for c in ALL_CAUSES}
        mx = max(score.values())
        ex = {c: np.exp((v - mx) / self.temperature) for c, v in score.items()}
        z = sum(ex.values())
        post = {c: float(v / z) for c, v in ex.items()}
        best = max(post, key=post.get)
        support = sorted([(e, v, s) for c, e, v, s in contrib if c == best and v > 0], key=lambda x: -x[1])[:6]
        against = {}
        for c, _ in sorted(post.items(), key=lambda kv: -kv[1])[1:4]:
            neg = sorted([(e, v, s) for cc, e, v, s in contrib if cc == c and v < 0], key=lambda x: x[1])[:3]
            # if nothing argues against an alternative directly, record that it simply lacks support
            against[c] = neg if neg else [("", 0.0, "")]
        return Diagnosis(cause=best, label=CAUSE_LABELS[best], posterior=post, score=score, support=support, against=against, channels=chan)
