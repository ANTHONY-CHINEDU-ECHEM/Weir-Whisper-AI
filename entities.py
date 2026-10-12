"""Entity linking and symptom extraction for plant text.

Operators write FV-207, fv207, FV 207 or "the depropaniser" for the same things. The
linker normalises every tag shaped token against the site registry, resolves column
nicknames, and tags each text with symptom families drawn from the efficiency loss
symptom matrix. Unknown tag shaped tokens are kept as unresolved mentions so a reviewer
can see what the linker could not place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..simulator.tags import EQUIPMENT_DESCRIPTIONS, equipment

TAG_RE = re.compile(r"\b(PSV|PDI|PIC|FIC|TIC|LIC|FI|TI|AI|FV|LV|XV|PI|E|C|P|V)[\s\-_]?(\d{3})([AB])?\b", re.IGNORECASE)

COLUMN_NICKNAMES = {
    "debutaniser": "C101", "debutanizer": "C101",
    "depropaniser": "C201", "depropanizer": "C201",
    "deethaniser": "C301", "deethanizer": "C301",
    "c3 splitter": "C401", "propylene splitter": "C401", "splitter": "C401",
    "naphtha splitter": "C501",
    "stabiliser": "C601", "stabilizer": "C601", "condensate stabiliser": "C601",
}

SYMPTOMS: dict[str, list[str]] = {
    "weeping": [r"weep", r"dumping", r"\bdump", r"lost the profile", r"profile (flattening|not recovered)", r"temps? sagging",
                r"not fractionating", r"converging", r"efficiency down", r"separation poor", r"froth density"],
    "dp_low": [r"d\s?p\b[^.]{0,40}\b(low|dropped|below|lower)", r"(low|lower)[^.]{0,20}\bd\s?p\b", r"pdi[\s\-]?\d{3}[^.]{0,15}low"],
    "dp_high": [r"d\s?p\b[^.]{0,30}\b(high|spik|erratic|jumping)", r"pdi[\s\-]?\d{3}[^.]{0,20}(near alarm|jumping)"],
    "cycling": [r"cycl", r"hunting", r"oscillat", r"sawtooth", r"stick", r"jumping", r"will not settle", r"swinging by"],
    "valve_saturated": [r"wide open", r"near 100", r"pinned", r"at 100", r"cannot reach setpoint", r"valve at max"],
    "chest_high": [r"chest pressure high", r"chest pressure"],
    "fouling": [r"foul", r"deposit", r"duty (limited|falling)", r"underperforming", r"hydro jet", r"approach"],
    "pressure_high": [r"pressure (setpoint|sp)[^.]{0,20}(raised|moved up)", r"running high on pressure", r"pressure creeping",
                      r"higher pressure", r"overhead pressure (running )?high", r"raise .* operating pressure", r"\bsp moved up"],
    "cold_feed": [r"cold feed", r"feed (now |still )?cold", r"preheater isolated", r"bypassed", r"feed temp", r"lost e[\s\-]?\d03",
                  r"feed preheat"],
    "trip_upset": [r"trip", r"psv", r"lifted", r"popped", r"upset", r"shutdown", r"pressure spike"],
    "rate_cut": [r"feed cut", r"rates? reduced", r"minimum throughput", r"min(imum)? rates", r"low rates", r"rate cut",
                 r"feed down to", r"bring .{0,20}feed down", r"minimum boilup"],
    "max_rates": [r"max rates", r"max throughput", r"pushed to max", r"max rate campaign"],
    "flooding": [r"flood", r"entrainment", r"carryover", r"carry over"],
    "foaming": [r"foam", r"antifoam", r"contamin", r"amine", r"inhibitor"],
    "analyzer_issue": [r"analy[sz]er", r"lab (disagrees|says|result on spec)", r"mismatch", r"recalibrat", r"versus lab", r"vs lab"],
    "tray_damage": [r"tray damage", r"gamma scan", r"missing panels", r"dislodged", r"displaced panels", r"tray repair"],
    "resolved": [r"back to normal", r"restored", r"recovered", r"steady now", r"repacked", r"cleaned", r"fixed", r"repaired",
                 r"back in service", r"returned to service", r"gone after", r"agrees with lab", r"much better"],
}
_SYM_RE = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in SYMPTOMS.items()}


@dataclass
class Mention:
    entity: str
    kind: str
    surface: str


@dataclass
class Extraction:
    entities: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)


class EntityLinker:
    def __init__(self, columns: list[str], tag_dict_rows: list[dict] | None = None):
        self.registry: dict[str, dict] = {}
        for c in columns:
            for role, tag in equipment(c).items():
                self.registry[tag] = {"kind": "equipment" if role not in ("steam_controller", "pressure_controller", "temp_controller") else "controller",
                                      "column": c, "role": role, "description": EQUIPMENT_DESCRIPTIONS[role]}
            u = c[1]
            # tags that are not in the equipment list (transmitters, extra valves)
            for extra in (f"FI{u}02", f"TI{u}03", f"FIC{u}04", f"FI{u}05", f"FI{u}06", f"PDI{u}09", f"PDI{u}10", f"TI{u}11",
                          f"TI{u}13", f"LIC{u}14", f"PI{u}17", f"LV{u}14", f"FV{u}04", f"XV{u}20"):
                self.registry.setdefault(extra, {"kind": "instrument", "column": c, "role": "instrument", "description": "instrument"})
        if tag_dict_rows:
            for row in tag_dict_rows:
                tag = row["tag"].split("_")[0]
                self.registry.setdefault(tag, {"kind": "instrument", "column": row["column"], "role": row["role"], "description": row["description"]})
        self.columns = set(columns)

    @staticmethod
    def normalise(prefix: str, digits: str, suffix: str | None) -> str:
        return f"{prefix.upper()}{digits}{(suffix or '').upper()}"

    def extract(self, text: str) -> Extraction:
        ex = Extraction()
        seen = set()
        for m in TAG_RE.finditer(text):
            tag = self.normalise(m.group(1), m.group(2), m.group(3))
            if tag in self.registry or tag in self.columns:
                if tag not in seen:
                    seen.add(tag)
                    kind = "column" if tag in self.columns else self.registry[tag]["kind"]
                    ex.mentions.append(Mention(tag, kind, m.group(0)))
                    ex.entities.append(tag)
                    col = tag if tag in self.columns else self.registry[tag]["column"]
                    if col not in ex.columns:
                        ex.columns.append(col)
            else:
                ex.unresolved.append(m.group(0))
        low = text.lower()
        for nick, col in COLUMN_NICKNAMES.items():
            if nick in low and col in self.columns and col not in ex.columns:
                ex.columns.append(col)
                ex.mentions.append(Mention(col, "column", nick))
                if col not in seen:
                    ex.entities.append(col)
                    seen.add(col)
        for sym, pats in _SYM_RE.items():
            if any(p.search(text) for p in pats):
                ex.symptoms.append(sym)
        # "weep" also describes small leaks at glands, seals and gauge glasses; that is not tray weeping
        if "weeping" in ex.symptoms and re.search(r"gland|seal|gauge glass|flange", low) and not re.search(r"tray", low):
            ex.symptoms.remove("weeping")
        return ex

    def column_of(self, entity: str) -> str | None:
        if entity in self.columns:
            return entity
        info = self.registry.get(entity)
        return info["column"] if info else None


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with plant tags normalised so FV-207 and fv207 match."""
    norm = TAG_RE.sub(lambda m: f" {m.group(1)}{m.group(2)}{m.group(3) or ''} ", text)
    return [t for t in re.findall(r"[a-z0-9]+", norm.lower()) if len(t) > 1 or t.isdigit()]
