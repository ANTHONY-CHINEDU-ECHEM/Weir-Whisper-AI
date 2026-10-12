"""Configuration loading.

The default profile lives in config/default.yaml. A second profile can be layered on
top by setting the WEIR_PROFILE environment variable (for example WEIR_PROFILE=quick
loads config/quick.yaml and deep merges it over the defaults).
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            # schedule blocks are replaced wholesale so a profile can drop columns
            if key == "schedule":
                out[key] = copy.deepcopy(value)
            else:
                out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class Settings:
    raw: dict[str, Any]
    plant: dict[str, Any] = field(default_factory=dict)
    profile: str = "default"

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.raw.get(item, default)

    def path(self, key: str) -> Path:
        p = ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def columns(self) -> list[str]:
        return list(self.raw["simulation"]["columns"])

    def column(self, tag: str) -> dict[str, Any]:
        return self.plant["columns"][tag]


def load_settings(profile: str | None = None) -> Settings:
    profile = profile or os.environ.get("WEIR_PROFILE", "default")
    with open(ROOT / "config" / "default.yaml", "r", encoding="utf8") as fh:
        raw = yaml.safe_load(fh)
    if profile and profile != "default":
        prof_path = ROOT / "config" / f"{profile}.yaml"
        if not prof_path.exists():
            raise FileNotFoundError(f"Profile {profile} not found at {prof_path}")
        with open(prof_path, "r", encoding="utf8") as fh:
            raw = deep_merge(raw, yaml.safe_load(fh))
    with open(ROOT / raw["paths"]["plant"], "r", encoding="utf8") as fh:
        plant = yaml.safe_load(fh)
    return Settings(raw=raw, plant=plant, profile=profile)
