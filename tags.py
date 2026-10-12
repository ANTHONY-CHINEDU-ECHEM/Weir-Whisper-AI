"""Tag and equipment naming conventions for the simulated site.

Every column C{u}01 owns a block of instrument tags numbered {u}02 to {u}17 and a set of
equipment items. The analytics layer never hard codes a tag: it resolves measurement
roles through the tag dictionary written by the generator, exactly as an engineer would
use the site instrument index.
"""
from __future__ import annotations

ROLE_SPECS = [
    # role, prefix, number, unit, description
    ("feed_flow", "FI", 2, "t/h", "Column feed flow"),
    ("feed_temp", "TI", 3, "degC", "Feed temperature downstream of preheater"),
    ("reflux_flow", "FIC", 4, "t/h", "Reflux flow"),
    ("distillate_flow", "FI", 5, "t/h", "Overhead product flow"),
    ("bottoms_flow", "FI", 6, "t/h", "Bottoms product flow"),
    ("steam_flow", "FIC", 7, "t/h", "Reboiler heating steam flow"),
    ("steam_sp", "FIC", 7, "t/h", "Reboiler steam flow setpoint"),
    ("steam_valve_op", "FV", 7, "%", "Reboiler steam control valve output"),
    ("top_pressure", "PIC", 8, "barg", "Column overhead pressure"),
    ("pressure_sp", "PIC", 8, "barg", "Column overhead pressure setpoint"),
    ("dp_rect", "PDI", 9, "mbar", "Differential pressure across the rectifying section"),
    ("dp_strip", "PDI", 10, "mbar", "Differential pressure across the stripping section"),
    ("top_temp", "TI", 11, "degC", "Overhead vapour temperature"),
    ("sens_temp", "TIC", 12, "degC", "Sensitive tray temperature (boilup control)"),
    ("sens_temp_sp", "TIC", 12, "degC", "Sensitive tray temperature setpoint"),
    ("bottom_temp", "TI", 13, "degC", "Column bottom temperature"),
    ("sump_level", "LIC", 14, "%", "Column sump level"),
    ("top_analyzer", "AI", 15, "mol%", "Overhead heavy key impurity analyzer"),
    ("bottom_analyzer", "AI", 16, "mol%", "Bottoms light key impurity analyzer"),
    ("chest_pressure", "PI", 17, "barg", "Reboiler steam chest pressure"),
]

SUFFIX = {"steam_sp": "_SP", "pressure_sp": "_SP", "sens_temp_sp": "_SP", "steam_valve_op": "_OP"}


def unit_of(column: str) -> int:
    return int(column[1])


def tag_for(column: str, role: str) -> str:
    u = unit_of(column)
    for r, prefix, num, _unit, _desc in ROLE_SPECS:
        if r == role:
            return f"{prefix}{u}{num:02d}{SUFFIX.get(role, '')}"
    raise KeyError(role)


def tag_dictionary(column: str, column_name: str) -> list[dict]:
    rows = []
    for role, prefix, num, unit, desc in ROLE_SPECS:
        rows.append({
            "column": column,
            "tag": tag_for(column, role),
            "role": role,
            "unit": unit,
            "description": f"{column} {column_name}: {desc}",
        })
    return rows


def equipment(column: str) -> dict[str, str]:
    u = unit_of(column)
    return {
        "column": f"C{u}01",
        "reboiler": f"E{u}01",
        "condenser": f"E{u}02",
        "preheater": f"E{u}03",
        "steam_valve": f"FV{u}07",
        "steam_controller": f"FIC{u}07",
        "pressure_controller": f"PIC{u}08",
        "temp_controller": f"TIC{u}12",
        "top_analyzer": f"AI{u}15",
        "bottom_analyzer": f"AI{u}16",
        "reflux_drum": f"V{u}01",
        "reflux_pump_a": f"P{u}01A",
        "reflux_pump_b": f"P{u}01B",
        "bottoms_pump_a": f"P{u}02A",
        "bottoms_pump_b": f"P{u}02B",
        "relief_valve": f"PSV{u}01",
    }


EQUIPMENT_DESCRIPTIONS = {
    "column": "distillation column",
    "reboiler": "kettle reboiler",
    "condenser": "overhead condenser",
    "preheater": "feed preheater",
    "steam_valve": "reboiler steam control valve",
    "steam_controller": "reboiler steam flow controller",
    "pressure_controller": "overhead pressure controller",
    "temp_controller": "sensitive tray temperature controller",
    "top_analyzer": "overhead online analyzer",
    "bottom_analyzer": "bottoms online analyzer",
    "reflux_drum": "reflux drum",
    "reflux_pump_a": "reflux pump A",
    "reflux_pump_b": "reflux pump B",
    "bottoms_pump_a": "bottoms pump A",
    "bottoms_pump_b": "bottoms pump B",
    "relief_valve": "overhead pressure safety valve",
}
