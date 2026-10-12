import numpy as np
import pandas as pd

from weir_whisper.simulator.hydraulics import derive_geometry
from weir_whisper.simulator.plant import simulate_column
from weir_whisper.simulator.scenarios import Scenario, build_exogenous, schedule_scenarios


def _index(days=6):
    return pd.date_range("2026-01-01", periods=days * 288, freq="5min")


def _scenario(cause, start_day, days, params):
    idx = _index()
    s = idx[0] + pd.Timedelta(days=start_day)
    return Scenario("SC001", "C201", cause, s, s + pd.Timedelta(days=days), start_day * 288, (start_day + days) * 288, "FAM", params)


def test_normal_operation_does_not_weep(settings):
    col = settings.column("C201")
    g = derive_geometry("C201", col)
    idx = _index()
    ex = build_exogenous("C201", col, [], len(idx), 5, idx, np.random.default_rng(1))
    out = simulate_column(col, g, ex, 7)
    assert np.maximum(out["true_wf_rect"], out["true_wf_strip"]).max() < 0.03
    assert out["steam_valve_op"].mean() < 90


def test_turndown_causes_weeping_and_suppression_removes_its_effect(settings):
    col = settings.column("C201")
    g = derive_geometry("C201", col)
    idx = _index()
    sc = _scenario("turndown", 2, 2, {"level": 0.48, "ramp_hours": 1.0, "reason": "test"})
    ex = build_exogenous("C201", col, [sc], len(idx), 5, idx, np.random.default_rng(1))
    out = simulate_column(col, g, ex, 7)
    wf = np.maximum(out["true_wf_rect"], out["true_wf_strip"])
    assert wf[2 * 288 + 60:4 * 288].mean() > 0.03
    sup = simulate_column(col, g, ex, 7, suppress_weep=True)
    assert np.allclose(sup["true_eff_rect"][2 * 288 + 60:4 * 288], col["murphree_eff"], atol=1e-9)


def test_stiction_makes_steam_cycle(settings):
    col = settings.column("C201")
    g = derive_geometry("C201", col)
    idx = _index()
    sc = _scenario("steam_valve_stiction", 1, 4, {"deadband_pct": 20.0, "slip_pct": 5.0})
    ex = build_exogenous("C201", col, [sc], len(idx), 5, idx, np.random.default_rng(1))
    out = simulate_column(col, g, ex, 7)
    before = out["steam_flow"][:288].std() / out["steam_flow"][:288].mean()
    during = out["steam_flow"][3 * 288:4 * 288].std() / out["steam_flow"][3 * 288:4 * 288].mean()
    assert during > 4 * before


def test_scheduler_respects_gaps(settings):
    geoms = {c: derive_geometry(c, settings.column(c)) for c in settings.columns}
    idx = pd.date_range(settings["simulation"]["start"], settings["simulation"]["end"], freq="5min")
    scs = schedule_scenarios(settings, geoms, idx, np.random.default_rng(3))
    gap = pd.Timedelta(days=settings["simulation"]["min_gap_days"])
    for c in settings.columns:
        mine = sorted([s for s in scs if s.column == c], key=lambda s: s.start)
        for a, b in zip(mine, mine[1:]):
            assert b.start >= a.end + gap
        assert all(s.start >= idx[0] + pd.Timedelta(days=settings["simulation"]["reference_days"]) for s in mine)
