import numpy as np

from weir_whisper.simulator.hydraulics import (derive_geometry, flood_fraction, section_state, weep_fraction,
                                               weir_crest_mm)


def test_design_margins_match_the_design_basis(settings):
    for tag, col in settings.plant["columns"].items():
        g = derive_geometry(tag, col)
        assert abs(min(g.weep_margin_rect_design, g.weep_margin_strip_design) - col["design_weep_margin"]) < 1e-6
        assert abs(max(g.flood_rect_design, g.flood_strip_design) - col["design_flood"]) < 1e-6
        assert 0.02 < g.hole_area_fraction < 0.15


def test_weep_margin_rises_with_vapour_and_falls_with_pressure(settings):
    g = derive_geometry("C201", settings.column("C201"))
    v = np.array([0.5, 0.75, 1.0]) * g.v_strip_design
    m = section_state(v, g.l_strip_design, g.rho_vapour_design, g)[2]
    assert np.all(np.diff(m) > 0)
    m_hi_p = section_state(g.v_strip_design, g.l_strip_design, g.rho_vapour_design * 1.3, g)[2]
    assert m_hi_p < m[-1]


def test_damaged_trays_weep_at_design_load(settings):
    g = derive_geometry("C301", settings.column("C301"))
    m = section_state(g.v_strip_design, g.l_strip_design, g.rho_vapour_design, g, hole_area_m2=g.hole_area_m2 * 2.0)[2]
    assert m < 1.0
    assert weep_fraction(m) > 0.03


def test_weep_fraction_and_crest_behave():
    assert weep_fraction(1.2) == 0.0
    assert 0 < weep_fraction(0.9) < weep_fraction(0.7) <= 0.6
    assert weir_crest_mm(20000, 500, 1.4) < weir_crest_mm(40000, 500, 1.4)
    assert flood_fraction(10000, 20, 500, 2.0, 0.1) < flood_fraction(20000, 20, 500, 2.0, 0.1)
