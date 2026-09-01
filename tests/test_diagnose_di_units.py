"""Diagnostic D_i units patch must not leak into production."""

from dataclasses import replace

import numpy as np

from scripts.diagnose_di_units import di_bits_per_byte
from src.aodt import delay_rate_floors, upload_times
from src.compute import service_rate
from src.config import DEFAULT
from src.repair import associated_rate_floors
from src.scenario import generate_scenario


def test_production_upload_still_uses_eight():
    cfg = DEFAULT.with_compute(task_size_bytes=1000.0)
    association = np.array([[1.0]])
    processing = np.array([[1.0]])
    rates = np.array([[8000.0]])
    np.testing.assert_allclose(upload_times(association, processing, rates, cfg), [1.0])


def test_paper_units_are_eight_times_faster_and_restore():
    cfg = DEFAULT.with_compute(task_size_bytes=1000.0)
    association = np.array([[1.0]])
    processing = np.array([[1.0]])
    rates = np.array([[8000.0]])
    with di_bits_per_byte(1.0):
        import src.aodt as aodt_mod

        d = aodt_mod.upload_times(association, processing, rates, cfg)
        np.testing.assert_allclose(d, [0.125])
    np.testing.assert_allclose(upload_times(association, processing, rates, cfg), [1.0])
    import src.aodt as aodt_mod

    np.testing.assert_allclose(aodt_mod.upload_times(association, processing, rates, cfg), [1.0])


def test_patch_reaches_repair_bandwidth_floors():
    s = generate_scenario(
        100,
        replace(
            DEFAULT.with_compute(task_size_bytes=12000.0, task_cycles=3.75e6),
            aodt_threshold=0.8,
        ),
    )
    xy = np.array([[120.0, 130.0], [380.0, 200.0], [250.0, 400.0]])
    from src.repair import nearest_association, process_consistent_processing

    a = nearest_association(s.iot_xy, xy)
    proc = process_consistent_processing(s, a)
    mu = service_rate(s.cfg)
    prod = delay_rate_floors(s, a, proc, mu)
    with di_bits_per_byte(1.0):
        import src.aodt as aodt_mod
        import src.repair as repair_mod

        hyp = aodt_mod.delay_rate_floors(s, a, proc, mu)
        hyp_repair = repair_mod.delay_rate_floors(s, a, proc, mu)
        np.testing.assert_allclose(hyp, hyp_repair)
        finite = (prod < 1e17) & (hyp < 1e17)
        if np.any(finite):
            np.testing.assert_allclose(hyp[finite] * 8.0, prod[finite], rtol=1e-9)
        floors_hyp = associated_rate_floors(s, a, proc)
    restored = delay_rate_floors(s, a, proc, mu)
    np.testing.assert_allclose(restored, prod)
    floors_prod = associated_rate_floors(s, a, proc)
    assert float(floors_prod.mean()) >= float(floors_hyp.mean()) - 1e-9
    np.testing.assert_allclose(associated_rate_floors(s, a, proc), floors_prod)
