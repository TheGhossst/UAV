from src.status_sync import (
    PAPER,
    _paper,
    count_tests,
    patch_sections,
    ranking_line,
    summarize_raw,
    table_compare,
    table_vs_axis,
)


def test_paper_lookup():
    assert _paper("fig6_j5", "sca") == "~8.8"
    assert _paper("fig6_j5", "pso") == "n/a"
    assert PAPER["fig7_i32"]["td3"] == 11.6


def test_summarize_and_ranking():
    rows = [
        {"method": "sca", "sum_rate": "200", "feasible": "1", "qos": "0", "runtime": "0.1", "aodt_mean": "2.0"},
        {"method": "sca", "sum_rate": "400", "feasible": "1", "qos": "0", "runtime": "0.2", "aodt_mean": "2.2"},
        {"method": "random", "sum_rate": "100", "feasible": "False", "qos": "1", "runtime": "0.0", "aodt_mean": "3.0"},
    ]
    st = summarize_raw(rows)
    assert st["sca"]["sum_rate"] == 300.0
    assert st["random"]["feasible"] == 0.0
    line = ranking_line(st)
    assert "SCA" in line and "Random" in line
    assert line.index("SCA") < line.index("Random")


def test_patch_sections_roundtrip():
    text = "before\n<!-- AUTO:foo -->\nold\n<!-- /AUTO:foo -->\nafter\n"
    out = patch_sections(text, {"foo": "new table\n"})
    assert "new table" in out
    assert "old" not in out
    assert out.startswith("before")


def test_compare_table_includes_paper_column():
    st = {
        "sca": {"sum_rate": 7.143e6, "std": 1.0, "feasible": 1.0, "qos": 0.0, "runtime": 0.04, "aodt": 2.375},
        "pso": {"sum_rate": 7.588e6, "std": 1.0, "feasible": 1.0, "qos": 0.0, "runtime": 0.5, "aodt": 2.42},
    }
    lines = table_compare(st, "fig6_j3", "placement", "Paper Fig. 6 at J=3 (Mbps)")
    body = "\n".join(lines)
    assert "Paper Fig. 6" in body
    assert "~7.1" in body
    assert "n/a" in body  # PSO


def test_axis_table_fills_paper_only_at_quoted_x():
    rows = [
        {"method": "sca", "n_uav": "3", "sum_rate_mean": "7142784"},
        {"method": "sca", "n_uav": "5", "sum_rate_mean": "8025287"},
        {"method": "td3", "n_uav": "3", "sum_rate_mean": "6511979"},
        {"method": "td3", "n_uav": "5", "sum_rate_mean": "7665014"},
    ]
    lines = table_vs_axis(rows, "n_uav", "J", {5.0: "fig6_j5"})
    j5 = [ln for ln in lines if ln.startswith("| 5 |")][0]
    assert "~8.8" in j5
    j3 = [ln for ln in lines if ln.startswith("| 3 |")][0]
    assert "~8.8" not in j3


def test_count_tests_positive():
    assert count_tests() >= 54
