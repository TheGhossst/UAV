"""S_i and L are external; CLI conversion is labeled."""

from __future__ import annotations

from uavdt.config import EXTERNAL_TASK_CYCLES, EXTERNAL_TASK_SIZE_BITS, SimConfig
from uavdt.experiments.cli import build_parser, _cfg_from_args


def test_external_defaults_are_not_hidden():
    cfg = SimConfig()
    assert cfg.task_size_bits == EXTERNAL_TASK_SIZE_BITS
    assert cfg.task_cycles == EXTERNAL_TASK_CYCLES


def test_task_size_bytes_is_times_eight():
    parser = build_parser()
    args = parser.parse_args(
        ["evaluate", "--task-size-bytes", "2000", "--bandwidth", "20000"]
    )
    cfg = _cfg_from_args(args)
    assert cfg.task_size_bits == 16_000.0
    assert cfg.max_bw_share is None


def test_max_bw_share_cli():
    parser = build_parser()
    args = parser.parse_args(["sca", "--max-bw-share", "0.25", "--bandwidth-preset", "8.8mhz"])
    cfg = _cfg_from_args(args)
    assert cfg.max_bw_share == 0.25
    assert abs(cfg.link_bandwidth_cap_hz - 0.25 * 8_800_000.0) < 1e-6
    assert cfg.download_time_s == 0.0
    sca_parser = None
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and "sca" in choices:
            sca_parser = choices["sca"]
            break
    assert sca_parser is not None
    help_text = sca_parser.format_help()
    assert "EXTERNAL PARAMETER" in help_text
    assert "Problem (P)" in help_text
