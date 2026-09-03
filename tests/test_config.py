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
