import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "fifo_storage_control", Path(__file__).parents[1] / "scripts/run_zcu104_fifo_storage_control.py"
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


@pytest.fixture
def synthesis_report(tmp_path):
    path = tmp_path / "synthesis.rpt"
    path.write_text("""| CLB LUTs* | 112,607 | 0 | 0 | 230400 | 48.87 |
| LUT as Logic | 77,647 | 0 | 0 | 230400 | 33.70 |
| LUT as Memory | 34,960 | 0 | 0 | 101760 | 34.36 |
| CLB Registers | 142,050 | 0 | 0 | 460800 | 30.83 |
| Block RAM Tile | 312 | 0 | 0 | 312 | 100.00 |
| DSPs | 278 | 0 | 0 | 1728 | 16.09 |
""")
    return path


def test_synthesis_annotation_and_absent_placement_statistics(synthesis_report):
    metrics = control.resource_metrics(synthesis_report)
    assert metrics["lut"] == 112607
    assert metrics["lut_as_memory"] == 34960
    assert metrics["clb"] is None
    assert control.resource_violations(metrics) == []


def test_missing_required_hardware_metric_still_rejects(synthesis_report):
    synthesis_report.write_text(synthesis_report.read_text().replace("| DSPs |", "| Other |"))
    with pytest.raises(ValueError, match="DSPs"):
        control.resource_metrics(synthesis_report)


def test_half_bram_over_capacity_is_not_rounded_down(synthesis_report):
    text = synthesis_report.read_text().replace("| Block RAM Tile | 312 |", "| Block RAM Tile | 312.5 |")
    synthesis_report.write_text(text)
    assert control.resource_violations(control.resource_metrics(synthesis_report)) == ["bram36>312"]


def test_placed_report_retains_clb_statistics(synthesis_report):
    synthesis_report.write_text(synthesis_report.read_text().replace("CLB LUTs*", "CLB LUTs")
                                + "| CLB | 28475 | 0 | 0 | 28800 | 98.87 |\n")
    assert control.resource_metrics(synthesis_report)["clb"] == 28475
