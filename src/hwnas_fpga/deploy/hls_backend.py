"""HLS/Vivado backend scaffolding and report ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Optional

from hwnas_fpga.deploy.report_parser import parse_backend_report


@dataclass(frozen=True)
class HLSProjectConfig:
    project_name: str = "hwnas_fpga"
    top_name: str = "hwnas_top"
    backend: str = "vivado_hls"
    board: Optional[str] = None
    fpga_part: Optional[str] = None
    clock_mhz: int = 200


def create_hls_project_stub(
    onnx_path: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[HLSProjectConfig] = None,
) -> dict[str, Any]:
    cfg = config or HLSProjectConfig()
    onnx_file = Path(onnx_path).expanduser().resolve()
    project_dir = Path(output_dir).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "onnx_path": str(onnx_file),
        "config": asdict(cfg),
        "generated_files": {
            "manifest": str(project_dir / "hls_project.json"),
            "script": str(project_dir / "run_hls.tcl"),
        },
    }

    (project_dir / "hls_project.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tcl_script = "\n".join(
        [
            f"open_project {cfg.project_name}",
            f"set_top {cfg.top_name}",
            f"# Import ONNX converted sources for {onnx_file}",
            f"open_solution solution1 -flow_target {cfg.backend}",
            f"create_clock -period {1000.0 / cfg.clock_mhz:.3f} -name default",
            "# add_files <generated_hls_sources.cpp>",
            "csynth_design",
            "export_design -format ip_catalog",
        ]
    )
    (project_dir / "run_hls.tcl").write_text(tcl_script + "\n", encoding="utf-8")
    (project_dir / "README.txt").write_text(
        "Place generated HLS C/C++ sources in this directory, then run run_hls.tcl.\n",
        encoding="utf-8",
    )
    return manifest


def attach_hls_report(
    project_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_dir).expanduser().resolve()
    metrics = parse_backend_report(report_path)
    output_path = project_root / "hls_report.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
