#!/usr/bin/env python3
"""Build and measure the arch_84 sonar classifier full-network board harness.

This script owns only the sonar_classifier_end_to_end_board result directory.
It does not update deployable LUT JSON files or legacy board measurement CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
BUILDER_ROOT = ROOT / "hls_lut_builder"
BOARD_HARNESS_DIR = BUILDER_ROOT / "board_harness"
E2E_DIR = BOARD_HARNESS_DIR / "results" / "sonar_classifier_end_to_end_board"
SPEC_PATH = E2E_DIR / "sonar_classifier_e2e_harness_spec.json"
TARGET_NAME = "sonar_classifier_top8_03_arch_84_e2e"
MEASUREMENT_ROOT = E2E_DIR / "real_measurement" / TARGET_NAME
PROJECT_ROOT = MEASUREMENT_ROOT / "harness_project"
VARIANTS_ROOT = MEASUREMENT_ROOT / "implementation_variants"
PROMOTION_MANIFEST_PATH = MEASUREMENT_ROOT / "official_promotion_manifest.json"

sys.path.insert(0, str(SCRIPT_DIR))

from batch_measure_ops import DEFAULT_BOARD_CONFIG, DEFAULT_VIVADO  # noqa: E402
import current84_fullcombo_board_validation as fullcombo  # noqa: E402
import generate_harness as gh  # noqa: E402


STATUS_FIELDS = [
    "target_name",
    "status",
    "bitstream_status",
    "uart_status_code",
    "e2e_cycles",
    "e2e_latency_ms",
    "axis_word_count",
    "classifier_class_count",
    "argmax",
    "wns_ns",
    "timing_clean",
    "measurement_json_path",
    "bitstream_path",
    "timing_report_path",
    "blocker_count",
    "blocker_categories",
    "notes",
]

MEASUREMENT_FIELDS = [
    "target_name",
    "status",
    "uart_status_code",
    "cycles",
    "latency_ms",
    "axis_word_count",
    "classifier_class_count",
    "checksum_argmax_packed",
    "checksum24",
    "argmax",
    "wns_ns",
    "timing_clean",
    "measurement_json_path",
    "bitstream_path",
    "timing_report_path",
]

BLOCKER_FIELDS = ["scope", "case_name", "blocker_category", "severity", "evidence_path", "evidence_detail", "required_resolution"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sonar classifier arch_84 full-network board validation")
    parser.add_argument("mode", choices=("refresh", "generate", "build", "measure", "promote", "all"))
    parser.add_argument("--variant-name", default="")
    parser.add_argument("--expected-bitstream-sha256", default="")
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG))
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--hw-server-url", default="TCP:localhost:3121")
    parser.add_argument("--hw-device-pattern", default="*xc7k325t*")
    parser.add_argument("--hw-target-index", type=int, default=0)
    parser.add_argument("--program-retries", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--uart-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--post-program-delay-seconds", type=float, default=1.0)
    parser.add_argument("--harness-fifo-depth", type=int, default=512)
    parser.add_argument("--harness-fifo-style", choices=("lut", "lut_registered", "bram"), default="bram")
    parser.add_argument("--watchdog-cycles", type=int, default=200_000_000)
    parser.add_argument("--timing-variant", default="performance_explore")
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=240)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-measure", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_variant_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not value:
        raise ValueError("--variant-name is required for promote mode")
    return value


def variant_target_name(variant_name: str) -> str:
    return f"{TARGET_NAME}__{variant_name}"


def variant_root(variant_name: str) -> Path:
    return VARIANTS_ROOT / variant_name


def variant_measurement_json_path(variant_name: str) -> Path:
    return variant_root(variant_name) / "measurements" / f"{variant_target_name(variant_name)}.measurement.json"


def canonical_bitstream_path() -> Path:
    return PROJECT_ROOT / "bitstream" / f"{TARGET_NAME}.bit"


def _replace_path_prefix(value: Any, source_root: Path, target_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, source_root, target_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_path_prefix(item, source_root, target_root) for item in value]
    if isinstance(value, str):
        text = value
        replacements = (
            (str(source_root), str(target_root)),
            (str(source_root).replace("\\", "/"), str(target_root).replace("\\", "/")),
        )
        for source_text, target_text in replacements:
            text = text.replace(source_text, target_text)
        return text
    return value


def _rewrite_project_text_paths(project_root: Path, source_project_root: Path) -> None:
    for pattern in ("*.tcl", "*.xdc"):
        for path in project_root.rglob(pattern):
            text = path.read_text(encoding="utf-8", errors="replace")
            updated = _replace_path_prefix(text, source_project_root, project_root)
            if updated != text:
                path.write_text(str(updated), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sanitize_role(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not value or value[0].isdigit():
        value = f"r_{value}"
    return value.lower()


def load_spec() -> dict[str, Any]:
    spec = read_json(SPEC_PATH)
    if not spec:
        raise FileNotFoundError(f"Missing e2e harness spec: {SPEC_PATH}")
    if spec.get("status") != "ready_to_generate":
        raise RuntimeError(f"E2E harness spec is not ready_to_generate: {spec.get('status')}")
    return spec


def measured_combo_result(layer: dict[str, Any]) -> dict[str, Any]:
    combo = str(layer["combo_case_name"])
    namespace = str(layer.get("measurement_namespace", ""))
    if namespace == "current84_fullcombo84":
        path = BOARD_HARNESS_DIR / "results" / "v2_current84_fullcombo_board" / "case_results" / f"{combo}.result.json"
    elif namespace == "v2_arch84_e2e_extension":
        path = BOARD_HARNESS_DIR / "results" / "v2_arch84_e2e_extension_board" / "case_results" / f"{combo}.result.json"
    else:
        return {}
    return read_json(path)


def component_case_names_for_layer(layer: dict[str, Any]) -> list[str]:
    result = measured_combo_result(layer)
    result_names = str(result.get("component_case_names") or "")
    if result_names:
        return [item for item in result_names.split("|") if item]
    return [item for item in str(layer["component_case_names"]).split("|") if item]


def flattened_components(spec: dict[str, Any]) -> list[dict[str, str]]:
    components: list[dict[str, str]] = []
    for layer in spec.get("topology", []):
        layer_index = int(layer["layer_index"])
        layer_role = sanitize_role(str(layer["layer_role"]))
        names = component_case_names_for_layer(layer)
        roles = [item for item in str(layer["component_roles"]).split("|") if item]
        if str(layer["decomposition"]) == "direct":
            roles = [str(layer.get("component_roles") or layer.get("op") or "direct")]
        if len(names) != len(roles):
            raise ValueError(f"Layer {layer_index} component names/roles mismatch: {names} {roles}")
        for role, case_name in zip(roles, names):
            role_name = sanitize_role(f"l{layer_index}_{layer_role}_{role}")
            components.append(
                {
                    "layer_index": str(layer_index),
                    "layer_role": layer_role,
                    "component_role": sanitize_role(role),
                    "role": role_name,
                    "case_name": case_name,
                    "combo_case_name": str(layer["combo_case_name"]),
                    "namespace": str(layer["measurement_namespace"]),
                }
            )
    return components


def axis_width(spec: dict[str, Any], prefix: str) -> tuple[int, int]:
    width = int(spec["top_ports"][f"{prefix}_stream_TDATA"])
    keep = int(spec["top_ports"].get(f"{prefix}_stream_TKEEP", max(1, width // 8)))
    return width, keep


def adapter_fifo_text(
    *,
    index: int,
    src_prefix: str,
    adapter_prefix: str,
    fifo_prefix: str,
    src_width: int,
    src_keep: int,
    dst_width: int,
    dst_keep: int,
    fifo_depth: int,
) -> str:
    if max(src_width, dst_width) % min(src_width, dst_width) != 0:
        raise ValueError(f"AXIS width ratio must be integral at boundary {index}: {src_width}->{dst_width}")
    return f"""
axis_width_adapter #(
    .S_DATA_WIDTH({src_width}),
    .M_DATA_WIDTH({dst_width}),
    .S_KEEP_WIDTH({src_keep}),
    .M_KEEP_WIDTH({dst_keep})
) u_e2e_repack_{index:02d} (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata({src_prefix}_tdata), .s_tkeep({src_prefix}_tkeep), .s_tstrb({src_prefix}_tstrb), .s_tlast({src_prefix}_tlast),
    .s_tvalid({src_prefix}_tvalid), .s_tready({src_prefix}_tready),
    .m_tdata({adapter_prefix}_tdata), .m_tkeep({adapter_prefix}_tkeep), .m_tstrb({adapter_prefix}_tstrb), .m_tlast({adapter_prefix}_tlast),
    .m_tvalid({adapter_prefix}_tvalid), .m_tready({adapter_prefix}_tready)
);

axis_simple_fifo #(.DATA_WIDTH({dst_width}), .KEEP_WIDTH({dst_keep}), .DEPTH({fifo_depth})) u_e2e_fifo_{index:02d} (
    .clk(clk_main), .rst_n(rst_n),
    .s_tdata({adapter_prefix}_tdata), .s_tkeep({adapter_prefix}_tkeep), .s_tstrb({adapter_prefix}_tstrb), .s_tlast({adapter_prefix}_tlast),
    .s_tvalid({adapter_prefix}_tvalid), .s_tready({adapter_prefix}_tready),
    .m_tdata({fifo_prefix}_tdata), .m_tkeep({fifo_prefix}_tkeep), .m_tstrb({fifo_prefix}_tstrb), .m_tlast({fifo_prefix}_tlast),
    .m_tvalid({fifo_prefix}_tvalid), .m_tready({fifo_prefix}_tready)
);
""".strip()


def input_mem_mapping_text(mapping: dict[str, str]) -> str:
    return ",\n    ".join(f".MEM_FILE_{idx:02d}({mapping[f'INPUT_MEM_FILE_{idx:02d}']})" for idx in range(32))


def write_full_network_top(
    project_root: Path,
    specs: list[dict[str, Any]],
    board_config: Path,
    fifo_depth: int,
    fifo_style: str,
    watchdog_cycles: int,
    argmax_pipeline: bool = False,
    completion_mode: str = "axil_and_sink",
) -> None:
    if completion_mode not in {"axil_and_sink", "sink_and_argmax"}:
        raise ValueError(f"unsupported e2e completion mode: {completion_mode}")
    board_cfg = fullcombo.load_yaml(board_config)
    defaults_cfg = fullcombo.load_yaml(BOARD_HARNESS_DIR / "configs" / "harness_defaults.yaml")
    data_dir = project_root / "data"
    rtl_dir = project_root / "rtl"
    data_dir.mkdir(parents=True, exist_ok=True)
    rtl_dir.mkdir(parents=True, exist_ok=True)
    gh.copy_tree_files(BOARD_HARNESS_DIR / "modules", rtl_dir)
    (rtl_dir / "axis_simple_fifo.v").write_text(fullcombo.axis_fifo_module_text(fifo_style), encoding="utf-8")
    (rtl_dir / "axis_width_adapter.v").write_text(fullcombo.axis_width_adapter_module_text(), encoding="utf-8")

    first = specs[0]
    last = specs[-1]
    input_width, input_keep = axis_width(first, "input")
    output_width, output_keep = axis_width(last, "output")
    input_word_count = int(first["case_meta"]["input_word_count"])
    axis_output_word_count = int(last["case_meta"]["output_word_count"])
    classifier_class_count = int(last["constants"].get("OUT_CH", max(1, output_width // 8)))

    input_mem_paths = gh.generate_stream_mem_files(data_dir, "input_stream", input_word_count, input_width)
    input_word_mem = gh.generate_stream_word_mem_file(data_dir, "input_stream_words", input_word_count, input_width)
    input_mem_mapping = {f"INPUT_MEM_FILE_{lane_idx:02d}": '""' for lane_idx in range(32)}
    for lane_idx, mem_path in enumerate(input_mem_paths):
        input_mem_mapping[f"INPUT_MEM_FILE_{lane_idx:02d}"] = f'"../data/{mem_path.name}"'

    axis_decls = [fullcombo.axis_wire_decl("input", input_width, input_keep)]
    chain_blocks: list[str] = []
    kernel_instances: list[str] = []
    current_prefix = "input"
    for idx, spec in enumerate(specs):
        is_last = idx == len(specs) - 1
        out_prefix = "output" if is_last else f"op{idx:02d}_out"
        out_width, out_keep = axis_width(spec, "output")
        axis_decls.append(fullcombo.axis_wire_decl(out_prefix, out_width, out_keep))
        kernel_instances.append(fullcombo.build_kernel_instance_for_spec(spec, current_prefix, out_prefix))
        if not is_last:
            next_width, next_keep = axis_width(specs[idx + 1], "input")
            adapter_prefix = f"adapt{idx:02d}_out"
            fifo_prefix = f"fifo{idx:02d}_out"
            axis_decls.append(fullcombo.axis_wire_decl(adapter_prefix, next_width, next_keep))
            axis_decls.append(fullcombo.axis_wire_decl(fifo_prefix, next_width, next_keep))
            chain_blocks.append(
                adapter_fifo_text(
                    index=idx,
                    src_prefix=out_prefix,
                    adapter_prefix=adapter_prefix,
                    fifo_prefix=fifo_prefix,
                    src_width=out_width,
                    src_keep=out_keep,
                    dst_width=next_width,
                    dst_keep=next_keep,
                    fifo_depth=fifo_depth,
                )
            )
            current_prefix = fifo_prefix

    param_logic = "\n\n".join(fullcombo.build_param_logic_for_spec(spec, data_dir) for spec in specs)
    axil_blocks = "\n\n".join(fullcombo.axil_wires_and_ctrl(spec) for spec in specs)
    axil_done_expr = " && ".join(f"{spec['role']}_axil_done" for spec in specs)
    axil_error_expr = " || ".join(f"{spec['role']}_axil_error" for spec in specs)
    status_led_width = int(board_cfg.get("status_led", {}).get("width", 2))

    argmax_logic = ""
    packed_checksum_expr = "output_checksum"
    argmax_done_expr = "1'b1"
    if output_width >= classifier_class_count * 8:
        packed_checksum_expr = "{output_argmax, output_checksum[23:0]}"
        if argmax_pipeline:
            if classifier_class_count != 8:
                raise ValueError("argmax_pipeline currently supports exactly 8 classifier classes")
            argmax_done_expr = "argmax_done"
            argmax_logic = """
wire argmax_sample_fire = output_tvalid && output_tready && output_word_count == 32'd0;
reg [7:0] output_argmax = 8'd0;
reg argmax_done = 1'b0;

reg argmax_s1_valid = 1'b0;
reg signed [7:0] argmax_s1_value0 = 8'sd0;
reg signed [7:0] argmax_s1_value1 = 8'sd0;
reg signed [7:0] argmax_s1_value2 = 8'sd0;
reg signed [7:0] argmax_s1_value3 = 8'sd0;
reg [7:0] argmax_s1_index0 = 8'd0;
reg [7:0] argmax_s1_index1 = 8'd0;
reg [7:0] argmax_s1_index2 = 8'd0;
reg [7:0] argmax_s1_index3 = 8'd0;

reg argmax_s2_valid = 1'b0;
reg signed [7:0] argmax_s2_value0 = 8'sd0;
reg signed [7:0] argmax_s2_value1 = 8'sd0;
reg [7:0] argmax_s2_index0 = 8'd0;
reg [7:0] argmax_s2_index1 = 8'd0;

reg argmax_s3_valid = 1'b0;

always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        output_argmax <= 8'd0;
        argmax_done <= 1'b0;
        argmax_s1_valid <= 1'b0;
        argmax_s2_valid <= 1'b0;
        argmax_s3_valid <= 1'b0;
    end else if (start_pulse) begin
        output_argmax <= 8'd0;
        argmax_done <= 1'b0;
        argmax_s1_valid <= 1'b0;
        argmax_s2_valid <= 1'b0;
        argmax_s3_valid <= 1'b0;
    end else begin
        argmax_s1_valid <= argmax_sample_fire;
        if (argmax_sample_fire) begin
            if ($signed(output_tdata[8 +: 8]) > $signed(output_tdata[0 +: 8])) begin
                argmax_s1_value0 <= $signed(output_tdata[8 +: 8]);
                argmax_s1_index0 <= 8'd1;
            end else begin
                argmax_s1_value0 <= $signed(output_tdata[0 +: 8]);
                argmax_s1_index0 <= 8'd0;
            end
            if ($signed(output_tdata[24 +: 8]) > $signed(output_tdata[16 +: 8])) begin
                argmax_s1_value1 <= $signed(output_tdata[24 +: 8]);
                argmax_s1_index1 <= 8'd3;
            end else begin
                argmax_s1_value1 <= $signed(output_tdata[16 +: 8]);
                argmax_s1_index1 <= 8'd2;
            end
            if ($signed(output_tdata[40 +: 8]) > $signed(output_tdata[32 +: 8])) begin
                argmax_s1_value2 <= $signed(output_tdata[40 +: 8]);
                argmax_s1_index2 <= 8'd5;
            end else begin
                argmax_s1_value2 <= $signed(output_tdata[32 +: 8]);
                argmax_s1_index2 <= 8'd4;
            end
            if ($signed(output_tdata[56 +: 8]) > $signed(output_tdata[48 +: 8])) begin
                argmax_s1_value3 <= $signed(output_tdata[56 +: 8]);
                argmax_s1_index3 <= 8'd7;
            end else begin
                argmax_s1_value3 <= $signed(output_tdata[48 +: 8]);
                argmax_s1_index3 <= 8'd6;
            end
        end

        argmax_s2_valid <= argmax_s1_valid;
        if (argmax_s1_valid) begin
            if (argmax_s1_value1 > argmax_s1_value0) begin
                argmax_s2_value0 <= argmax_s1_value1;
                argmax_s2_index0 <= argmax_s1_index1;
            end else begin
                argmax_s2_value0 <= argmax_s1_value0;
                argmax_s2_index0 <= argmax_s1_index0;
            end
            if (argmax_s1_value3 > argmax_s1_value2) begin
                argmax_s2_value1 <= argmax_s1_value3;
                argmax_s2_index1 <= argmax_s1_index3;
            end else begin
                argmax_s2_value1 <= argmax_s1_value2;
                argmax_s2_index1 <= argmax_s1_index2;
            end
        end

        argmax_s3_valid <= argmax_s2_valid;
        if (argmax_s2_valid) begin
            output_argmax <= (argmax_s2_value1 > argmax_s2_value0) ? argmax_s2_index1 : argmax_s2_index0;
        end
        if (argmax_s3_valid) begin
            argmax_done <= 1'b1;
        end
    end
end
"""
        else:
            argmax_logic = f"""
reg [7:0] output_argmax = 8'd0;
reg signed [7:0] argmax_value_next;
reg [7:0] argmax_next;
integer arg_lane;
always @(*) begin
    argmax_next = 8'd0;
    argmax_value_next = $signed(output_tdata[7:0]);
    for (arg_lane = 1; arg_lane < {classifier_class_count}; arg_lane = arg_lane + 1) begin
        if ($signed(output_tdata[arg_lane*8 +: 8]) > argmax_value_next) begin
            argmax_value_next = $signed(output_tdata[arg_lane*8 +: 8]);
            argmax_next = arg_lane[7:0];
        end
    end
end

always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        output_argmax <= 8'd0;
    end else if (start_pulse) begin
        output_argmax <= 8'd0;
    end else if (output_tvalid && output_tready && output_word_count == 32'd0) begin
        output_argmax <= argmax_next;
    end
end
"""

    if completion_mode == "sink_and_argmax":
        completion_done_expr = f"sink_done && {argmax_done_expr}"
    else:
        completion_done_expr = f"sink_done && all_axil_done && {argmax_done_expr}"

    top = f"""`timescale 1ns / 1ps

module harness_top (
    input  wire sys_clk_p,
    input  wire sys_clk_n,
    input  wire sys_rst_n,
    input  wire uart_rx,
    output wire uart_tx,
    output wire [{status_led_width - 1}:0] status_led
);

localparam integer CLK_FREQ_HZ = {int(board_cfg["clock"]["freq_hz"])};
localparam integer UART_BAUD = {int(board_cfg.get("uart", {}).get("baud", defaults_cfg["uart_baud"]))};
localparam integer BOOT_DELAY_CYCLES = {int(defaults_cfg["boot_delay_cycles"])};
localparam integer STATUS_LED_WIDTH = {status_led_width};
localparam integer INPUT_TDATA_WIDTH = {input_width};
localparam integer INPUT_KEEP_WIDTH = {input_keep};
localparam integer OUTPUT_TDATA_WIDTH = {output_width};
localparam integer OUTPUT_KEEP_WIDTH = {output_keep};
localparam integer INPUT_WORD_COUNT = {input_word_count};
localparam integer AXIS_OUTPUT_WORD_COUNT = {axis_output_word_count};
localparam integer CLASSIFIER_CLASS_COUNT = {classifier_class_count};
localparam integer WATCHDOG_CYCLES = {watchdog_cycles};

wire clk_ibuf;
wire clk_main;
IBUFDS u_sys_clk_ibufds (.I(sys_clk_p), .IB(sys_clk_n), .O(clk_ibuf));
BUFG u_sys_clk_bufg (.I(clk_ibuf), .O(clk_main));

wire rst_n;
reset_sync u_reset_sync (.clk(clk_main), .rst_n_async(sys_rst_n), .rst_n_sync(rst_n));

reg [31:0] boot_counter = 32'd0;
reg boot_started = 1'b0;
wire start_pulse = (boot_counter == BOOT_DELAY_CYCLES - 1) && !boot_started;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        boot_counter <= 32'd0;
        boot_started <= 1'b0;
    end else if (!boot_started) begin
        if (boot_counter == BOOT_DELAY_CYCLES - 1) begin
            boot_started <= 1'b1;
        end else begin
            boot_counter <= boot_counter + 32'd1;
        end
    end
end

{chr(10).join(axis_decls)}

wire source_done;
axis_rom_source #(
    .DATA_WIDTH(INPUT_TDATA_WIDTH),
    .KEEP_WIDTH(INPUT_KEEP_WIDTH),
    .WORD_COUNT(INPUT_WORD_COUNT),
    .WORD_MEM_FILE("../data/{input_word_mem.name}"),
    {input_mem_mapping_text(input_mem_mapping)}
) u_axis_rom_source (
    .clk(clk_main), .rst_n(rst_n), .start(start_pulse),
    .tdata(input_tdata), .tkeep(input_tkeep), .tstrb(input_tstrb), .tlast(input_tlast),
    .tvalid(input_tvalid), .tready(input_tready), .done(source_done)
);

{chr(10).join(chain_blocks)}

wire sink_done;
wire [31:0] output_checksum;
wire [31:0] output_word_count;
axis_checksum_sink #(
    .DATA_WIDTH(OUTPUT_TDATA_WIDTH),
    .KEEP_WIDTH(OUTPUT_KEEP_WIDTH),
    .EXPECTED_WORD_COUNT(AXIS_OUTPUT_WORD_COUNT)
) u_axis_checksum_sink (
    .clk(clk_main), .rst_n(rst_n), .clear(start_pulse),
    .tdata(output_tdata), .tkeep(output_tkeep), .tstrb(output_tstrb), .tlast(output_tlast),
    .tvalid(output_tvalid), .tready(output_tready), .done(sink_done),
    .checksum(output_checksum), .word_count(output_word_count)
);

{argmax_logic}

{axil_blocks}

wire [31:0] measured_cycles;
wire counter_running;
wire counter_done;
wire all_axil_done = {axil_done_expr};
wire any_axil_error = {axil_error_expr};
wire strict_e2e_done = {completion_done_expr};
cycle_counter u_cycle_counter (
    .clk(clk_main), .rst_n(rst_n), .clear(start_pulse), .start(start_pulse), .stop(strict_e2e_done),
    .cycles(measured_cycles), .running(counter_running), .done(counter_done)
);

reg [31:0] watchdog_counter = 32'd0;
reg watchdog_expired = 1'b0;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        watchdog_counter <= 32'd0;
        watchdog_expired <= 1'b0;
    end else if (start_pulse) begin
        watchdog_counter <= 32'd0;
        watchdog_expired <= 1'b0;
    end else if (!strict_e2e_done && !watchdog_expired && boot_started) begin
        if (watchdog_counter >= WATCHDOG_CYCLES) begin
            watchdog_expired <= 1'b1;
        end else begin
            watchdog_counter <= watchdog_counter + 32'd1;
        end
    end
end

reg result_sent = 1'b0;
reg result_start = 1'b0;
always @(posedge clk_main or negedge rst_n) begin
    if (!rst_n) begin
        result_sent <= 1'b0;
        result_start <= 1'b0;
    end else if (start_pulse) begin
        result_sent <= 1'b0;
        result_start <= 1'b0;
    end else begin
        if (result_tx_done) result_sent <= 1'b1;
        result_start <= !result_sent && !result_tx_busy && (strict_e2e_done || watchdog_expired || any_axil_error);
    end
end

wire result_tx_busy;
wire result_tx_done;
wire [7:0] result_status_code = any_axil_error ? 8'd2 : (strict_e2e_done ? 8'd0 : (watchdog_expired ? 8'd3 : 8'd1));
wire [31:0] result_cycles = strict_e2e_done ? measured_cycles : watchdog_counter;
wire [31:0] result_checksum = {packed_checksum_expr};
result_uart_tx #(.CLK_FREQ_HZ(CLK_FREQ_HZ), .BAUD(UART_BAUD)) u_result_uart_tx (
    .clk(clk_main), .rst_n(rst_n), .start(result_start),
    .status_code(result_status_code), .cycles(result_cycles), .checksum(result_checksum),
    .word_count(output_word_count), .tx(uart_tx), .busy(result_tx_busy), .done(result_tx_done)
);

{param_logic}

{chr(10).join(kernel_instances)}

wire [7:0] status_bits;
assign status_bits[0] = counter_running;
assign status_bits[1] = sink_done;
assign status_bits[2] = all_axil_done;
assign status_bits[3] = any_axil_error;
assign status_bits[4] = source_done;
assign status_bits[5] = watchdog_expired;
assign status_bits[6] = result_sent;
assign status_bits[7] = counter_done;

genvar status_idx;
generate
    for (status_idx = 0; status_idx < STATUS_LED_WIDTH; status_idx = status_idx + 1) begin : g_status_led
        assign status_led[status_idx] = status_bits[status_idx];
    end
endgenerate

endmodule
"""
    (rtl_dir / "harness_top.v").write_text(top, encoding="utf-8")


def generate_harness(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec()
    board_config = Path(args.board_config).expanduser().resolve()
    if PROJECT_ROOT.exists() and args.force:
        shutil.rmtree(PROJECT_ROOT)
    (PROJECT_ROOT / "export_rtl").mkdir(parents=True, exist_ok=True)
    component_rows = flattened_components(spec)
    specs: list[dict[str, Any]] = []
    for row in component_rows:
        case_dir = fullcombo.resolve_case_dir(row["case_name"])
        specs.append(fullcombo.build_component_spec(row["role"], row["case_name"], case_dir, PROJECT_ROOT / "export_rtl"))
        specs[-1].update(row)
    write_full_network_top(
        PROJECT_ROOT,
        specs,
        board_config,
        args.harness_fifo_depth,
        args.harness_fifo_style,
        args.watchdog_cycles,
    )
    fullcombo.write_constraints(PROJECT_ROOT, board_config)
    fullcombo.write_build_tcl(PROJECT_ROOT, TARGET_NAME, board_config, recovery_variant=args.timing_variant)
    manifest = {
        "case_name": TARGET_NAME,
        "module_name": "sonar_classifier_e2e_harness",
        "board_config": str(board_config),
        "harness_kind": "single_full_network_latency_only_harness",
        "parameter_mode": "latency_only_deterministic",
        "accuracy_claim": "none",
        "timing_variant": args.timing_variant,
        "harness_fifo_depth": args.harness_fifo_depth,
        "harness_fifo_style": args.harness_fifo_style,
        "generated": {
            "project_root": str(PROJECT_ROOT),
            "rtl_dir": str(PROJECT_ROOT / "rtl"),
            "export_rtl_dir": str(PROJECT_ROOT / "export_rtl"),
            "constraints": str(PROJECT_ROOT / "constraints" / "harness.xdc"),
            "build_tcl": str(PROJECT_ROOT / "scripts" / "build_bitstream.tcl"),
            "bitstream": str(PROJECT_ROOT / "bitstream" / f"{TARGET_NAME}.bit"),
        },
        "streaming": {
            "input_axis_word_count": int(specs[0]["case_meta"]["input_word_count"]),
            "output_axis_word_count": int(specs[-1]["case_meta"]["output_word_count"]),
            "classifier_class_count": int(specs[-1]["constants"].get("OUT_CH", 8)),
            "uart_checksum_field": "packed {argmax[7:0], checksum24[23:0]} for e2e only",
        },
        "layers": spec.get("topology", []),
        "components": [
            {
                "role": item["role"],
                "case_name": item["case_name"],
                "layer_index": item["layer_index"],
                "layer_role": item["layer_role"],
                "renamed_module": item["renamed_module"],
                "input_word_count": item["case_meta"]["input_word_count"],
                "output_word_count": item["case_meta"]["output_word_count"],
                "input_tdata_width": item["top_ports"]["input_stream_TDATA"],
                "output_tdata_width": item["top_ports"]["output_stream_TDATA"],
            }
            for item in specs
        ],
    }
    write_json(PROJECT_ROOT / "harness_manifest.json", manifest)
    return manifest


def build_bitstream(args: argparse.Namespace) -> dict[str, Any]:
    bitstream = fullcombo.build_bitstream(
        PROJECT_ROOT,
        Path(args.vivado).expanduser().resolve(),
        force=args.force,
        timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
    )
    logs_dir = MEASUREMENT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / "vivado_bitstream.out.log"
    stdout_log.write_text(str(bitstream.get("stdout", "")), encoding="utf-8", errors="replace")
    bitstream.update(
        {
            "target_name": TARGET_NAME,
            "stdout_log": str(stdout_log),
            "failure_summary": summarize_build_failure(str(bitstream.get("stdout", ""))),
            "updated_at": timestamp(),
        }
    )
    write_json(build_result_json_path(), bitstream)
    return bitstream


def measurement_json_path() -> Path:
    return MEASUREMENT_ROOT / "measurements" / f"{TARGET_NAME}.measurement.json"


def build_result_json_path() -> Path:
    return MEASUREMENT_ROOT / "build_result.json"


def _source_variant_evidence(args: argparse.Namespace) -> dict[str, Any]:
    variant_name = sanitize_variant_name(args.variant_name)
    expected_target = variant_target_name(variant_name)
    root = variant_root(variant_name)
    project = root / "harness_project"
    build_path = root / "build_result.json"
    manifest_path = project / "harness_manifest.json"
    measurement_path = variant_measurement_json_path(variant_name)

    build_payload = read_json(build_path)
    manifest = read_json(manifest_path)
    measurement_payload = read_json(measurement_path)
    if not build_payload:
        raise FileNotFoundError(f"Missing variant build result: {build_path}")
    if not manifest:
        raise FileNotFoundError(f"Missing variant harness manifest: {manifest_path}")
    if not measurement_payload:
        raise FileNotFoundError(f"Missing variant COM5 measurement JSON: {measurement_path}")
    if manifest.get("implementation_variant") is not True:
        raise RuntimeError(f"Variant manifest is not marked implementation_variant=true: {manifest_path}")
    if manifest.get("variant_name") != variant_name or manifest.get("case_name") != expected_target:
        raise RuntimeError(
            "Variant manifest identity mismatch: "
            f"variant={manifest.get('variant_name')} case={manifest.get('case_name')} expected={expected_target}"
        )
    if build_payload.get("target_name") != expected_target:
        raise RuntimeError(f"Variant build target mismatch: {build_payload.get('target_name')} != {expected_target}")
    if build_payload.get("status") != "success":
        raise RuntimeError(f"Variant build is not successful: {build_payload.get('status')}")

    bitstream_path = Path(str(build_payload.get("bitstream", ""))).expanduser()
    if not bitstream_path.exists():
        raise FileNotFoundError(f"Variant bitstream not found: {bitstream_path}")
    actual_hash = sha256_file(bitstream_path)
    expected_hash = str(args.expected_bitstream_sha256 or "").strip().lower()
    if expected_hash and actual_hash.lower() != expected_hash:
        raise RuntimeError(
            "Refusing promotion: bitstream SHA256 mismatch "
            f"expected={expected_hash} actual={actual_hash}"
        )

    wns_text = str(build_payload.get("wns_ns") or fullcombo.parse_wns(project / "reports" / "timing_summary.rpt"))
    wns_value = fullcombo.parse_float(wns_text)
    uart_status_code = fullcombo.parse_int(fullcombo.frame_value(measurement_payload, "status_code"))
    measured_cycles = fullcombo.parse_int(fullcombo.frame_value(measurement_payload, "cycles"))
    measured_word_count = fullcombo.parse_int(fullcombo.frame_value(measurement_payload, "word_count"))
    expected_word_count = fullcombo.parse_int(manifest.get("streaming", {}).get("output_axis_word_count"))
    if uart_status_code != 0:
        raise RuntimeError(f"Refusing promotion: variant UART status_code is not 0: {uart_status_code}")
    if wns_value is None or wns_value < 0.0:
        raise RuntimeError(f"Refusing promotion: variant is not timing-clean: WNS={wns_text}")
    if measured_cycles is None or measured_cycles <= 0:
        raise RuntimeError(f"Refusing promotion: variant measurement cycles are invalid: {measured_cycles}")
    if expected_word_count is not None and measured_word_count != expected_word_count:
        raise RuntimeError(
            "Refusing promotion: variant output word_count mismatch "
            f"expected={expected_word_count} actual={measured_word_count}"
        )

    return {
        "variant_name": variant_name,
        "target_name": expected_target,
        "root": root,
        "project_root": project,
        "build_result_path": build_path,
        "build_payload": build_payload,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "measurement_path": measurement_path,
        "measurement_payload": measurement_payload,
        "bitstream_path": bitstream_path,
        "bitstream_sha256": actual_hash,
        "wns_ns": wns_text,
        "uart_status_code": uart_status_code,
        "measured_cycles": measured_cycles,
        "measured_word_count": measured_word_count,
    }


def _archive_current_official_evidence(variant_name: str, *, force: bool) -> str:
    existing_promotion = read_json(PROMOTION_MANIFEST_PATH)
    current_manifest = read_json(PROJECT_ROOT / "harness_manifest.json")
    already_promoted = (
        existing_promotion.get("source", {}).get("variant_name") == variant_name
        and current_manifest.get("promoted_from_implementation_variant") == variant_name
    )
    if already_promoted and not force:
        return str(existing_promotion.get("archive", {}).get("previous_official_evidence", ""))

    existing_items = [
        item
        for item in ("harness_project", "build_result.json", "logs", "measurements", "vivado_reports")
        if (MEASUREMENT_ROOT / item).exists()
    ]
    if not existing_items:
        return ""

    archive_dir = MEASUREMENT_ROOT / f"pre_promotion_original_official_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for item in existing_items:
        shutil.move(str(MEASUREMENT_ROOT / item), str(archive_dir / item))
    return str(archive_dir)


def _copytree_replace(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def promote_variant(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_variant_evidence(args)
    variant_name = str(source["variant_name"])
    archive_path = _archive_current_official_evidence(variant_name, force=bool(args.force))

    source_project = Path(source["project_root"])
    _copytree_replace(source_project, PROJECT_ROOT)
    _rewrite_project_text_paths(PROJECT_ROOT, source_project)

    source_reports = Path(source["root"]) / "vivado_reports"
    if source_reports.exists():
        _copytree_replace(source_reports, MEASUREMENT_ROOT / "vivado_reports")

    logs_dir = MEASUREMENT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    source_stdout = Path(str(source["build_payload"].get("stdout_log", "")))
    canonical_stdout = logs_dir / "vivado_bitstream.promoted_source.out.log"
    if source_stdout.exists():
        shutil.copy2(source_stdout, canonical_stdout)

    bitstream_path = canonical_bitstream_path()
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source["bitstream_path"]), bitstream_path)

    manifest = _replace_path_prefix(dict(source["manifest"]), source_project, PROJECT_ROOT)
    manifest.update(
        {
            "case_name": TARGET_NAME,
            "base_target_name": TARGET_NAME,
            "implementation_variant": False,
            "official_implementation_source": "promoted_implementation_variant",
            "promoted_from_implementation_variant": variant_name,
            "source_variant_target_name": source["target_name"],
            "source_variant_manifest": str(source["manifest_path"]),
            "source_variant_measurement_json": str(source["measurement_path"]),
            "source_variant_build_result_json": str(source["build_result_path"]),
        }
    )
    generated = dict(manifest.get("generated", {}))
    generated.update(
        {
            "project_root": str(PROJECT_ROOT),
            "rtl_dir": str(PROJECT_ROOT / "rtl"),
            "export_rtl_dir": str(PROJECT_ROOT / "export_rtl"),
            "constraints": str(PROJECT_ROOT / "constraints" / "harness.xdc"),
            "build_tcl": str(PROJECT_ROOT / "scripts" / "build_bitstream.tcl"),
            "bitstream": str(bitstream_path),
        }
    )
    manifest["generated"] = generated
    notes = list(manifest.get("notes", [])) if isinstance(manifest.get("notes"), list) else []
    notes.append(
        "Official e2e implementation promoted from a timing-clean low-DSP implementation variant; "
        "the original full-resource implementation remains recorded as a build failure."
    )
    manifest["notes"] = notes
    write_json(PROJECT_ROOT / "harness_manifest.json", manifest)

    build_payload = dict(source["build_payload"])
    build_payload.update(
        {
            "target_name": TARGET_NAME,
            "official_implementation_source": "promoted_implementation_variant",
            "promoted_from_variant": variant_name,
            "source_variant_target_name": source["target_name"],
            "source_build_result_json": str(source["build_result_path"]),
            "source_measurement_json": str(source["measurement_path"]),
            "source_bitstream": str(source["bitstream_path"]),
            "source_bitstream_sha256": source["bitstream_sha256"],
            "bitstream": str(bitstream_path),
            "bitstream_sha256": source["bitstream_sha256"],
            "project_root": str(PROJECT_ROOT),
            "stdout_log": str(canonical_stdout) if canonical_stdout.exists() else str(source_stdout),
            "archived_previous_official_evidence": archive_path,
            "updated_at": timestamp(),
        }
    )
    write_json(build_result_json_path(), build_payload)

    promotion_manifest = {
        "generated_at": timestamp(),
        "status": "promoted_ready_for_official_com5_measurement",
        "target_name": TARGET_NAME,
        "official_policy": "real e2e latency is valid only after the canonical promoted bitstream is measured over COM5",
        "source": {
            "variant_name": variant_name,
            "variant_target_name": source["target_name"],
            "variant_root": str(source["root"]),
            "build_result_json": str(source["build_result_path"]),
            "measurement_json": str(source["measurement_path"]),
            "bitstream": str(source["bitstream_path"]),
            "bitstream_sha256": source["bitstream_sha256"],
            "uart_status_code": source["uart_status_code"],
            "cycles": source["measured_cycles"],
            "word_count": source["measured_word_count"],
            "wns_ns": source["wns_ns"],
        },
        "canonical": {
            "measurement_root": str(MEASUREMENT_ROOT),
            "project_root": str(PROJECT_ROOT),
            "harness_manifest": str(PROJECT_ROOT / "harness_manifest.json"),
            "build_result_json": str(build_result_json_path()),
            "bitstream": str(bitstream_path),
            "bitstream_sha256": sha256_file(bitstream_path),
            "expected_measurement_json": str(measurement_json_path()),
        },
        "archive": {
            "previous_official_evidence": archive_path,
        },
        "guardrails": [
            "source UART status_code=0",
            "source WNS>=0",
            "source bitstream SHA256 matched the expected value",
            "canonical latency remains unset until canonical COM5 measurement is run",
        ],
    }
    write_json(PROMOTION_MANIFEST_PATH, promotion_manifest)
    return promotion_manifest


def summarize_build_failure(stdout: str, limit: int = 6) -> str:
    lines = [
        line.strip()
        for line in stdout.splitlines()
        if "ERROR:" in line or "CRITICAL WARNING:" in line or "Design is not routable" in line
    ]
    return " | ".join(lines[-limit:])[:1200]


def measure(args: argparse.Namespace) -> dict[str, Any]:
    bitstream_path = canonical_bitstream_path()
    if not bitstream_path.exists():
        raise FileNotFoundError(f"Canonical e2e bitstream not found: {bitstream_path}")
    expected_hash = str(args.expected_bitstream_sha256 or "").strip().lower()
    if expected_hash:
        actual_hash = sha256_file(bitstream_path).lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Refusing measurement: canonical bitstream SHA256 mismatch "
                f"expected={expected_hash} actual={actual_hash}"
            )
    if measurement_json_path().exists() and not args.force:
        raise FileExistsError(f"Measurement JSON already exists; pass --force to overwrite: {measurement_json_path()}")
    measurements_dir = MEASUREMENT_ROOT / "measurements"
    logs_dir = MEASUREMENT_ROOT / "logs"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurement = fullcombo.measure_harness(
        manifest=PROJECT_ROOT / "harness_manifest.json",
        bitstream=bitstream_path,
        csv_path=measurements_dir / "measurements_raw.csv",
        json_out=measurement_json_path(),
        serial_port=args.serial_port,
        vivado=Path(args.vivado).expanduser().resolve(),
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        runs=args.runs,
        uart_timeout_seconds=args.uart_timeout_seconds,
        post_program_delay_seconds=args.post_program_delay_seconds,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        skip_measure=args.skip_measure,
    )
    (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
    return measurement


def current_result() -> dict[str, Any]:
    payload = read_json(measurement_json_path())
    build_payload = read_json(build_result_json_path())
    promotion_payload = read_json(PROMOTION_MANIFEST_PATH)
    bitstream_path = canonical_bitstream_path()
    wns = fullcombo.parse_wns(PROJECT_ROOT / "reports" / "timing_summary.rpt") or str(build_payload.get("wns_ns", ""))
    status, board_execution, failure_category, failure_reason, timing_clean = fullcombo.classify_measurement(payload, wns)
    bitstream_status = "success" if bitstream_path.exists() else build_payload.get("status", "not_run")
    if not payload and bitstream_status in {"failed", "timeout"}:
        status = "bitstream_build_fail"
        board_execution = "not_measured"
        failure_category = "bitstream_build_fail"
        failure_reason = build_payload.get("failure_summary", "") or str(build_payload.get("returncode", ""))
    packed = fullcombo.frame_value(payload, "checksum")
    argmax = ""
    checksum24 = ""
    if isinstance(packed, int):
        argmax = (packed >> 24) & 0xFF
        checksum24 = packed & 0xFFFFFF
    manifest = read_json(PROJECT_ROOT / "harness_manifest.json")
    promoted_from_variant = str(
        manifest.get("promoted_from_implementation_variant")
        or build_payload.get("promoted_from_variant")
        or promotion_payload.get("source", {}).get("variant_name", "")
    )
    notes = "Latency-only deterministic parameters; argmax is channel sanity only, not trained sonar accuracy."
    if promoted_from_variant:
        notes = (
            f"{notes} Official implementation promoted from variant {promoted_from_variant}; "
            "original full-resource implementation remains recorded as a build failure."
        )
    return {
        "target_name": TARGET_NAME,
        "status": status if payload or bitstream_status in {"failed", "timeout"} else "not_run_ready_for_e2e_harness_build",
        "board_execution_status": board_execution,
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "bitstream_status": bitstream_status,
        "uart_status_code": fullcombo.frame_value(payload, "status_code") if payload else "",
        "e2e_cycles": fullcombo.frame_value(payload, "cycles") if payload else "",
        "e2e_latency_ms": payload.get("latency_ms", "") if payload else "",
        "axis_word_count": fullcombo.frame_value(payload, "word_count") if payload else "",
        "classifier_class_count": manifest.get("streaming", {}).get("classifier_class_count", ""),
        "checksum_argmax_packed": packed if packed != "" else "",
        "checksum24": checksum24,
        "argmax": argmax,
        "wns_ns": wns,
        "timing_clean": timing_clean,
        "measurement_json_path": str(measurement_json_path()) if payload else "",
        "bitstream_path": str(bitstream_path) if bitstream_path.exists() else "",
        "bitstream_sha256": sha256_file(bitstream_path) if bitstream_path.exists() else "",
        "timing_report_path": str(PROJECT_ROOT / "reports" / "timing_summary.rpt") if (PROJECT_ROOT / "reports" / "timing_summary.rpt").exists() else "",
        "notes": notes,
        "official_implementation_source": build_payload.get("official_implementation_source", manifest.get("official_implementation_source", "")),
        "promoted_from_variant": promoted_from_variant,
        "official_promotion_manifest": str(PROMOTION_MANIFEST_PATH) if promotion_payload else "",
        "completion_mode": manifest.get("completion_mode", build_payload.get("completion_mode", "")),
        "harness_fifo_depth": manifest.get("harness_fifo_depth", build_payload.get("fifo_depth", "")),
        "harness_fifo_style": manifest.get("harness_fifo_style", build_payload.get("fifo_style", "")),
        "board_harness_lut": build_payload.get("lut", ""),
        "board_harness_ff": build_payload.get("ff", ""),
        "board_harness_bram_tile": build_payload.get("bram_tile", ""),
        "board_harness_dsp": build_payload.get("dsp", ""),
        "build_result_json_path": str(build_result_json_path()) if build_payload else "",
        "build_stdout_log": str(build_payload.get("stdout_log", "")) if build_payload else "",
        "build_failure_summary": str(build_payload.get("failure_summary", "")) if build_payload else "",
    }


def build_outputs(mode: str) -> dict[str, Any]:
    result = current_result()
    blockers: list[dict[str, Any]] = []
    if result["bitstream_status"] != "success":
        blocker_category = "e2e_bitstream_build_fail" if result["bitstream_status"] in {"failed", "timeout"} else "e2e_bitstream_not_built"
        evidence_path = result.get("build_stdout_log") or result.get("build_result_json_path") or str(PROJECT_ROOT)
        evidence_detail = result.get("build_failure_summary") or "No full-network bitstream exists yet."
        blockers.append(
            {
                "scope": "e2e",
                "case_name": TARGET_NAME,
                "blocker_category": blocker_category,
                "severity": "blocking",
                "evidence_path": evidence_path,
                "evidence_detail": evidence_detail,
                "required_resolution": "resolve implementation failure, then rerun sonar_classifier_e2e_board_validation.py build",
            }
        )
    if result["bitstream_status"] == "success" and not result["measurement_json_path"]:
        blockers.append(
            {
                "scope": "e2e",
                "case_name": TARGET_NAME,
                "blocker_category": "e2e_board_not_measured",
                "severity": "blocking",
                "evidence_path": str(PROJECT_ROOT / "bitstream" / f"{TARGET_NAME}.bit"),
                "evidence_detail": "Full-network bitstream exists but COM5 measurement JSON is missing.",
                "required_resolution": "run sonar_classifier_e2e_board_validation.py measure",
            }
        )
    if result["measurement_json_path"] and result["status"] != "board_measured_timing_clean":
        blockers.append(
            {
                "scope": "e2e",
                "case_name": TARGET_NAME,
                "blocker_category": "e2e_not_strict_timing_clean",
                "severity": "strict_latency_blocking",
                "evidence_path": result["measurement_json_path"],
                "evidence_detail": f"status={result['status']} uart={result['uart_status_code']} WNS={result['wns_ns']}",
                "required_resolution": "recover timing or runtime until UART status_code=0 and WNS>=0",
            }
        )

    blocker_categories = "|".join(sorted(Counter(row["blocker_category"] for row in blockers)))
    status_row = {**result, "blocker_count": len(blockers), "blocker_categories": blocker_categories}
    measurement_rows = []
    if result["measurement_json_path"]:
        measurement_rows.append(
            {
                "target_name": TARGET_NAME,
                "status": result["status"],
                "uart_status_code": result["uart_status_code"],
                "cycles": result["e2e_cycles"],
                "latency_ms": result["e2e_latency_ms"],
                "axis_word_count": result["axis_word_count"],
                "classifier_class_count": result["classifier_class_count"],
                "checksum_argmax_packed": result["checksum_argmax_packed"],
                "checksum24": result["checksum24"],
                "argmax": result["argmax"],
                "wns_ns": result["wns_ns"],
                "timing_clean": result["timing_clean"],
                "measurement_json_path": result["measurement_json_path"],
                "bitstream_path": result["bitstream_path"],
                "timing_report_path": result["timing_report_path"],
            }
        )
    timing_failures = measurement_rows if measurement_rows and result["status"] != "board_measured_timing_clean" else []
    summary = {
        "generated_at": timestamp(),
        "mode": mode,
        "target_name": TARGET_NAME,
        "status": result["status"],
        "bitstream_status": result["bitstream_status"],
        "uart_status_code": result["uart_status_code"],
        "e2e_cycles": result["e2e_cycles"],
        "e2e_latency_ms": result["e2e_latency_ms"],
        "axis_word_count": result["axis_word_count"],
        "classifier_class_count": result["classifier_class_count"],
        "argmax": result["argmax"],
        "wns_ns": result["wns_ns"],
        "timing_clean": result["timing_clean"],
        "strict_latency_usable": result["status"] == "board_measured_timing_clean",
        "official_implementation_source": result.get("official_implementation_source", ""),
        "promoted_from_variant": result.get("promoted_from_variant", ""),
        "official_promotion_manifest": result.get("official_promotion_manifest", ""),
        "completion_mode": result.get("completion_mode", ""),
        "harness_fifo_depth": result.get("harness_fifo_depth", ""),
        "harness_fifo_style": result.get("harness_fifo_style", ""),
        "bitstream_path": result.get("bitstream_path", ""),
        "bitstream_sha256": result.get("bitstream_sha256", ""),
        "measurement_json_path": result.get("measurement_json_path", ""),
        "build_result_json_path": result.get("build_result_json_path", ""),
        "resources": {
            "lut": result.get("board_harness_lut", ""),
            "ff": result.get("board_harness_ff", ""),
            "bram_tile": result.get("board_harness_bram_tile", ""),
            "dsp": result.get("board_harness_dsp", ""),
        },
        "blocker_count": len(blockers),
        "outputs": {
            "status_csv": str(E2E_DIR / "sonar_classifier_e2e_status.csv"),
            "measurements_csv": str(E2E_DIR / "sonar_classifier_e2e_measurements.csv"),
            "timing_failures_csv": str(E2E_DIR / "sonar_classifier_e2e_timing_failures.csv"),
            "blockers_csv": str(E2E_DIR / "sonar_classifier_e2e_blockers.csv"),
            "summary_json": str(E2E_DIR / "sonar_classifier_e2e_summary.json"),
            "latency_overlay_json": str(E2E_DIR / "sonar_classifier_e2e_latency_overlay.json"),
        },
    }

    readme = f"""# Sonar Classifier End-to-End Board Evidence

This directory is for the complete arch_84 sonar classifier board harness.

- End-to-end latency is valid only when it comes from this full-network harness.
- Primitive or combo latency sums are prediction baselines only.
- Timing-clean is strict: UART status_code=0 and WNS>=0.
- Parameters are latency-only deterministic test values; argmax is a channel
  sanity result and is not a trained sonar classifier accuracy claim.
- The UART checksum field is packed for this e2e harness as
  {{argmax[7:0], checksum24[23:0]}}.

Current status: `{result["status"]}`.
Strict latency usable: `{summary["strict_latency_usable"]}`.
"""

    write_csv(E2E_DIR / "sonar_classifier_e2e_status.csv", [status_row], STATUS_FIELDS)
    write_csv(E2E_DIR / "sonar_classifier_e2e_measurements.csv", measurement_rows, MEASUREMENT_FIELDS)
    write_csv(E2E_DIR / "sonar_classifier_e2e_timing_failures.csv", timing_failures, MEASUREMENT_FIELDS)
    write_csv(E2E_DIR / "sonar_classifier_e2e_blockers.csv", blockers, BLOCKER_FIELDS)
    write_json(E2E_DIR / "sonar_classifier_e2e_summary.json", summary)
    write_json(E2E_DIR / "sonar_classifier_e2e_latency_overlay.json", {"target_name": TARGET_NAME, "measurement": result})
    (E2E_DIR / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.mode == "promote":
        promote_variant(args)
    if args.mode in {"generate", "all"}:
        generate_harness(args)
    if args.mode in {"build", "all"}:
        if not (PROJECT_ROOT / "harness_manifest.json").exists() or args.force:
            generate_harness(args)
        build_bitstream(args)
    if args.mode in {"measure", "all"}:
        measure(args)
    summary = build_outputs(args.mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
