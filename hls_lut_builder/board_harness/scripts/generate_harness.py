#!/usr/bin/env python3
"""Generate a single-operator board harness project from an exported HLS case."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[3]
BOARD_HARNESS_DIR = ROOT / "hls_lut_builder" / "board_harness"
TEMPLATES_DIR = BOARD_HARNESS_DIR / "templates"
MODULES_DIR = BOARD_HARNESS_DIR / "modules"
DEFAULTS_PATH = BOARD_HARNESS_DIR / "configs" / "harness_defaults.yaml"
VIVADO_BUILD_ROOT = BOARD_HARNESS_DIR / "_vivado_builds"

SPIRIT_NS = {"spirit": "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single-operator board harness project")
    parser.add_argument("--case-dir", required=True, help="Path to an exported HLS case directory")
    parser.add_argument("--board-config", required=True, help="Path to board YAML config")
    parser.add_argument(
        "--output-dir",
        default=str(BOARD_HARNESS_DIR / "projects"),
        help="Directory for generated harness projects",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing harness directory")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def render_template(path: Path, mapping: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def parse_constants(source_text: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    pattern = re.compile(r"static constexpr int (\w+) = ([^;]+);")
    for match in pattern.finditer(source_text):
        name = match.group(1)
        expr = match.group(2).strip()
        ternary_match = re.match(r"^\((.+)\)\s*\?\s*(.+)\s*:\s*(.+)$", expr)
        if ternary_match:
            cond_expr = ternary_match.group(1).strip()
            true_expr = ternary_match.group(2).strip()
            false_expr = ternary_match.group(3).strip()
            expr = f"({true_expr} if ({cond_expr}) else {false_expr})"
        try:
            value = eval(expr, {"__builtins__": {}}, constants)
        except Exception:
            continue
        constants[name] = int(value)
    return constants


def parse_top_ports(verilog_text: str) -> dict[str, int]:
    ports: dict[str, int] = {}
    params: dict[str, int] = {}
    param_pattern = re.compile(r"^\s*parameter\s+(\w+)\s*=\s*([^;]+);", re.MULTILINE)
    port_pattern = re.compile(r"^\s*(input|output)\s*(\[[^\]]+\])?\s*([A-Za-z0-9_]+)\s*;", re.MULTILINE)
    width_pattern = re.compile(r"\[(\d+)\s*:\s*(\d+)\]")
    expr_width_pattern = re.compile(r"\[(.+):(.+)\]")

    for match in param_pattern.finditer(verilog_text):
        name = match.group(1)
        expr = match.group(2).strip()
        try:
            value = eval(expr, {"__builtins__": {}}, params)
        except Exception:
            continue
        params[name] = int(value)

    for match in port_pattern.finditer(verilog_text):
        width_token = match.group(2)
        name = match.group(3)
        width = 1
        if width_token:
            width_match = width_pattern.search(width_token)
            if width_match:
                msb = int(width_match.group(1))
                lsb = int(width_match.group(2))
                width = abs(msb - lsb) + 1
            else:
                expr_match = expr_width_pattern.search(width_token)
                if expr_match:
                    msb_expr = expr_match.group(1).strip()
                    lsb_expr = expr_match.group(2).strip()
                    try:
                        msb = int(eval(msb_expr, {"__builtins__": {}}, params))
                        lsb = int(eval(lsb_expr, {"__builtins__": {}}, params))
                        width = abs(msb - lsb) + 1
                    except Exception:
                        width = 1
        ports[name] = width
    return ports


def parse_component(component_path: Path) -> dict[str, Any]:
    tree = ET.parse(component_path)
    root = tree.getroot()
    component_name = root.findtext("spirit:name", namespaces=SPIRIT_NS)
    if not component_name:
        raise ValueError(f"Unable to find component name in {component_path}")

    busif_data: dict[str, dict[str, str]] = {}
    for busif in root.findall("./spirit:busInterfaces/spirit:busInterface", namespaces=SPIRIT_NS):
        name = busif.findtext("spirit:name", namespaces=SPIRIT_NS)
        if not name:
            continue
        params: dict[str, str] = {}
        for param in busif.findall("./spirit:parameters/spirit:parameter", namespaces=SPIRIT_NS):
            param_name = param.findtext("spirit:name", namespaces=SPIRIT_NS)
            param_value = param.findtext("spirit:value", namespaces=SPIRIT_NS)
            if param_name and param_value is not None:
                params[param_name] = param_value
        busif_data[name] = params
    return {"component_name": component_name, "bus_interfaces": busif_data}


def compute_case_meta(case_name: str, constants: dict[str, int]) -> dict[str, int]:
    in_h = int(constants.get("IN_H", constants.get("OUT_H", 1)))
    in_w = int(constants.get("IN_W", constants.get("OUT_W", 1)))
    out_h = int(constants.get("OUT_H", constants.get("IN_H", 1)))
    out_w = int(constants.get("OUT_W", constants.get("IN_W", 1)))
    input_packets_per_pixel = int(constants.get("INPUT_PACKETS_PER_PIXEL", 1))
    output_packets_per_pixel = int(constants.get("OUTPUT_PACKETS_PER_PIXEL", 1))
    input_word_count = in_h * in_w * input_packets_per_pixel
    output_word_count = out_h * out_w * output_packets_per_pixel

    if "WEIGHT_SIZE" in constants and "BIAS_SIZE" in constants:
        return {
            "input_word_count": input_word_count,
            "output_word_count": output_word_count,
            "weights_depth": int(constants["WEIGHT_SIZE"]),
            "biases_depth": int(constants["BIAS_SIZE"]),
        }

    if case_name.startswith("stem_conv_k3_s2"):
        return {
            "input_word_count": input_word_count,
            "output_word_count": output_word_count,
            "weights_depth": int(constants["OUT_CH"] * constants["IN_CH"] * constants["K"] * constants["K"]),
            "biases_depth": int(constants["OUT_CH"]),
        }

    if case_name.startswith("pw_conv") or case_name.startswith("fc_layer"):
        if case_name.startswith("fc_layer"):
            input_word_count = int(constants.get("INPUT_PACKETS", 1))
            output_word_count = 1
        return {
            "input_word_count": input_word_count,
            "output_word_count": output_word_count,
            "weights_depth": int(constants["OUT_CH"] * constants["IN_CH"]),
            "biases_depth": int(constants["OUT_CH"]),
        }

    if case_name.startswith("dw_conv"):
        channels = int(constants.get("OUT_CH", constants["IN_CH"]))
        return {
            "input_word_count": input_word_count,
            "output_word_count": output_word_count,
            "weights_depth": int(channels * constants["K"] * constants["K"]),
            "biases_depth": int(channels),
        }

    if case_name.startswith("skip"):
        return {
            "input_word_count": input_word_count,
            "output_word_count": input_word_count,
            "weights_depth": 1,
            "biases_depth": 1,
        }

    if case_name.startswith("global_avg_pool"):
        return {
            "input_word_count": input_word_count,
            "output_word_count": 1,
            "weights_depth": 1,
            "biases_depth": 1,
        }

    raise ValueError(f"Unsupported case metadata formula for {case_name}")


def twos_complement_hex(value: int, width_bits: int) -> str:
    mask = (1 << width_bits) - 1
    digits = max(1, width_bits // 4)
    return f"{(value & mask):0{digits}x}"


def deterministic_byte(word_idx: int, lane_idx: int) -> int:
    return ((word_idx * 3 + lane_idx * 5) % 23) - 11


def generate_stream_mem_files(data_dir: Path, stem: str, word_count: int, data_width: int) -> list[Path]:
    lanes = data_width // 8
    lane_lines = [[] for _ in range(lanes)]
    for word_idx in range(word_count):
        for lane_idx in range(lanes):
            byte_value = deterministic_byte(word_idx, lane_idx) & 0xFF
            lane_lines[lane_idx].append(f"{byte_value:02x}")

    mem_paths: list[Path] = []
    for lane_idx, lines in enumerate(lane_lines):
        lane_path = data_dir / f"{stem}_lane{lane_idx:02d}.mem"
        lane_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        mem_paths.append(lane_path)
    return mem_paths


def generate_stream_word_mem_file(data_dir: Path, stem: str, word_count: int, data_width: int) -> Path:
    lanes = data_width // 8
    lines: list[str] = []
    for word_idx in range(word_count):
        byte_hexes: list[str] = []
        for lane_idx in range(lanes):
            byte_value = deterministic_byte(word_idx, lane_idx) & 0xFF
            byte_hexes.append(f"{byte_value:02x}")
        lines.append("".join(reversed(byte_hexes)))
    word_path = data_dir / f"{stem}.mem"
    word_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return word_path


def generate_param_mem(path: Path, depth: int, data_width: int, seed_offset: int) -> None:
    lines = []
    for idx in range(depth):
        value = ((idx * 7 + seed_offset) % 19) - 9
        lines.append(twos_complement_hex(value, data_width))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_tree_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        if child.is_file():
            shutil.copy2(child, dst_dir / child.name)


def build_bram_defaults(kind: str, data_width: int, addr_width: int, has_port_b: bool) -> str:
    if has_port_b:
        return ""
    bytes_width = max(1, data_width // 8)
    return "\n".join(
        [
            f"assign {kind}_addrb = {{{addr_width}{{1'b0}}}};",
            f"assign {kind}_enb = 1'b0;",
            f"assign {kind}_wenb = {{{bytes_width}{{1'b0}}}};",
            f"assign {kind}_dinb = {{{data_width}{{1'b0}}}};",
            f"assign {kind}_clkb = clk_main;",
            f"assign {kind}_rstb = ~rst_n;",
        ]
    )


def detect_param_interface(top_ports: dict[str, int], kind: str) -> dict[str, Any]:
    scalar_port = kind
    bram_data_port = f"{kind}_Dout_A"
    bram_addr_port = f"{kind}_Addr_A"
    if bram_data_port in top_ports and bram_addr_port in top_ports:
        return {
            "mode": "bram",
            "data_width": int(top_ports[bram_data_port]),
            "addr_width": int(top_ports[bram_addr_port]),
            "has_port_b": f"{kind}_Addr_B" in top_ports,
            "port_name": scalar_port,
        }
    if scalar_port in top_ports:
        return {
            "mode": "scalar",
            "data_width": int(top_ports[scalar_port]),
            "addr_width": 1,
            "has_port_b": False,
            "port_name": scalar_port,
        }
    raise KeyError(f"Unable to detect {kind} interface from RTL top ports")


def build_scalar_param_logic(kind: str, data_width: int) -> str:
    if data_width <= 1:
        return f"wire {kind}_scalar = 1'b0;"
    return f"wire [{data_width - 1}:0] {kind}_scalar = {data_width}'d0;"


def build_bram_param_logic(
    kind: str,
    data_width: int,
    addr_width: int,
    depth: int,
    mem_file: str,
    has_port_b: bool,
) -> str:
    bytes_width = max(1, data_width // 8)
    return "\n".join(
        [
            f"wire [{addr_width - 1}:0] {kind}_addra;",
            f"wire {kind}_ena;",
            f"wire [{bytes_width - 1}:0] {kind}_wena;",
            f"wire [{data_width - 1}:0] {kind}_dina;",
            f"wire [{data_width - 1}:0] {kind}_douta;",
            f"wire {kind}_clka;",
            f"wire {kind}_rsta;",
            f"wire [{addr_width - 1}:0] {kind}_addrb;",
            f"wire {kind}_enb;",
            f"wire [{bytes_width - 1}:0] {kind}_wenb;",
            f"wire [{data_width - 1}:0] {kind}_dinb;",
            f"wire [{data_width - 1}:0] {kind}_doutb;",
            f"wire {kind}_clkb;",
            f"wire {kind}_rstb;",
            build_bram_defaults(kind, data_width, addr_width, has_port_b),
            "",
            "simple_dual_port_bram #(",
            f"    .DATA_WIDTH({data_width}),",
            f"    .ADDR_WIDTH({addr_width}),",
            f"    .DEPTH({depth}),",
            f'    .INIT_FILE("{mem_file}")',
            f") u_{kind}_mem (",
            f"    .clka({kind}_clka),",
            f"    .rsta({kind}_rsta),",
            f"    .ena({kind}_ena),",
            f"    .wea({kind}_wena),",
            f"    .addra({kind}_addra),",
            f"    .dina({kind}_dina),",
            f"    .douta({kind}_douta),",
            f"    .clkb({kind}_clkb),",
            f"    .rstb({kind}_rstb),",
            f"    .enb({kind}_enb),",
            f"    .web({kind}_wenb),",
            f"    .addrb({kind}_addrb),",
            f"    .dinb({kind}_dinb),",
            f"    .doutb({kind}_doutb)",
            ");",
        ]
    )


def build_kernel_instance(
    module_name: str,
    ports: dict[str, int],
    axil_addr_width: int,
    axil_data_width: int,
    param_interfaces: dict[str, dict[str, Any]],
) -> str:
    lines = [f"{module_name} u_kernel ("]

    def connect(name: str, signal: str) -> None:
        if name in ports:
            lines.append(f"    .{name}({signal}),")

    connect("ap_clk", "clk_main")
    connect("ap_rst_n", "rst_n")
    connect("interrupt", "interrupt")

    axil_signals = {
        "s_axi_control_AWVALID": "s_axi_control_AWVALID",
        "s_axi_control_AWREADY": "s_axi_control_AWREADY",
        "s_axi_control_AWADDR": "s_axi_control_AWADDR",
        "s_axi_control_WVALID": "s_axi_control_WVALID",
        "s_axi_control_WREADY": "s_axi_control_WREADY",
        "s_axi_control_WDATA": "s_axi_control_WDATA",
        "s_axi_control_WSTRB": "s_axi_control_WSTRB",
        "s_axi_control_ARVALID": "s_axi_control_ARVALID",
        "s_axi_control_ARREADY": "s_axi_control_ARREADY",
        "s_axi_control_ARADDR": "s_axi_control_ARADDR",
        "s_axi_control_RVALID": "s_axi_control_RVALID",
        "s_axi_control_RREADY": "s_axi_control_RREADY",
        "s_axi_control_RDATA": "s_axi_control_RDATA",
        "s_axi_control_RRESP": "s_axi_control_RRESP",
        "s_axi_control_BVALID": "s_axi_control_BVALID",
        "s_axi_control_BREADY": "s_axi_control_BREADY",
        "s_axi_control_BRESP": "s_axi_control_BRESP",
    }
    for port_name, signal_name in axil_signals.items():
        connect(port_name, signal_name)

    axis_signals = {
        "input_stream_TDATA": "input_tdata",
        "input_stream_TKEEP": "input_tkeep",
        "input_stream_TSTRB": "input_tstrb",
        "input_stream_TLAST": "input_tlast",
        "input_stream_TVALID": "input_tvalid",
        "input_stream_TREADY": "input_tready",
        "output_stream_TDATA": "output_tdata",
        "output_stream_TKEEP": "output_tkeep",
        "output_stream_TSTRB": "output_tstrb",
        "output_stream_TLAST": "output_tlast",
        "output_stream_TVALID": "output_tvalid",
        "output_stream_TREADY": "output_tready",
    }
    for port_name, signal_name in axis_signals.items():
        connect(port_name, signal_name)

    for kind in ("weights", "biases"):
        iface = param_interfaces[kind]
        if iface["mode"] == "scalar":
            connect(kind, f"{kind}_scalar")
            continue
        for suffix, signal in (
            ("Addr_A", f"{kind}_addra"),
            ("EN_A", f"{kind}_ena"),
            ("WEN_A", f"{kind}_wena"),
            ("Din_A", f"{kind}_dina"),
            ("Dout_A", f"{kind}_douta"),
            ("Clk_A", f"{kind}_clka"),
            ("Rst_A", f"{kind}_rsta"),
            ("Addr_B", f"{kind}_addrb"),
            ("EN_B", f"{kind}_enb"),
            ("WEN_B", f"{kind}_wenb"),
            ("Din_B", f"{kind}_dinb"),
            ("Dout_B", f"{kind}_doutb"),
            ("Clk_B", f"{kind}_clkb"),
            ("Rst_B", f"{kind}_rstb"),
        ):
            connect(f"{kind}_{suffix}", signal)

    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append(");")
    return "\n".join(lines)


def make_constraint_line(port_name: str, package_pin: str | None, iostandard: str | None) -> str:
    if not package_pin or package_pin.startswith("TODO_"):
        return ""
    lines = [f"set_property PACKAGE_PIN {package_pin} [get_ports {port_name}]"]
    if iostandard:
        lines.append(f"set_property IOSTANDARD {iostandard} [get_ports {port_name}]")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def short_build_tag(case_name: str, module_name: str) -> str:
    safe_module = re.sub(r"[^A-Za-z0-9_]+", "_", module_name)[:12]
    digest = hashlib.sha1(case_name.encode("utf-8")).hexdigest()[:8]
    return f"h_{safe_module}_{digest}"


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()
    board_cfg = load_yaml(Path(args.board_config).expanduser().resolve())
    defaults_cfg = load_yaml(DEFAULTS_PATH)

    case_name = case_dir.name
    component_path = case_dir / "project" / "solution1" / "impl" / "ip" / "component.xml"
    export_rtl_src = case_dir / "project" / "solution1" / "impl" / "ip" / "hdl" / "verilog"
    source_cpp_path = case_dir / "src"

    if not component_path.exists():
        raise FileNotFoundError(f"component.xml not found: {component_path}")
    if not export_rtl_src.exists():
        raise FileNotFoundError(f"exported RTL directory not found: {export_rtl_src}")

    component_info = parse_component(component_path)
    module_name = component_info["component_name"]
    build_tag = short_build_tag(case_name, module_name)
    top_verilog_path = export_rtl_src / f"{module_name}.v"
    if not top_verilog_path.exists():
        raise FileNotFoundError(f"top RTL not found: {top_verilog_path}")

    source_cpp_candidates = list(source_cpp_path.glob("*.cpp"))
    if not source_cpp_candidates:
        raise FileNotFoundError(f"No generated C++ source found in {source_cpp_path}")
    source_cpp_text = source_cpp_candidates[0].read_text(encoding="utf-8")
    constants = parse_constants(source_cpp_text)
    case_meta = compute_case_meta(case_name, constants)

    top_ports = parse_top_ports(top_verilog_path.read_text(encoding="utf-8"))
    busif = component_info["bus_interfaces"]

    project_root = Path(args.output_dir).expanduser().resolve() / f"harness_{case_name}"
    if project_root.exists():
        if not args.force:
            raise FileExistsError(f"Harness project already exists: {project_root}")
        shutil.rmtree(project_root)

    rtl_dir = project_root / "rtl"
    export_rtl_dir = project_root / "export_rtl"
    data_dir = project_root / "data"
    constraints_dir = project_root / "constraints"
    scripts_dir = project_root / "scripts"
    vivado_project_dir = (VIVADO_BUILD_ROOT / build_tag).resolve()
    bitstream_output_dir = project_root / "bitstream"
    reports_output_dir = project_root / "reports"

    data_dir.mkdir(parents=True, exist_ok=True)
    copy_tree_files(MODULES_DIR, rtl_dir)
    copy_tree_files(export_rtl_src, export_rtl_dir)

    input_tdata_width = top_ports["input_stream_TDATA"]
    input_keep_width = top_ports.get("input_stream_TKEEP", max(1, input_tdata_width // 8))
    output_tdata_width = top_ports["output_stream_TDATA"]
    output_keep_width = top_ports.get("output_stream_TKEEP", max(1, output_tdata_width // 8))
    axil_addr_width = top_ports["s_axi_control_AWADDR"]
    axil_data_width = top_ports["s_axi_control_WDATA"]

    param_interfaces = {
        "weights": detect_param_interface(top_ports, "weights"),
        "biases": detect_param_interface(top_ports, "biases"),
    }
    weights_data_width = int(param_interfaces["weights"]["data_width"])
    weights_addr_width = int(param_interfaces["weights"]["addr_width"])
    weights_has_port_b = bool(param_interfaces["weights"]["has_port_b"])
    biases_data_width = int(param_interfaces["biases"]["data_width"])
    biases_addr_width = int(param_interfaces["biases"]["addr_width"])
    biases_has_port_b = bool(param_interfaces["biases"]["has_port_b"])

    weights_mem_file = "weights.mem"
    biases_mem_file = "biases.mem"

    input_mem_paths = generate_stream_mem_files(
        data_dir,
        stem="input_stream",
        word_count=case_meta["input_word_count"],
        data_width=input_tdata_width,
    )
    input_word_mem_path = generate_stream_word_mem_file(
        data_dir,
        stem="input_stream_words",
        word_count=case_meta["input_word_count"],
        data_width=input_tdata_width,
    )
    if param_interfaces["weights"]["mode"] == "bram":
        generate_param_mem(data_dir / weights_mem_file, case_meta["weights_depth"], weights_data_width, seed_offset=3)
    if param_interfaces["biases"]["mode"] == "bram":
        generate_param_mem(data_dir / biases_mem_file, case_meta["biases_depth"], biases_data_width, seed_offset=11)

    input_mem_mapping = {f"INPUT_MEM_FILE_{lane_idx:02d}": '""' for lane_idx in range(32)}
    for lane_idx, mem_path in enumerate(input_mem_paths):
        input_mem_mapping[f"INPUT_MEM_FILE_{lane_idx:02d}"] = f'"../data/{mem_path.name}"'

    kernel_instance = build_kernel_instance(
        module_name,
        top_ports,
        axil_addr_width,
        axil_data_width,
        param_interfaces,
    )
    weights_support_logic = (
        build_bram_param_logic(
            "weights",
            weights_data_width,
            weights_addr_width,
            case_meta["weights_depth"],
            f"../data/{weights_mem_file}",
            weights_has_port_b,
        )
        if param_interfaces["weights"]["mode"] == "bram"
        else build_scalar_param_logic("weights", weights_data_width)
    )
    biases_support_logic = (
        build_bram_param_logic(
            "biases",
            biases_data_width,
            biases_addr_width,
            case_meta["biases_depth"],
            f"../data/{biases_mem_file}",
            biases_has_port_b,
        )
        if param_interfaces["biases"]["mode"] == "bram"
        else build_scalar_param_logic("biases", biases_data_width)
    )
    harness_mapping = {
            "CLK_FREQ_HZ": str(board_cfg["clock"]["freq_hz"]),
            "UART_BAUD": str(board_cfg.get("uart", {}).get("baud", defaults_cfg["uart_baud"])),
            "BOOT_DELAY_CYCLES": str(defaults_cfg["boot_delay_cycles"]),
            "STATUS_LED_WIDTH": str(board_cfg.get("status_led", {}).get("width", 2)),
            "INPUT_TDATA_WIDTH": str(input_tdata_width),
            "INPUT_KEEP_WIDTH": str(input_keep_width),
            "OUTPUT_TDATA_WIDTH": str(output_tdata_width),
            "OUTPUT_KEEP_WIDTH": str(output_keep_width),
            "INPUT_WORD_COUNT": str(case_meta["input_word_count"]),
            "OUTPUT_WORD_COUNT": str(case_meta["output_word_count"]),
            "AXIL_ADDR_WIDTH": str(axil_addr_width),
            "AXIL_DATA_WIDTH": str(axil_data_width),
            "WEIGHTS_DATA_WIDTH": str(weights_data_width),
            "WEIGHTS_ADDR_WIDTH": str(weights_addr_width),
            "WEIGHTS_DEPTH": str(case_meta["weights_depth"]),
            "WEIGHTS_HAS_PORT_B": "1" if weights_has_port_b else "0",
            "WEIGHTS_IMPL_MODE": param_interfaces["weights"]["mode"],
            "BIASES_DATA_WIDTH": str(biases_data_width),
            "BIASES_ADDR_WIDTH": str(biases_addr_width),
            "BIASES_DEPTH": str(case_meta["biases_depth"]),
            "BIASES_HAS_PORT_B": "1" if biases_has_port_b else "0",
            "BIASES_IMPL_MODE": param_interfaces["biases"]["mode"],
            "WEIGHTS_MEM_FILE": f"../data/{weights_mem_file}",
            "BIASES_MEM_FILE": f"../data/{biases_mem_file}",
            "INPUT_WORD_MEM_FILE": f"../data/{input_word_mem_path.name}",
            "WEIGHTS_SUPPORT_LOGIC": weights_support_logic,
            "BIASES_SUPPORT_LOGIC": biases_support_logic,
            "KERNEL_INSTANCE": kernel_instance,
        }
    harness_mapping.update(input_mem_mapping)
    harness_top = render_template(
        TEMPLATES_DIR / "harness_top.v.tmpl",
        harness_mapping,
    )
    write_text(rtl_dir / "harness_top.v", harness_top)

    clock_cfg = board_cfg["clock"]
    led_cfg = board_cfg.get("status_led", {})
    led_constraints = []
    for idx, pin in enumerate(led_cfg.get("package_pins", [])):
        led_line = make_constraint_line(f"status_led[{idx}]", pin, led_cfg.get("iostandard"))
        if led_line:
            led_constraints.append(led_line)

    if "package_pin_p" in clock_cfg and "package_pin_n" in clock_cfg:
        clock_constraints = [
            make_constraint_line(clock_cfg["port_name_p"], clock_cfg.get("package_pin_p"), clock_cfg.get("iostandard")),
            make_constraint_line(clock_cfg["port_name_n"], clock_cfg.get("package_pin_n"), clock_cfg.get("iostandard")),
        ]
        clock_create_port = clock_cfg["port_name_p"]
    else:
        clock_constraints = [
            make_constraint_line(
                clock_cfg["port_name"],
                clock_cfg.get("package_pin"),
                clock_cfg.get("iostandard"),
            )
        ]
        clock_create_port = clock_cfg["port_name"]

    xdc_text = render_template(
        TEMPLATES_DIR / "harness_constraints.xdc.tmpl",
        {
            "CLOCK_PERIOD_NS": str(clock_cfg["period_ns"]),
            "CLOCK_CREATE_PORT": str(clock_create_port),
            "CLOCK_PIN_CONSTRAINTS": "\n".join([line for line in clock_constraints if line]),
            "RESET_PIN_CONSTRAINT": make_constraint_line(
                board_cfg["reset"]["port_name"],
                board_cfg["reset"].get("package_pin"),
                board_cfg["reset"].get("iostandard"),
            ),
            "UART_RX_PIN_CONSTRAINT": make_constraint_line(
                board_cfg["uart"].get("rx_port_name", "uart_rx"),
                board_cfg["uart"].get("rx_package_pin"),
                board_cfg["uart"].get("iostandard"),
            ),
            "UART_TX_PIN_CONSTRAINT": make_constraint_line(
                board_cfg["uart"]["tx_port_name"],
                board_cfg["uart"].get("package_pin"),
                board_cfg["uart"].get("iostandard"),
            ),
            "LED_PIN_CONSTRAINTS": "\n".join(led_constraints),
        },
    )
    write_text(constraints_dir / "harness.xdc", xdc_text)

    drc_relax = ""
    if board_cfg.get("allow_unconstrained_io", False):
        drc_relax = "\n".join(
            [
                "set_property SEVERITY {Warning} [get_drc_checks UCIO-1]",
                "set_property SEVERITY {Warning} [get_drc_checks NSTD-1]",
            ]
        )
    build_tcl = render_template(
        TEMPLATES_DIR / "build_bitstream.tcl.tmpl",
        {
            "PROJECT_NAME": build_tag,
            "TARGET_PART": str(board_cfg["target_part"]),
            "VIVADO_PROJECT_DIR": str(vivado_project_dir).replace("\\", "/"),
            "RTL_DIR": str(rtl_dir).replace("\\", "/"),
            "EXPORT_RTL_DIR": str(export_rtl_dir).replace("\\", "/"),
            "CONSTRAINT_FILE": str(constraints_dir / "harness.xdc").replace("\\", "/"),
            "BITSTREAM_OUTPUT_DIR": str(bitstream_output_dir).replace("\\", "/"),
            "REPORTS_OUTPUT_DIR": str(reports_output_dir).replace("\\", "/"),
            "ARTIFACT_BASENAME": case_name,
            "OPTIONAL_DRC_RELAX": drc_relax,
        },
    )
    write_text(scripts_dir / "build_bitstream.tcl", build_tcl)

    manifest = {
        "case_name": case_name,
        "case_dir": str(case_dir),
        "module_name": module_name,
        "board_config": str(Path(args.board_config).expanduser().resolve()),
        "generated": {
            "project_root": str(project_root),
            "rtl_dir": str(rtl_dir),
            "export_rtl_dir": str(export_rtl_dir),
            "constraints": str(constraints_dir / "harness.xdc"),
            "build_tcl": str(scripts_dir / "build_bitstream.tcl"),
            "vivado_project_dir": str(vivado_project_dir),
            "input_mem_files": [str(path) for path in input_mem_paths],
            "input_word_mem": str(input_word_mem_path),
            "weights_mem": str(data_dir / weights_mem_file) if param_interfaces["weights"]["mode"] == "bram" else None,
            "biases_mem": str(data_dir / biases_mem_file) if param_interfaces["biases"]["mode"] == "bram" else None,
        },
        "interface_summary": {
            "input_tdata_width": input_tdata_width,
            "output_tdata_width": output_tdata_width,
            "axil_addr_width": axil_addr_width,
            "axil_data_width": axil_data_width,
            "weights_impl_mode": param_interfaces["weights"]["mode"],
            "weights_has_port_b": weights_has_port_b,
            "biases_impl_mode": param_interfaces["biases"]["mode"],
            "biases_has_port_b": biases_has_port_b,
        },
        "case_meta": case_meta,
    }
    write_text(project_root / "harness_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
