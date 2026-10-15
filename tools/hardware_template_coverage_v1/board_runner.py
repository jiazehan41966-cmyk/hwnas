"""Manifest-bound PYNQ runner for Experiment 3; importing never accesses hardware.

Offline packaging: prepare_bundle(case_dir, mode="probe") admits five numeric
correctness calls under source/route/hash gates to observe integrated phases.
Full timing uses prepare_bundle(case_dir) after reviewed phase validation.
Board CLI requires --case-dir, --output (new directory),
and --board-id (serial:<device-tree serial> or machine-id:</etc/machine-id>).
Machine-id identifies the board OS installation, not an immutable silicon ID.

Timeout policy: no reset and no freeing device-owned memory. A failed library
session remains in QUARANTINE. The CLI can release buffers only while its
original PL identity is unchanged and its original core is idle. PL identity
loss permanently quarantines that session; even Ctrl-C does not release unsafe
buffers. Terminating the process by
external force defeats that guarantee and is not a supported recovery action.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pilot_measurement", HERE / "measurement.py")
measurement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measurement)
QUARANTINE = []
GOVERNOR_QUARANTINE = []
POINTER_NAMES = ("a", "b", "weight", "bias", "output", "params")
PHASE_CONTRACT = "1_load_2_onchip_compute_3_store_repeated_tiles_0_idle"
P3T1_TIMER_MAGIC = 0x50335431
HOST_TIMING_GUARD_VERSION = "p3t1_pl_epoch_guard_v1"
RUNTIME_PROTOCOL_ID = "t6_tiled_p3t1_pl_epoch_guard_v1"
OWNERSHIP_GUARD_VERSION = "t6_shared_machine_reservation_v1"
INDIVIDUAL_CALL_NS_SCOPE = (
    "individual invoke: used-buffer flush, pointer writes, PL identity check before AP_START, "
    "start/done/idle poll, PL identity check after completion, output invalidate and capture; "
    "includes both identity checks and their software overhead; excludes entry/final identity "
    "checks, poison/idle check and counter capture")
GROUP_CALL_NS_SCOPE = (
    "group host interval: entry PL identity check + output poison + idle check + used-buffer flush "
    "+ pointer writes + PL identity check before AP_START + start + done/idle poll + PL identity "
    "check after completion + output invalidate + preallocated output copy + raw counter capture "
    "+ final PL identity check; includes all four identity checks and their software overhead")
HOST_TIMING_COMPARABILITY = (
    "guarded host timing must not be pooled with older unguarded host timing; "
    "PL compute/transfer cycles remain separate hardware intervals")
SHELL_FILES = ("top.bit", "top.hwh", "shell_metadata.json", "ROUTE_TIMING_PASS",
               "timing.rpt", "setup_paths.rpt", "hold_paths.rpt", "utilization.rpt",
               "utilization_hier.rpt", "route_status.rpt", "drc.rpt", "shell_stages.jsonl")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _child(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("artifact path must be a nonempty relative POSIX path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path escapes case directory")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact resolves outside case directory") from exc
    if result == root:
        raise ValueError("artifact must be a file inside case directory")
    return result


def _checked_files(root, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("nonempty file hashes required")
    for name, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid SHA256: " + str(name))
        target = _child(root, name)
        if not target.is_file() or _sha(target) != digest:
            raise ValueError("artifact missing/changed: " + name)


def _case_identity(root):
    case = _load(root / "case.json")
    if case.get('phase_counter_abi') != 'P3T1' or case.get('phase_contract') != PHASE_CONTRACT:
        raise ValueError('case must bind the tiled P3T1 timer and repeated tile phase contract')
    unhashed = dict(case)
    declared = unhashed.pop("case_sha256")
    if declared != _canonical(unhashed):
        raise ValueError("canonical case SHA256 differs")
    _checked_files(root, case["source_files"])
    if case["numeric_contract_sha256"] != _canonical(_load(root / "numeric_contract.json")):
        raise ValueError("numeric contract binding differs")
    fixtures = case.get("fixtures")
    if not isinstance(fixtures, list) or [row.get("fixture_id") for row in fixtures] != [f"test{i}" for i in range(5)]:
        raise ValueError("exactly ordered test0..test4 correctness fixtures required")
    for fixture in fixtures:
        _checked_files(root, fixture["files"])
        name = fixture["fixture_id"] + "_expected.bin"
        if fixture["files"].get(name) != fixture["expected_output_sha256"]:
            raise ValueError("fixture expected-output identity differs")
    return case


def _phase_identity(root, case, path, files=None):
    phase = _load(path)
    if (phase.get("schema") != "t6_integrated_phase_validation_v1" or
        phase.get("status") != "PASS_INTEGRATED_PHASE_VALIDATION" or
        phase.get("case_sha256") != case["case_sha256"] or
        phase.get("bitstream_sha256") != _sha(root / "vivado_shell/top.bit") or
        phase.get("phase_contract") != PHASE_CONTRACT or
        phase.get("phase_interface") != "ap_vld_latched" or
        phase.get("single_clock_domain") is not True or phase.get("phases_disjoint") is not True):
        raise ValueError("NEEDS_INTEGRATED_PHASE_VALIDATION: independent timer simulation is insufficient")
    sources = phase.get("source_files", {})
    required = {name: case["source_files"][name] for name in ("kernel.cpp", "config.h", "phase_timer.v")}
    if sources != required:
        raise ValueError("integrated phase validation does not bind exact kernel/config/timer sources")
    synthesis = phase.get("synthesis_report", {})
    if synthesis.get("path") != "hls_project/solution1/syn/report/pilot_kernel_csynth.xml":
        raise ValueError("integrated phase validation must bind the HLS synthesis report")
    checked = dict(sources)
    checked[synthesis["path"]] = synthesis.get("sha256")
    evidence = phase.get("evidence_files")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("integrated phase trace evidence files are required")
    checked.update(evidence)
    _checked_files(root, checked)
    if files is not None and any(files.get(name) != digest for name, digest in checked.items()):
        raise ValueError("phase-validation evidence is absent from board bundle")
    return phase, checked


def _route_identity(root):
    directory = root / "vivado_shell"
    for name in SHELL_FILES:
        if not (directory / name).is_file():
            raise ValueError("missing accepted shell artifact: " + name)
    if "board evidence NOT_RUN" not in (directory / "ROUTE_TIMING_PASS").read_text(encoding="utf-8"):
        raise ValueError("unrecognized route acceptance marker")
    metadata = _load(directory / "shell_metadata.json")
    if (metadata.get("part") != "xczu7ev-ffvc1156-2-e" or metadata.get("scope") != "TEMPLATE_COVERAGE_ONLY" or
        metadata.get("route_timing_drc") != "PASS" or metadata.get("clock_hz") != 187500000 or
        metadata.get("phase_interface") != "ap_vld_latched" or metadata.get("schema_version") != 3 or
        metadata.get("routing_errors") != 0 or metadata.get("routable_nets", 0) <= 0 or
        metadata.get("routable_nets") != metadata.get("fully_routed_nets")):
        raise ValueError("integrated route/part/clock acceptance differs")
    if metadata.get('phase_counter_abi') != 'P3T1' or metadata.get('phase_contract') != PHASE_CONTRACT:
        raise ValueError('routed shell does not bind the tiled P3T1 phase contract')
    for key in ("WNS_ns", "WHS_ns", "WPWS_ns"):
        value = float(metadata[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError("negative/nonfinite routed timing slack")
    required = dict(core_control_base=0xA0000000, timer_counts_base=0xA0010000,
                    timer_sequence_base=0xA0020000, timer_identity_base=0xA0030000,
                    gpio_data_offset=0, gpio_data2_offset=8, timer_magic=0x50335431)
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("unexpected timer/control ABI")
    if metadata.get("status_bits") != dict(valid=0, busy=1, counter_overflow=2, phase_error=3, sequence_overflow=4):
        raise ValueError("timer status ABI differs")
    _reset_wiring_identity(directory / 'top.hwh')
    return metadata


def _reset_wiring_identity(path):
    """Check every reset-controller input before any overlay programming/MMIO."""
    document = ET.parse(path)
    modules = {element.get('INSTANCE'): element for element in document.iter('MODULE')}
    def parameter(module, name):
        matches = [p.get('VALUE') for p in modules[module].findall('./PARAMETERS/PARAMETER') if p.get('NAME') == name]
        if len(matches) != 1:
            raise ValueError('missing reset/constant parameter: ' + module + '.' + name)
        return matches[0]
    def port(module, name):
        matches = [p for p in modules[module].findall('./PORTS/PORT') if p.get('NAME') == name]
        if len(matches) != 1 or not matches[0].get('SIGNAME'):
            raise ValueError('missing reset port/net: ' + module + '.' + name)
        return matches[0]
    for module in ('rst', 'ps', 'reset_high', 'reset_low', 'core', 'timer', 'ctrl', 'mem',
                   'timer_counts', 'timer_sequence', 'timer_identity'):
        if module not in modules:
            raise ValueError('reset audit needs module: ' + module)
    if int(parameter('reset_high','CONST_VAL'),0) != 1 or int(parameter('reset_low','CONST_VAL'),0) != 0:
        raise ValueError('reset constants are not high1/low0')
    high, low = port('reset_high','dout').get('SIGNAME'), port('reset_low','dout').get('SIGNAME')
    if parameter('rst','C_EXT_RESET_HIGH') != '0' or parameter('rst','C_AUX_RESET_HIGH') != '0':
        raise ValueError('reset-controller polarity differs from the reviewed active-low contract')
    ext, aux, debug = port('rst','ext_reset_in'), port('rst','aux_reset_in'), port('rst','mb_debug_sys_rst')
    if ext.get('POLARITY') != 'ACTIVE_LOW' or ext.get('SIGNAME') != port('ps','pl_resetn0').get('SIGNAME'):
        raise ValueError('external reset wiring/polarity differs')
    if aux.get('POLARITY') != 'ACTIVE_LOW' or aux.get('SIGNAME') != high:
        raise ValueError('RESET_INTERFACE_INVALID: auxiliary active-low reset must be tied inactive high')
    if debug.get('POLARITY') != 'ACTIVE_HIGH' or debug.get('SIGNAME') != low:
        raise ValueError('debug active-high reset must be tied inactive low')
    if port('rst','dcm_locked').get('SIGNAME') != high:
        raise ValueError('reset controller dcm_locked is not asserted high')
    common = port('rst','peripheral_aresetn').get('SIGNAME')
    for module,name in [('core','ap_rst_n'),('timer','aresetn'),('ctrl','aresetn'),('mem','aresetn')]+[
            (module,'s_axi_aresetn') for module in ('timer_counts','timer_sequence','timer_identity')]:
        item = port(module,name)
        if item.get('POLARITY') != 'ACTIVE_LOW' or item.get('SIGNAME') != common:
            raise ValueError('reset sink wiring/polarity differs: ' + module + '.' + name)
    return dict(status='PASS_COMPLETE_RESET_WIRING', external_reset='ps.pl_resetn0',
                auxiliary_inactive_level=1, debug_inactive_level=0, dcm_locked=1,
                hwh_sha256=_sha(path))


def prepare_bundle(case_dir, *, phase_validation="phase_validation.json", mode="timing"):
    """Offline, no hardware imports. Never overwrite any existing bundle file."""
    root = Path(case_dir).resolve()
    if mode not in ("probe", "timing"):
        raise ValueError("bundle mode must be probe or timing")
    destination = root / ("board_bundle.json" if mode == "timing" else "board_probe_bundle.json")
    if destination.exists():
        raise FileExistsError("preserve existing board bundle; use a new immutable case package")
    case = _case_identity(root)
    _route_identity(root)
    phase_files = {}
    if mode == "timing":
        phase_path = _child(root, phase_validation)
        _, phase_files = _phase_identity(root, case, phase_path)
    component = "hls_project/solution1/impl/ip/component.xml"
    if not (root / component).is_file():
        raise ValueError("exported IP component.xml required for non-guessed register ABI")
    names = set(case["source_files"]) | {"case.json", component, "hls_project/solution1/syn/report/pilot_kernel_csynth.xml"}
    if mode == "timing":
        names.add(phase_validation)
    names.update(phase_files)
    names.update("vivado_shell/" + name for name in SHELL_FILES)
    for fixture in case["fixtures"]:
        names.update(fixture["files"])
    # Validate the register contract before creating the deployable bundle.
    registers = parse_component(ET.parse(root / component).getroot())
    resolve_abi(registers, _required_arguments(case["operator"]), 65536)
    for name in ("board_runner.py", "measurement.py", "board_ownership.py"):
        target = root / name
        if target.exists() and _sha(target) != _sha(HERE / name):
            raise FileExistsError("refusing to overwrite runner source: " + str(target))
    for name in ("board_runner.py", "measurement.py", "board_ownership.py"):
        if not (root / name).exists():
            shutil.copyfile(HERE / name, root / name)
        names.add(name)
    files = {name: _sha(_child(root, name)) for name in sorted(names)}
    bundle = dict(schema="pilot_board_bundle_v1", mode=mode, case_id=case["case_id"], case_sha256=case["case_sha256"],
                  numeric_contract_sha256=case["numeric_contract_sha256"], files=files,
                  phase_validation=phase_validation if mode == "timing" else None,
                  component_xml=component, formal_cost_eligible=False)
    # Exclusive create also protects against a concurrent packaging invocation.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return bundle


def validate_bundle(case_dir, *, mode="timing"):
    root = Path(case_dir).resolve()
    bundle = _load(root / ("board_bundle.json" if mode == "timing" else "board_probe_bundle.json"))
    if (bundle.get("schema") != "pilot_board_bundle_v1" or bundle.get("formal_cost_eligible") is not False or
        mode not in ("timing", "probe") or bundle.get("mode") != mode):
        raise ValueError("unrecognized/nonpilot board bundle")
    _checked_files(root, bundle["files"])
    case = _case_identity(root)
    if (bundle.get("case_sha256") != case["case_sha256"] or bundle.get("case_id") != case["case_id"] or
        bundle.get("numeric_contract_sha256") != case["numeric_contract_sha256"]):
        raise ValueError("board bundle does not bind this case/numeric contract")
    required = set(case["source_files"]) | {"case.json", "board_runner.py", "measurement.py", "board_ownership.py",
        bundle["component_xml"], "hls_project/solution1/syn/report/pilot_kernel_csynth.xml"}
    if mode == "timing":
        required.add(bundle["phase_validation"])
    required.update("vivado_shell/" + name for name in SHELL_FILES)
    for fixture in case["fixtures"]:
        required.update(fixture["files"])
    if not required <= set(bundle["files"]):
        raise ValueError("board bundle is missing required file bindings")
    if bundle["files"]["board_runner.py"] != _sha(__file__) or bundle["files"]["measurement.py"] != _sha(measurement.__file__):
        raise ValueError("executing runner/statistics code differs from bound package")
    if bundle["files"]["board_ownership.py"] != _sha(HERE / "board_ownership.py"):
        raise ValueError("executing ownership code differs from bound package")
    metadata = _route_identity(root)
    if mode == "timing":
        _phase_identity(root, case, _child(root, bundle["phase_validation"]), bundle["files"])
    component = _child(root, bundle["component_xml"])
    registers = parse_component(ET.parse(component).getroot())
    return root, bundle, case, metadata, registers


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _text(element, name):
    values = [(child.text or "").strip() for child in element if _tag(child) == name]
    if len(values) != 1 or not values[0]:
        raise ValueError("ambiguous/missing IP-XACT " + name)
    return values[0]


def parse_component(document):
    """Read the sole control address block from namespaced IP-XACT XML."""
    candidates = []
    for block in document.iter():
        if _tag(block) != "addressBlock":
            continue
        registers = {}
        for register in block:
            if _tag(register) != "register":
                continue
            name = _text(register, "name")
            if name in registers:
                raise ValueError("duplicate IP-XACT register name")
            fields = {}
            for field in register:
                if _tag(field) == "field":
                    fname = _text(field, "name").upper()
                    if fname in fields:
                        raise ValueError("duplicate IP-XACT field name")
                    fields[fname] = dict(bit_offset=int(_text(field, "bitOffset"), 0),
                                         bit_width=int(_text(field, "bitWidth"), 0))
            registers[name] = dict(address_offset=int(_text(register, "addressOffset"), 0),
                                   size=int(_text(register, "size"), 0), fields=fields)
        if any("AP_START" in row["fields"] for row in registers.values()):
            candidates.append(registers)
    if len(candidates) != 1:
        raise ValueError("exactly one IP-XACT AP_START control block is required")
    return candidates[0]


def _required_arguments(operator):
    if operator in ("pw1x1", "dw3", "dw5", "stem_conv3", "conv3x3", "stem3x3"):
        return {"a", "weight", "bias", "output", "params"}
    if operator in ("nearest2x", "maxpool", "maxpool3", "maxpool3x3"):
        return {"a", "output"}
    if operator in ("requant_add", "requant_concat"):
        return {"a", "b", "output", "params"}
    if operator in ("body8_to_head16", "body8_to_head16_requantize", "input_requant", "requantize", "requant"):
        return {"a", "output", "params"}
    raise ValueError("operator is not registered by the board runner: " + str(operator))


def resolve_abi(registers, required_arguments, address_range):
    control = [(name, row) for name, row in registers.items() if "AP_START" in row.get("fields", {})]
    if len(control) != 1:
        raise ValueError("unambiguous AP_START register required")
    control_name, control_row = control[0]
    offset = control_row["address_offset"]
    if control_row["size"] != 32 or offset < 0 or offset % 4 or offset + 4 > address_range:
        raise ValueError("invalid control register geometry")
    masks = {}
    for name in ("AP_START", "AP_DONE", "AP_IDLE"):
        field = control_row["fields"].get(name, {})
        bit = field.get("bit_offset", -1)
        if field.get("bit_width") != 1 or not isinstance(bit, int) or not 0 <= bit < 32:
            raise ValueError("missing/non-single-bit control field: " + name)
        masks[name] = 1 << bit
    if len(set(masks.values())) != 3:
        raise ValueError("overlapping control status bits")
    # Check the whole exported register map, including interrupt registers.
    occupied = set()
    for name, row in registers.items():
        at, size = row['address_offset'], row['size']
        if size not in (32, 64) or at < 0 or at % 4 or at + size // 8 > address_range:
            raise ValueError('invalid exported register geometry: ' + name)
        locations = set(range(at, at + size // 8))
        if occupied & locations:
            raise ValueError('overlapping exported control registers')
        occupied.update(locations)
    pointers, occupied = {}, set(range(offset, offset + 4))
    for argument in POINTER_NAMES:
        # Vitis 2022.2 IP-XACT exports a 64-bit pointer as two named 32-bit
        # registers, *_1 then *_2. The Verilog keyword output is output_r.
        aliases = (argument, 'output_r') if argument == 'output' else (argument,)
        matches = [(name, row, alias) for alias in aliases for name, row in registers.items()
                   if name.lower() in (alias, alias + '_1')]
        if len(matches) > 1:
            raise ValueError("ambiguous pointer register: " + argument)
        if not matches:
            if argument in required_arguments:
                raise ValueError("required pointer register missing: " + argument)
            continue
        name, row, alias = matches[0]
        offset = row["address_offset"]
        parts = [dict(name=name, offset=offset, size=row['size'])]
        if row['size'] == 32 and name.lower() == alias + '_1':
            high = [(n, v) for n, v in registers.items() if n.lower() == alias + '_2']
            if len(high) != 1 or high[0][1]['size'] != 32 or high[0][1]['address_offset'] != offset + 4:
                raise ValueError('split pointer needs its contiguous high 32-bit register: ' + argument)
            for n, part in [(name, row), high[0]]:
                if part.get('fields') != {alias.upper(): dict(bit_offset=0, bit_width=32)}:
                    raise ValueError('split pointer field identity/width differs: ' + n)
            parts.append(dict(name=high[0][0], offset=offset + 4, size=32))
        elif row['size'] != 64:
            raise ValueError('pointer is neither one 64-bit nor two verified 32-bit registers: ' + argument)
        if offset < 0 or offset % 4 or offset + 8 > address_range:
            raise ValueError("pointer must have a bound contiguous 64-bit register: " + argument)
        locations = set(range(offset, offset + 8))
        if occupied & locations:
            raise ValueError("overlapping pointer registers")
        occupied.update(locations)
        pointers[argument] = dict(register=name, offset=offset, high_offset=offset + 4, registers=parts,
                                 encoding='low32_then_high32')
    return dict(control_name=control_name, control_offset=control_row["address_offset"], masks=masks, pointers=pointers)


def verified_board_identity(requested):
    options = []
    for filename in ("/sys/firmware/devicetree/base/serial-number", "/proc/device-tree/serial-number"):
        path = Path(filename)
        if path.is_file():
            value = path.read_bytes().rstrip(b"\x00\r\n").decode("ascii", errors="strict").strip()
            if value:
                options.append(dict(board_id="serial:" + value, evidence_path=filename,
                                    evidence_sha256=_sha(path), identity_scope="device_tree_serial"))
    path = Path("/etc/machine-id")
    if path.is_file():
        value = path.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-fA-F]{32}", value):
            options.append(dict(board_id="machine-id:" + value, evidence_path=str(path),
                                evidence_sha256=_sha(path), identity_scope="board_OS_installation_not_silicon"))
    found = [option for option in options if option["board_id"] == requested]
    if not found:
        raise ValueError("--board-id does not match local device-tree serial or machine-id")
    return found[0]


def _utc():
    return datetime.now(timezone.utc).isoformat()


class CpuGovernor:
    """Change only present cpufreq policies and restore their exact prior values."""
    def __init__(self):
        self.records, self.changed, self.deviations = [], [], []
        paths = sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"))
        if not paths:
            self.deviations.append("CPU_GOVERNOR_INTERFACE_UNAVAILABLE")
        for path in paths:
            row = dict(path=str(path))
            try:
                original = path.read_text(encoding="ascii").strip()
                row["previous"] = original
                available = (path.parent / "scaling_available_governors").read_text(encoding="ascii").split()
                row["available"] = available
                if "performance" not in available:
                    row["condition_deviation"] = "PERFORMANCE_GOVERNOR_NOT_AVAILABLE"
                elif original != "performance":
                    path.write_text("performance", encoding="ascii")
                    self.changed.append((path, original))
                row["actual"] = path.read_text(encoding="ascii").strip()
                if row["actual"] != "performance":
                    self.deviations.append(str(path) + ":NOT_PERFORMANCE")
            except OSError as exc:
                row["condition_deviation"] = type(exc).__name__ + ": " + str(exc)
                self.deviations.append(str(path) + ":READ_OR_SET_FAILED")
            self.records.append(row)

    def snapshot(self):
        readings = []
        for row in self.records:
            current = dict(row)
            try:
                current["actual"] = Path(row["path"]).read_text(encoding="ascii").strip()
            except OSError:
                current["actual"] = "UNAVAILABLE"
            readings.append(current)
        label = "performance" if readings and all(row.get("actual") == "performance" for row in readings) else "UNAVAILABLE"
        # Raw current/min/max frequencies provide evidence, never a fabricated fixed-clock claim.
        frequency = {}
        for row in self.records:
            folder = Path(row["path"]).parent
            for name in ("scaling_cur_freq", "scaling_min_freq", "scaling_max_freq", "cpuinfo_cur_freq",
                         "scaling_available_frequencies"):
                try:
                    frequency[str(folder / name)] = (folder / name).read_text(encoding="ascii").strip()
                except OSError:
                    pass
        return dict(cpu_governor=label, policies=readings, frequency_raw_khz=frequency,
                    condition_deviations=list(self.deviations),
                    host_interpretation="NORMAL" if label == "performance" else "LIMITED_CPU_CONTROL_UNAVAILABLE")

    def restore(self):
        results = []
        for path, original in reversed(self.changed):
            try:
                path.write_text(original, encoding="ascii")
                actual = path.read_text(encoding="ascii").strip()
                results.append(dict(path=str(path), requested=original, actual=actual,
                                    status="RESTORED" if actual == original else "FAILED"))
            except OSError as exc:
                results.append(dict(path=str(path), requested=original, status="FAILED", error=str(exc)))
        return results


def temperature_snapshot():
    records = []
    paths = list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
    paths += list(Path("/sys/class/hwmon").glob("hwmon*/temp*_input"))
    for path in sorted(paths):
        row = dict(path=str(path))
        try:
            raw = path.read_text(encoding="ascii").strip()
            value = float(raw) / 1000.0
            label_path = path.parent / ("type" if path.name == "temp" else path.name.replace("_input", "_label"))
            row.update(raw_millidegrees_c=raw, temperature_c=value,
                       label=label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else path.name)
            row["valid"] = math.isfinite(value) and -40 < value < 150
        except (OSError, ValueError) as exc:
            row.update(valid=False, error=str(exc))
        records.append(row)
    # Zynq XADC/SYSMON commonly exposes temperature through IIO, not thermal_zone.
    for device in sorted(Path("/sys/bus/iio/devices").glob("iio:device*")):
        for path in sorted(device.glob("in_temp*_raw")):
            row = dict(path=str(path), source="IIO_RAW")
            try:
                prefix = path.name[:-4]
                scale_path = device / (prefix + "_scale")
                offset_path = device / (prefix + "_offset")
                if not scale_path.is_file():
                    scale_path = device / "in_temp_scale"
                if not offset_path.is_file():
                    offset_path = device / "in_temp_offset"
                raw = path.read_text(encoding="ascii").strip()
                scale = scale_path.read_text(encoding="ascii").strip()
                offset = offset_path.read_text(encoding="ascii").strip() if offset_path.is_file() else "0"
                value = (float(raw) + float(offset)) * float(scale) / 1000.0
                label_path = device / (prefix + "_label")
                row.update(raw=raw, offset=offset, scale=scale, scale_path=str(scale_path),
                    offset_path=str(offset_path) if offset_path.is_file() else None,
                    offset_default_zero=not offset_path.is_file(), formula_c="(raw + offset) * scale / 1000",
                    label=label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else
                        (device / "name").read_text(encoding="utf-8").strip() if (device / "name").is_file() else device.name,
                    temperature_c=value if math.isfinite(value) else None,
                    valid=math.isfinite(value) and -40 < value < 150)
            except (OSError, ValueError) as exc:
                row.update(valid=False, error=str(exc))
            records.append(row)
        for path in sorted(device.glob("in_temp*_input")):
            row = dict(path=str(path), source="IIO_PROCESSED")
            try:
                raw = path.read_text(encoding="ascii").strip()
                value = float(raw) / 1000.0
                row.update(raw_millidegrees_c=raw, temperature_c=value if math.isfinite(value) else None,
                    label=(device / "name").read_text(encoding="utf-8").strip() if (device / "name").is_file() else device.name,
                    valid=math.isfinite(value) and -40 < value < 150)
            except (OSError, ValueError) as exc:
                row.update(valid=False, error=str(exc))
            records.append(row)
    valid = [row["temperature_c"] for row in records if row["valid"]]
    pl = [row["temperature_c"] for row in records if row["valid"] and row.get("label", "").lower() == "pl_temp"]
    return dict(temperature_c=max(pl) if pl else max(valid) if valid else None, sensors=records,
                selection="pl_temp_label" if pl else "maximum_valid_local_sensor",
                condition_deviations=[] if valid else ["TEMPERATURE_SENSOR_UNAVAILABLE"])


class UnsafeDeviceState(RuntimeError):
    pass


class PLIdentityLost(UnsafeDeviceState):
    """The original core can no longer be observed to own or release buffers."""
    pass


class BoardSession:
    def __init__(self, root, case, metadata, registers, timeout, progress=None, identity_record=None,
                 ownership_guard=None):
        if not callable(ownership_guard):
            raise RuntimeError("verified shared board reservation is required before hardware access")
        ownership_guard("session_entry")
        emit = progress or (lambda event: None)
        emit('pynq_import_before')
        import numpy as np
        from pynq import Overlay, allocate, MMIO, Clocks, PL
        emit('pynq_import_after')
        self.np, self.clocks, self.pl, self.case, self.metadata = np, Clocks, PL, case, metadata
        self.owned, self.buffers, self.pending, self.closed = [], {}, False, False
        self.pl_identity_lost = False
        self.device_state = "ORIGINAL_PL_IDENTITY_BOUND"
        self.pl_identity_checks = 0
        self.identity_record = identity_record or (lambda value: None)
        self.expected_bitfile_name = str((root / "vivado_shell/top.bit").resolve())
        self.timeout_ns = int(timeout * 1e9)
        self.required = _required_arguments(case["operator"])
        emit('overlay_parse_before')
        self.overlay = Overlay(self.expected_bitfile_name, download=False)
        emit('overlay_parse_after')
        info = self._ip("core", "hwnas.org:hls:pilot_kernel:", metadata["core_control_base"])
        self.abi = resolve_abi(registers, self.required, int(info["addr_range"]))
        # If HWH includes a register table, it must agree with the IP-XACT export.
        hwh_registers = info.get("registers", {})
        for pointer in self.abi["pointers"].values():
            if hwh_registers:
                for part in pointer['registers']:
                    row = hwh_registers.get(part['name'])
                    if not row or int(row["address_offset"]) != part['offset'] or int(row["size"]) != part['size']:
                        raise ValueError("HWH and IP-XACT pointer ABI disagree")
        timer_infos = {}
        for name, key in (("timer_counts", "timer_counts_base"), ("timer_sequence", "timer_sequence_base"),
                          ("timer_identity", "timer_identity_base")):
            timer_infos[name] = self._ip(name, "xilinx.com:ip:axi_gpio:", metadata[key])
        emit('overlay_download_before')
        ownership_guard("before_overlay_download")
        self.overlay.download()
        emit('overlay_download_after')
        self.downloaded_at_utc = _utc()
        self.independence_id = "reload_bitstream:" + str(time.time_ns())
        self.clock_hz = self._clock()
        self.core = MMIO(int(info["phys_addr"]), int(info["addr_range"]))
        self.gpios = {name: MMIO(int(row["phys_addr"]), int(row["addr_range"])) for name, row in timer_infos.items()}
        emit('first_timer_identity_mmio_read_before')
        self.pl_identity = self._read_pl_identity()
        emit('first_timer_identity_mmio_read_after')
        self.identity_record(self.identity_observation("overlay_download_after"))
        if (metadata["timer_magic"] != P3T1_TIMER_MAGIC or not self.pl_identity["stable"] or
            self.pl_identity["timer_magic"] != P3T1_TIMER_MAGIC or
            self.pl_identity["bitfile_name"] != self.expected_bitfile_name or not self.pl_identity["timestamp"]):
            self.device_state = "INITIAL_PL_IDENTITY_NOT_BOUND"
            self.identity_record(self.identity_observation("overlay_download_after"))
            raise RuntimeError("downloaded P3T1 magic or actual PYNQ PL bitfile/timestamp identity mismatch")
        emit('first_core_idle_mmio_read_before')
        if not self.idle() or self.gpios["timer_identity"].read(8) != 0 or self.gpios["timer_sequence"].read(8) != 0:
            raise RuntimeError("newly loaded core/timer did not reset idle")
        emit('first_core_idle_mmio_read_after')
        self.sequence = 0
        self.fixtures = []
        input_bits, output_bits = case["input_bits"], case["output_bits"]
        if input_bits not in (8, 16) or output_bits not in (8, 16):
            raise ValueError("unsupported signed activation width")
        input_dtype = np.dtype("i1" if input_bits == 8 else "<i2")
        output_dtype = np.dtype("i1" if output_bits == 8 else "<i2")
        weight_bits = case.get("weight_bits") or input_bits
        if weight_bits not in (8, 16):
            raise ValueError("unsupported weight width")
        weight_dtype = np.dtype("i1" if weight_bits == 8 else "<i2")
        input_count = math.prod(case["shape"]["input_shapes"][0])
        output_count = math.prod(case["shape"]["output_shape"])
        dtypes = dict(a=input_dtype, b=input_dtype, weight=weight_dtype, bias=np.dtype("<i4"),
                      params=np.dtype("<i4"), output=output_dtype)
        suffixes = dict(a="a", b="b", weight="w", bias="bias", params="params", output="expected")
        try:
            for fixture in case["fixtures"]:
                arrays = {}
                for name, suffix in suffixes.items():
                    path = root / (fixture["fixture_id"] + "_" + suffix + ".bin")
                    if path.stat().st_size % dtypes[name].itemsize:
                        raise ValueError("fixture bytes do not fit declared dtype")
                    arrays[name] = np.fromfile(path, dtype=dtypes[name])
                    if not arrays[name].size:
                        raise ValueError("empty fixture array")
                if (arrays["a"].size != input_count or arrays["output"].size != output_count or
                    arrays["params"].size != 8 or fixture["expected_elements"] != output_count):
                    raise ValueError("fixture length differs from case tensor geometry")
                if "b" in self.required and arrays["b"].size != math.prod(case["shape"]["input_shapes"][1]):
                    raise ValueError("second input geometry differs")
                if "weight" in self.required:
                    cout = case["shape"]["output_shape"][0]
                    k = case["shape"].get("kernel", 1)
                    ci = 1 if case["operator"].startswith("dw") else case["shape"]["input_shapes"][0][0]
                    if arrays["weight"].size != cout * ci * k * k or arrays["bias"].size != cout:
                        raise ValueError("convolution parameters differ from kernel geometry")
                # Every element differs by one bit, preventing stale-output parity.
                arrays["poison"] = np.bitwise_xor(arrays["output"], np.array(1, dtype=output_dtype))
                self.fixtures.append((fixture, arrays))
            for name in POINTER_NAMES:
                shapes = {arrays[name].shape for _, arrays in self.fixtures}
                if len(shapes) != 1:
                    raise ValueError("fixture buffers differ in size")
                buffer = allocate(shape=next(iter(shapes)), dtype=dtypes[name])
                self.owned.append(buffer)
                self.buffers[name] = buffer
            # Maximum adaptive group size. These arrays are allocated before timing.
            self.captured_outputs = np.empty((16, output_count), dtype=output_dtype)
            emit('buffers_allocated_ready')
        except BaseException:
            self.close()
            raise

    def _ip(self, name, prefix, address):
        rows = [row for key, row in self.overlay.ip_dict.items() if key == name or key.endswith("/" + name)]
        if len(rows) != 1 or not rows[0].get("type", "").startswith(prefix) or int(rows[0]["phys_addr"]) != address:
            raise ValueError("HWH core/timer identity or address differs: " + name)
        return rows[0]

    def _clock(self):
        mhz = float(self.clocks.fclk0_mhz)
        implemented_mhz = float(self.metadata["clock_hz"]) / 1000000.0
        if not math.isfinite(mhz) or mhz <= 0 or mhz > implemented_mhz + 0.05:
            raise RuntimeError("actual PL0 clock is invalid or exceeds the implemented PL0 constraint")
        value = mhz * 1000000.0
        if hasattr(self, "clock_hz") and value != self.clock_hz:
            raise RuntimeError("actual PL0 clock changed during board session")
        return value

    def _read_pl_identity(self):
        # Bracket the MMIO read with actual global PYNQ PL state. A mixed
        # snapshot during a reload must never be accepted as a stable epoch.
        first = (str(self.pl.bitfile_name or ""), str(self.pl.timestamp or ""))
        magic = int(self.gpios["timer_identity"].read(0))
        second = (str(self.pl.bitfile_name or ""), str(self.pl.timestamp or ""))
        return dict(bitfile_name=second[0], timestamp=second[1], timer_magic=magic,
                    stable=first == second, first_bitfile_name=first[0], first_timestamp=first[1])

    def identity_observation(self, stage, observed=None):
        return dict(schema="template_pl_identity_guard_v1", device_state=self.device_state,
            stage=stage, observed_utc=_utc(), original_identity=dict(self.pl_identity),
            observed_identity=observed, successful_checks=self.pl_identity_checks,
            identity_lost_permanent=self.pl_identity_lost,
            exclusion_scope="OBSERVED_PYNQ_PL_EPOCH_AND_P3T1_MAGIC_NOT_GLOBAL_MUTEX_PROOF")

    def _assert_pl_identity(self, stage):
        if self.pl_identity_lost:
            raise PLIdentityLost("PL_IDENTITY_LOST: original buffers remain quarantined")
        observed, reason = None, None
        try:
            observed = self._read_pl_identity()
            if (not observed["stable"] or observed["timer_magic"] != P3T1_TIMER_MAGIC or
                observed["bitfile_name"] != self.pl_identity["bitfile_name"] or
                observed["timestamp"] != self.pl_identity["timestamp"]):
                reason = "P3T1 magic or actual PYNQ PL bitfile/timestamp changed"
        except BaseException:
            reason = "original PL identity cannot be read"
        if reason is not None:
            self.pl_identity_lost, self.pending = True, True
            self.device_state = "PL_IDENTITY_LOST"
            if self not in QUARANTINE:
                QUARANTINE.append(self)
            # Persist only on load/loss, never on every timed invocation.
            self.identity_record(self.identity_observation(stage, observed))
            raise PLIdentityLost("PL_IDENTITY_LOST: " + reason + "; original buffers remain quarantined")
        self.pl_identity_checks += 1
        return observed

    def idle(self):
        return bool(self.core.read(self.abi["control_offset"]) & self.abi["masks"]["AP_IDLE"])

    def select_fixture(self, index):
        self._assert_pl_identity("select_fixture_before")
        fixture, arrays = self.fixtures[index]
        if not self.idle():
            self.pending = True
            raise UnsafeDeviceState("core is active before fixture update")
        for name in POINTER_NAMES:
            if name != "output":
                self.buffers[name][:] = arrays[name]
        self.current = fixture, arrays

    def invoke(self, capture, *, validate_phase=True):
        """Return raw timing; comparisons and output hashes are intentionally outside."""
        self._assert_pl_identity("invoke_before")
        if self.closed or self.pending or not self.idle():
            self.pending = True
            raise UnsafeDeviceState("core is not idle before invocation")
        fixture, arrays = self.current
        self.buffers["output"][:] = arrays["poison"]
        before = self.gpios["timer_sequence"].read(8)
        if validate_phase and (before != self.sequence or self.gpios["timer_identity"].read(8) & 2):
            raise RuntimeError("timer sequence/busy drift before invocation")
        if validate_phase and before == 0xFFFFFFFF:
            raise RuntimeError("timer sequence exhausted; a new reset session is required")
        started = time.perf_counter_ns()
        try:
            for name in sorted(self.required):
                self.buffers[name].flush()
            for name, entry in self.abi["pointers"].items():
                address = int(self.buffers[name].physical_address)
                if not 0 <= address <= 0xFFFFFFFFFFFFFFFF:
                    raise ValueError("buffer physical address outside 64-bit ABI")
                self.core.write(entry["offset"], address & 0xFFFFFFFF)
                self.core.write(entry["high_offset"], address >> 32)
            self._assert_pl_identity("invoke_before_start")
            self.pending = True  # Set before write: an exception can follow a successful bus write.
            self.core.write(self.abi["control_offset"], self.abi["masks"]["AP_START"])
            saw_done = False
            while True:
                control = self.core.read(self.abi["control_offset"])
                saw_done = saw_done or bool(control & self.abi["masks"]["AP_DONE"])
                if saw_done and control & self.abi["masks"]["AP_IDLE"]:
                    self._assert_pl_identity("invoke_after_core_completion")
                    self.pending = False
                    break
                if time.perf_counter_ns() - started > self.timeout_ns:
                    self._assert_pl_identity("invoke_timeout")
                    raise UnsafeDeviceState("core completion timed out; buffers remain device-owned")
            self.buffers["output"].invalidate()
            self.np.copyto(capture, self.buffers["output"])
            call_ns = time.perf_counter_ns() - started
        except BaseException:
            self.last_call_ns = time.perf_counter_ns() - started
            self._assert_pl_identity("invoke_exception")
            raise
        self.last_call_ns = call_ns
        deadline = time.perf_counter_ns() + self.timeout_ns
        while True:
            status = self.gpios["timer_identity"].read(8)
            sequence = self.gpios["timer_sequence"].read(8)
            if not status & 2 or not validate_phase:
                break
            if time.perf_counter_ns() > deadline:
                self._assert_pl_identity("phase_commit_timeout")
                raise RuntimeError("phase timer did not commit after core completion")
        if validate_phase and (status != 1 or sequence != before + 1):
            raise RuntimeError("timer invalid/error or sequence did not increment exactly once")
        compute = self.gpios["timer_counts"].read(0)
        transfer = self.gpios["timer_counts"].read(8)
        total = self.gpios["timer_sequence"].read(0)
        if validate_phase and (compute <= 0 or transfer <= 0 or total != compute + transfer or
            self.gpios["timer_sequence"].read(8) != sequence or self.gpios["timer_identity"].read(8) != 1):
            raise RuntimeError("timer counters are zero, overlapping, or changed during capture")
        self._assert_pl_identity("invoke_after")
        self.sequence = sequence
        compute_ns, transfer_ns = compute * 1e9 / self.clock_hz, transfer * 1e9 / self.clock_hz
        if validate_phase and call_ns < compute_ns + transfer_ns:
            raise RuntimeError("host call interval does not contain the PL phase intervals")
        return dict(compute_ns=compute_ns, transfer_ns=transfer_ns, call_ns=call_ns,
                    compute_cycles=compute, transfer_cycles=transfer, total_active_cycles=total,
                    sequence=sequence, sequence_before=before, timer_status=status,
                    phase_counts_verified=validate_phase, clock_hz=self.clock_hz,
                    pl_identity_verified=True, pl_identity_epoch=self.pl_identity["timestamp"],
                    host_timing_guard_version=HOST_TIMING_GUARD_VERSION,
                    runtime_protocol_id=RUNTIME_PROTOCOL_ID, call_ns_scope=INDIVIDUAL_CALL_NS_SCOPE)

    def compare(self, capture):
        expected = self.current[1]["output"]
        actual = self.np.asarray(capture)
        return dict(compared_elements=int(expected.size), mismatched_elements=int(self.np.count_nonzero(actual != expected)),
                    observed_output_sha256=hashlib.sha256(actual.tobytes(order="C")).hexdigest())

    def close(self):
        if self.closed:
            return
        try:
            self._assert_pl_identity("close_before")
            safe = self.idle()
            self._assert_pl_identity("close_after_idle")
        except PLIdentityLost:
            raise
        except BaseException:
            safe = False
        if not safe:
            self.pending = True
            if self not in QUARANTINE:
                QUARANTINE.append(self)
            raise UnsafeDeviceState("cannot prove core idle; session/buffers quarantined")
        self.pending = False
        for buffer in reversed(self.owned):
            buffer.freebuffer()
        self.owned.clear()
        self.closed = True
        if self in QUARANTINE:
            QUARANTINE.remove(self)


def _new_output(case_dir, output, timeout):
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("timeout must be in (0, 3600] seconds")
    output = Path(output).resolve()
    try:
        output.relative_to(Path(case_dir).resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output must be outside the immutable case package")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _failure(output, session, exc):
    failure = dict(status="FAIL", error_type=type(exc).__name__, error=str(exc),
        completed_samples_are_preserved=True, formal_cost_eligible=False)
    if session is not None:
        failure["device_state"] = session.device_state
    _save(output / "failure.json", failure)
    if session is not None:
        try:
            session.close()
        except UnsafeDeviceState as unsafe:
            failure.update(device_state=session.device_state, retention_error_type=type(unsafe).__name__,
                           retention_reason=str(unsafe))
            _save(output / "failure.json", failure)
            _save(output / "device_buffer_retention.json", dict(status="QUARANTINED_DO_NOT_TERMINATE_PROCESS",
                reason=str(unsafe), device_state=session.device_state,
                pl_identity=session.identity_observation("failure_retention"),
                reset_performed=False, buffers_freed=False))
            raise unsafe from exc


def _stage(output, started, success, stage="board_test"):
    with (output / "board_stages.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(stage=stage, wall_seconds=time.monotonic() - started,
                                    returncode=0 if success else 1, status="PASS" if success else "FAIL")) + "\n")


def _progress(output, event):
    # Persist boundaries before operations that can block inside the kernel.
    # This file I/O runs only during setup, outside all inference timing windows.
    with (Path(output) / 'setup_progress.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=event,observed_utc=_utc(),pid=os.getpid()))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def _correctness(session, case, directory, *, validate_phase):
    records = []
    for index in range(5):
        session.select_fixture(index)
        session._clock()
        fixture = case["fixtures"][index]
        capture = session.captured_outputs[0]
        timing = session.invoke(capture, validate_phase=validate_phase)
        session._clock()
        parity = session.compare(capture)
        # Probe counters are observations only, never admitted compute/transfer latency.
        if not validate_phase:
            timing = {("unvalidated_" + key if key in ("compute_ns", "transfer_ns", "call_ns") else key): value
                      for key, value in timing.items()}
        records.append(dict(fixture_id=fixture["fixture_id"], **timing, **parity,
                            expected_output_sha256=fixture["expected_output_sha256"]))
        (directory / (fixture["fixture_id"] + "_board.bin")).write_bytes(capture.tobytes())
        _save(directory / "correctness_fixtures.json", records)
        if parity["mismatched_elements"] or parity["observed_output_sha256"] != fixture["expected_output_sha256"]:
            raise RuntimeError("full fixture parity failed: " + fixture["fixture_id"])
    return records


def _record_group(session, base, batch_id, phase, first, count, group_id, writer, stream, counters):
    """One host interval; copies inside, comparisons/hashes/log I/O outside."""
    session._clock()
    timings = [None] * count
    failure, attempted = None, 0
    started = time.perf_counter_ns()
    try:
        for index in range(count):
            attempted = index + 1
            timings[index] = session.invoke(session.captured_outputs[index])
    except BaseException as exc:
        failure = exc
    group_total_ns = time.perf_counter_ns() - started
    # Every attempted invocation, including a failed/incomplete one, is retained.
    rows, parity_failed = [], False
    for index in range(attempted):
        row = {key: "" for key in measurement.CSV_FIELDS}
        row.update(base, batch_id=batch_id, phase=phase, iteration=first + index, group_id=group_id,
                   group_size=attempted, group_call_total_ns=group_total_ns, call_ns=group_total_ns / attempted)
        timing = timings[index]
        if timing is not None:
            parity = session.compare(session.captured_outputs[index])
            row.update(compute_ns=timing["compute_ns"], transfer_ns=timing["transfer_ns"], **parity)
            counters.write(json.dumps(dict(batch_id=batch_id, phase=phase, iteration=first + index,
                group_id=group_id, planned_group_size=count, **timing), allow_nan=False) + "\n")
            parity_failed |= bool(row["mismatched_elements"]) or row["observed_output_sha256"] != row["expected_output_sha256"]
        else:
            counters.write(json.dumps(dict(batch_id=batch_id, phase=phase, iteration=first + index,
                group_id=group_id, planned_group_size=count, status="INCOMPLETE_ATTEMPT",
                error=repr(failure), observed_host_interval_ns=getattr(session, "last_call_ns", None))) + "\n")
        writer.writerow(row)
        rows.append(row)
    counters.flush()
    stream.flush()
    if failure is not None:
        raise failure
    session._clock()
    if parity_failed:
        raise RuntimeError("timing-output parity failed; entire group retained")
    return group_total_ns / attempted


def run_probe(case_dir, output, board_id, *, timeout=30.0, ownership_guard=None):
    """Five fixture correctness calls to collect phase evidence; no latency acceptance."""
    if not callable(ownership_guard):
        raise RuntimeError('shared board reservation is required for a probe')
    ownership_guard('probe_entry')
    output = _new_output(case_dir, output, timeout)
    started, session, success = time.monotonic(), None, False
    try:
        root, bundle, case, metadata, registers = validate_bundle(case_dir, mode="probe")
        identity = verified_board_identity(board_id)
        _save(output / "board_identity.json", identity)
        before_temperature = temperature_snapshot()
        session = BoardSession(root, case, metadata, registers, timeout,
                               progress=lambda event: _progress(output,event),
                               ownership_guard=ownership_guard,
                               identity_record=lambda value: _save(output / "pl_identity.json", value))
        _progress(output,'correctness_fixtures_before')
        records = _correctness(session, case, output, validate_phase=False)
        _progress(output,'correctness_fixtures_after')
        summary_path = root / "hls_project/solution1/syn/report/pilot_kernel_csynth.xml"
        document = ET.parse(summary_path).getroot()
        latency = [{_tag(child): (child.text or "").strip() for child in element}
                   for element in document.iter() if _tag(element) == "SummaryOfOverallLatency"]
        result = dict(status="PASS_BOARD_CORRECTNESS_PHASE_VALIDATION_PENDING", case_id=case["case_id"],
            case_sha256=case["case_sha256"], board_id=board_id, clock_hz=session.clock_hz,
            clock_hz_end=session._clock(), correctness_fixtures_passed=len(records),
            phase_validation="PENDING_REVIEW_OF_INTEGRATED_OBSERVATIONS_AND_HLS_RTL",
            latency_claim="NONE", compute_scope="NEEDS_INTEGRATED_PHASE_VALIDATION", formal_cost_eligible=False,
            board_probe_bundle_sha256=_sha(root / "board_probe_bundle.json"),
            bitstream_sha256=bundle["files"]["vivado_shell/top.bit"], timer_magic=metadata["timer_magic"],
            actual_pl_identity=session.pl_identity,
            ownership_guard_version=OWNERSHIP_GUARD_VERSION,
            runtime_protocol_id=RUNTIME_PROTOCOL_ID, host_timing_guard_version=HOST_TIMING_GUARD_VERSION,
            register_abi=session.abi, hls_synthesis_report_sha256=_sha(summary_path),
            hls_overall_latency_raw=latency, temperature_start=before_temperature,
            temperature_end=temperature_snapshot(), source_files={name:case["source_files"][name]
                for name in ("kernel.cpp", "config.h", "phase_timer.v")})
        session.close()
        _save(output / "probe_summary.json", result)
        success = True
        return result
    except BaseException as exc:
        _failure(output, session, exc)
        raise
    finally:
        _stage(output, started, success, stage="board_probe")


def run_case(case_dir, output, board_id, *, timeout=30.0, ownership_guard=None):
    """Three reload batches; five fixtures, 50 warmups and >=1000 trials each."""
    if not callable(ownership_guard):
        raise RuntimeError('shared board reservation is required for timing')
    ownership_guard('timing_entry_before_cpu_setup')
    output = _new_output(case_dir, output, timeout)
    started, session, governor, success = time.monotonic(), None, None, False
    try:
        root, bundle, case, metadata, registers = validate_bundle(case_dir)
        identity = verified_board_identity(board_id)
        _save(output / "board_identity.json", identity)
        governor = CpuGovernor()
        _save(output / "cpu_governor_setup.json", governor.snapshot())
        fixture = case["fixtures"][0]
        base = dict(case_id=case["case_id"], case_sha256=case["case_sha256"],
                    numeric_contract_sha256=case["numeric_contract_sha256"],
                    bitstream_sha256=bundle["files"]["vivado_shell/top.bit"], board_id=board_id,
                    expected_output_sha256=fixture["expected_output_sha256"])
        manifest = dict(cases=[dict(**base, expected_elements=fixture["expected_elements"], clock_hz=metadata["clock_hz"])],
                        batches=[], independent_batches_minimum=3, warmup_minimum=50, measured_minimum=1000,
                        cv_limit=0.05, host_median_relative_limit=None, protocol_status="DRAFT",
                        phase_clock_evidence_verified=False, actual_clock_register_evidence_verified=False,
                        independent_clock_scale_calibration="NOT_RUN")
        manifest.update(runtime_protocol_id=RUNTIME_PROTOCOL_ID, host_timing_guard_version=HOST_TIMING_GUARD_VERSION,
            ownership_guard_version=OWNERSHIP_GUARD_VERSION, ownership_checks_scope="outside timed groups, before every PL reload",
            individual_call_ns_scope=INDIVIDUAL_CALL_NS_SCOPE, group_call_ns_scope=GROUP_CALL_NS_SCOPE,
            individual_invoke_identity_checks_inside_call_ns=2,
            invocation_identity_checks_inside_group_total_ns=4,
            fixture_selection_scope="outside timed warmup/measured groups",
            host_timing_comparability=HOST_TIMING_COMPARABILITY)
        _save(output / "timing_manifest.json", manifest)
        _save(output / "session.json", dict(**base, board_bundle_sha256=_sha(root / "board_bundle.json"),
            ownership_guard_version=OWNERSHIP_GUARD_VERSION,
            ownership_checks_scope="outside timed groups, before every PL reload",
            phase_validation_sha256=bundle["files"][bundle["phase_validation"]],
            actual_clock_source="pynq.Clocks.fclk0_mhz_before_after_each_group_and_batch",
            clock_evidence_scope="PS clock-register-derived rate and stable reads; no independent physical-frequency calibration",
            independent_clock_scale_calibration="NOT_RUN",
            runtime_protocol_id=RUNTIME_PROTOCOL_ID, host_timing_guard_version=HOST_TIMING_GUARD_VERSION,
            individual_call_ns_scope=INDIVIDUAL_CALL_NS_SCOPE, group_call_ns_scope=GROUP_CALL_NS_SCOPE,
            call_scope=GROUP_CALL_NS_SCOPE,
            fixture_selection_scope="outside timed warmup/measured groups",
            host_timing_comparability=manifest["host_timing_comparability"],
            excludes="DMA/array allocation, input fixture preparation, artifact hashing, comparisons, output hashing, log I/O, reprogramming",
            compute_scope="validated phase 2 on-chip interval", transfer_scope="validated phase 1 load plus phase 3 store intervals",
            group_selection="50 N=1 warmups; if host median <100000 ns, measured 63 complete N=16 groups (1008 calls); otherwise 1000 N=1 groups",
            protocol_status="DRAFT", formal_cost_eligible=False))
        csv_path = output / "timing_raw.csv"
        with csv_path.open("x", encoding="utf-8", newline="") as stream, \
                (output / "counter_raw.jsonl").open("x", encoding="utf-8") as counters:
            writer = csv.DictWriter(stream, fieldnames=measurement.CSV_FIELDS)
            writer.writeheader()
            stream.flush()
            for batch_index in range(3):
                batch_id = "batch" + str(batch_index + 1)
                folder = output / batch_id
                folder.mkdir(exist_ok=False)
                environment_start = dict(cpu=governor.snapshot(), temperature=temperature_snapshot(), timestamp_utc=_utc())
                session = BoardSession(root, case, metadata, registers, timeout,
                                       progress=lambda event: _progress(folder,event),
                                       ownership_guard=ownership_guard,
                                       identity_record=lambda value: _save(folder / "pl_identity.json", value))
                batch = dict(case_id=case["case_id"], batch_id=batch_id, independence_event="reload_bitstream",
                    independence_id=session.independence_id, clock_hz=session.clock_hz, clock_hz_end=None,
                    cpu_governor=environment_start["cpu"]["cpu_governor"],
                    temperature_c=environment_start["temperature"]["temperature_c"], temperature_c_end=None,
                    started_utc=session.downloaded_at_utc, finished_utc=None, protocol_status="DRAFT")
                batch["actual_pl_identity"] = session.pl_identity
                manifest["batches"].append(batch)
                _save(output / "timing_manifest.json", manifest)
                _save(folder / "register_abi.json", session.abi)
                _save(folder / "environment_start.json", environment_start)
                _correctness(session, case, folder, validate_phase=True)
                session.select_fixture(0)
                row_base = dict(**base, clock_hz=session.clock_hz)
                warmups = []
                for iteration in range(50):
                    warmups.append(_record_group(session, row_base, batch_id, "warmup", iteration, 1,
                                                "warmup_" + str(iteration), writer, stream, counters))
                warmup_median = statistics.median(warmups)
                group_size = 16 if warmup_median < 100000 else 1
                measured_calls = ((1000 + group_size - 1) // group_size) * group_size
                _save(folder / "group_selection.json", dict(warmup_rows=50, warmup_host_median_ns=warmup_median,
                    threshold_ns=100000, measured_group_size=group_size, selected_before_measured=True,
                    measured_calls=measured_calls, protocol_status="DRAFT"))
                for first in range(0, measured_calls, group_size):
                    _record_group(session, row_base, batch_id, "measured", first, group_size,
                                  "measured_" + str(first // group_size), writer, stream, counters)
                batch["clock_hz_end"] = session._clock()
                environment_end = dict(cpu=governor.snapshot(), temperature=temperature_snapshot(), timestamp_utc=_utc())
                batch["temperature_c_end"] = environment_end["temperature"]["temperature_c"]
                batch["finished_utc"] = environment_end["timestamp_utc"]
                batch["condition_deviations"] = (environment_start["cpu"]["condition_deviations"] +
                    environment_start["temperature"]["condition_deviations"] +
                    environment_end["cpu"]["condition_deviations"] + environment_end["temperature"]["condition_deviations"])
                if environment_end["cpu"]["cpu_governor"] != batch["cpu_governor"]:
                    batch["condition_deviations"].append("CPU_GOVERNOR_CHANGED_WITHIN_BATCH")
                _save(folder / "environment_end.json", environment_end)
                _save(output / "timing_manifest.json", manifest)
                session.close()
                session = None
        manifest["actual_clock_register_evidence_verified"] = all(
            row["clock_hz"] == row["clock_hz_end"] for row in manifest["batches"])
        manifest["phase_clock_evidence_verified"] = manifest["actual_clock_register_evidence_verified"]
        _save(output / "timing_manifest.json", manifest)
        summary = measurement.summarize_csv(csv_path, manifest)
        summary.update(correctness_fixtures_passed=15, integrated_phase_validation="PASS_BOUND_ARTIFACT",
                       board_bundle_sha256=_sha(root / "board_bundle.json"), protocol_status="DRAFT", formal_cost_eligible=False)
        _save(output / "summary.json", summary)
        success = summary["status"] == "PASS"
        return summary
    except BaseException as exc:
        _failure(output, session, exc)
        raise
    finally:
        if governor is not None:
            if QUARANTINE:
                GOVERNOR_QUARANTINE.append((governor, output))
            else:
                _save(output / "cpu_governor_restore.json", governor.restore())
        _stage(output, started, success)


def _retain_cli(output):
    print("FAILED: retaining original buffers; release requires unchanged original PL identity and idle. "
          "PL_IDENTITY_LOST permanently quarantines this session; no reset will be issued.", flush=True)
    while QUARANTINE:
        try:
            for session in list(QUARANTINE):
                try:
                    session.close()
                except Exception:
                    pass
            if QUARANTINE:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Core may still own buffers; process retention continues.", flush=True)
    for governor, directory in GOVERNOR_QUARANTINE:
        _save(directory / "cpu_governor_restore.json", governor.restore())
    GOVERNOR_QUARANTINE.clear()
    _save(Path(output) / "device_buffer_retention.json", dict(status="CORE_IDLE_BUFFERS_RELEASED_AFTER_FAILURE",
        reset_performed=False, buffers_freed=True, measurement_status="FAIL"))


def make_ownership_guard(args):
    """Bind this runner and its parent worker to an immutable cooperative lease."""
    context_path = args.ownership_request.resolve()
    expected_sha = args.ownership_request_sha256
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or _sha(context_path) != expected_sha:
        raise ValueError('runner ownership context changed')
    context = _load(context_path)
    root = args.case_dir.resolve()
    if context.get('schema') != 't6_runner_ownership_context_v1' or context.get('package') != str(root):
        raise ValueError('runner ownership package differs')
    if _sha(context['request_path']) != context['request_sha256']:
        raise ValueError('parent job request changed')
    module_path = root / 'board_ownership.py'
    bundle_name = 'board_probe_bundle.json' if args.mode == 'probe' else 'board_bundle.json'
    bound = _load(root / bundle_name)['files']['board_ownership.py']
    if _sha(module_path) != bound or bound != context['ownership_module_sha256']:
        raise ValueError('bound ownership helper differs')
    spec = importlib.util.spec_from_file_location('t6_runner_ownership', module_path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    argv = context['runner_argv_prefix'] + ['--ownership-request', str(context_path),
                                           '--ownership-request-sha256', expected_sha]
    own_runner = owner.process_identity(os.getpid(), 'runner', request_path=str(context_path),
        request_sha256=expected_sha, runner_path=str(Path(__file__).resolve()), expected_argv=argv)

    def guard(stage):
        if _sha(context_path) != expected_sha or _sha(context['request_path']) != context['request_sha256']:
            raise ValueError('ownership context or original job request changed')
        observation = owner.check(context['ownership_config'], own_board_lock=context['own_board_lock'],
                                  own_worker=context['own_worker'], own_runner=own_runner)
        if args.output.is_dir():
            with (args.output / 'ownership_checks.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(stage=stage, guard_version=OWNERSHIP_GUARD_VERSION,
                    ownership_context_sha256=expected_sha, observation=observation), allow_nan=False) + '\n')
        return observation
    return guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--mode", choices=("probe", "timing"), default="timing")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--ownership-request", type=Path, required=True)
    parser.add_argument("--ownership-request-sha256", required=True)
    args = parser.parse_args()
    try:
        action = run_probe if args.mode == "probe" else run_case
        guard = make_ownership_guard(args)
        guard("runner_entry")
        result = action(args.case_dir, args.output, args.board_id, timeout=args.timeout_seconds, ownership_guard=guard)
    except UnsafeDeviceState:
        _retain_cli(args.output)
        return 1
    print(result["status"], flush=True)
    return 0 if result["status"] in ("PASS", "PASS_BOARD_CORRECTNESS_PHASE_VALIDATION_PENDING") else 1


if __name__ == "__main__":
    raise SystemExit(main())
