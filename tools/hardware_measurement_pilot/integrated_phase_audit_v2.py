"""Version 2: include verified unused parameter-AXI tieoffs in nearest2x RTL.

This is a narrow parser for the Vitis 2022.2 RTL emitted by this pilot, not a
general Verilog prover. It explores every syntactic FSM successor, including
branches that may be infeasible. Unsupported shapes require manual review.
No result from this program is described as dynamic RTL/AXI simulation.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from tools.hardware_measurement_pilot import board_runner as board

RTL = "hls_project/solution1/syn/verilog"
SYNTHESIS = "hls_project/solution1/syn/report/pilot_kernel_csynth.xml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def strip_comments(text):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def states(text, kind="CS"):
    return sorted(set(map(int, re.findall(r"ap_" + kind + r"_fsm_state(\d+)", text))))


def always_blocks(text):
    """Generated always blocks start at column zero; use begin/end balance."""
    output = []
    for match in re.finditer(r"(?m)^always\s*@[^\n]*\bbegin\b", text):
        depth, finish = 0, None
        for token in re.finditer(r"\bbegin\b|\bend\b", text[match.start():]):
            depth += 1 if token.group() == "begin" else -1
            if depth == 0:
                finish = match.start() + token.end()
                break
        require(finish is not None, "unbalanced generated always block")
        output.append(dict(start=match.start(), line=text.count("\n", 0, match.start())+1,
                           text=text[match.start():finish]))
    return output


def block_assigning(blocks, signal, operator="="):
    pattern = r"\b" + re.escape(signal) + r"\s*" + re.escape(operator) + r"\s*"
    found = [b for b in blocks if re.search(pattern, b["text"])]
    require(len(found) == 1, f"expected one driver block for {signal}, got {len(found)}")
    return found[0]


def completion_condition(body, state):
    match = re.fullmatch(r"\s*if\s*\((.*?)\)\s*begin\s*ap_NS_fsm\s*=\s*ap_ST_fsm_state(\d+);\s*"
                         r"end\s*else\s*begin\s*ap_NS_fsm\s*=\s*ap_ST_fsm_state(\d+);\s*end\s*end\s*",
                         body, flags=re.S)
    require(match and int(match[3]) == state and int(match[2]) != state, "unsupported completion branch polarity")
    return match[1], int(match[2])


def fsm_analysis(text, blocks):
    phase_block = block_assigning(blocks, "phase")
    markers = {}
    for state, value in re.findall(
            r"(?:if|else\s+if)\s*\(\(1'b1\s*==\s*ap_CS_fsm_state(\d+)\)\)\s*begin\s*phase\s*=\s*8'd([0-3]);",
            phase_block["text"]):
        require(int(value) not in markers, "duplicate phase marker value")
        markers[int(value)] = int(state)
    require(set(markers) == {0, 1, 2, 3}, "four unique phase markers required")
    require(len(re.findall(r"\bphase\s*=", phase_block["text"])) == 5 and "phase = 'bx;" in phase_block["text"],
            "phase assignments include unsupported branches")
    valid_block = block_assigning(blocks, "phase_ap_vld")
    require(set(states(valid_block["text"])) == set(markers.values()), "valid gating differs from marker states")
    require(len(re.findall(r"phase_ap_vld\s*=\s*1'b1", valid_block["text"])) == 1 and
            "phase_ap_vld = 1'b0" in valid_block["text"], "unrecognized phase valid gating")
    valid_condition = re.search(r"if\s*\((.*?)\)\s*begin", valid_block["text"], re.S)[1]
    normalized = re.sub(r"\(1'b1\s*==\s*ap_CS_fsm_state\d+\)", "S", valid_condition)
    require(re.fullmatch(r"\(?S(?:\s*\|\s*S){3}\)?", normalized), "phase valid is not the OR of four marker states")
    match = re.search(r"case\s*\(ap_CS_fsm\)(.*?)endcase", text, flags=re.S)
    require(match, "generated top FSM missing")
    state_bodies = {int(m.group(1)): m.group(2) for m in re.finditer(
        r"ap_ST_fsm_state(\d+)\s*:\s*begin(.*?)(?=\n\s*ap_ST_fsm_state\d+\s*:|\n\s*default\s*:)",
        match.group(1), flags=re.S)}
    graph = {state: sorted(set(map(int, re.findall(r"ap_NS_fsm\s*=\s*ap_ST_fsm_state(\d+)", body))))
             for state, body in state_bodies.items()}
    require(graph and all(graph.values()), "unparsed FSM successors")
    require(all(t in graph for values in graph.values() for t in values), "missing FSM destination")
    reset = re.findall(r"ap_CS_fsm\s*<=\s*ap_ST_fsm_state(\d+);", text)
    require(len(reset) == 1, "unrecognized FSM reset")
    entry = int(reset[0])
    phase_by_state, seen, queue = {}, set(), deque([(entry, 0)])
    marker_by_state = {state: phase for phase, state in markers.items()}
    allowed = {(0, 1), (1, 2), (2, 3), (3, 0)}
    while queue:
        state, previous = queue.popleft()
        if (state, previous) in seen:
            continue
        seen.add((state, previous))
        current = marker_by_state.get(state, previous)
        if state in marker_by_state:
            require((previous, current) in allowed, f"invalid/repeated phase transition {previous}->{current} at {state}")
        phase_by_state.setdefault(state, set()).add(current)
        queue.extend((target, current) for target in graph[state])
    require(set(phase_by_state) == set(graph), "unreachable states need manual inspection")
    require(all(len(v) == 1 for v in phase_by_state.values()), "FSM state is reachable under conflicting phases")
    phases = {s: next(iter(v)) for s, v in phase_by_state.items()}
    return dict(markers=markers, graph=graph, state_bodies=state_bodies, phases=phases,
                marker_line=phase_block["line"], valid_line=valid_block["line"], reset_state=entry,
                exploration="all syntactic FSM successors x latched phase; no state-number ordering assumption")


def check_phase_states(numbers, expected, fsm, context):
    require(numbers, "no state controls found: " + context)
    require(all(fsm["phases"].get(s) == expected for s in numbers),
            f"{context} escapes phase {expected}: {numbers}")


def inspect_rtl(directory):
    directory = Path(directory)
    top = directory / "pilot_kernel.v"
    original = top.read_text(encoding="utf-8")
    text = strip_comments(original)
    blocks = always_blocks(text)
    fsm = fsm_analysis(text, blocks)
    groups = []
    # Only top pipeline groups. The separate memory/AXI adapters are handled below.
    for match in re.finditer(r"(?m)^(pilot_kernel_(?:pilot_kernel_)?Pipeline_\w+)\s+(grp_\w+)\s*\((.*?)\n\);", text, re.S):
        module, instance, body = match.groups()
        ports = dict(re.findall(r"\.(\w+)\(([^()]*)\)", body))
        names = set(ports)
        memory_bundles = {name.split("_")[2] for name in names if name.startswith("m_axi_")}
        if memory_bundles & {"inputs", "parameters"}:
            require(not memory_bundles & {"outputs"}, "group mixes transfer directions")
            kind, phase = "load", 1
        elif "outputs" in memory_bundles:
            kind, phase = "store", 3
        else:
            require(any(name.startswith("input_local") for name in names) and
                    any(name.startswith("output_local") for name in names), "unrecognized non-AXI compute group")
            kind, phase = "onchip_compute", 2
        start, done = ports.get("ap_start"), ports.get("ap_done")
        require(start and done, "pipeline handshake missing")
        driver = block_assigning(blocks, start + "_reg", "<=")
        require(driver["text"].startswith("always @ (posedge ap_clk)"), "pipeline start is not clocked by ap_clk")
        launch = states(driver["text"])
        check_phase_states(launch, phase, fsm, instance + " launch")
        check_phase_states([t for s in launch for t in fsm["graph"][s]], phase, fsm, instance + " registered start")
        require(re.search(re.escape(start) + r"_reg\s*<=\s*1'b1", driver["text"]), "no start assertion")
        done_states, done_evidence = [], []
        for state, body_text in fsm["state_bodies"].items():
            expanded = body_text
            for signal in set(re.findall(r"\bap_block_state\w+", body_text)):
                expanded += "\n" + block_assigning(blocks, signal)["text"]
            if done in expanded:
                done_states.append(state)
                done_evidence.append(dict(state=state, control=expanded.strip()))
                # A wait must hold itself while not completed and have one exit.
                require(state in fsm["graph"][state] and len(fsm["graph"][state]) == 2,
                        "unrecognized completion wait topology")
                require(re.search(re.escape(done) + r"\s*==\s*1'b[01]", expanded), "unrecognized done predicate")
                condition, destination = completion_condition(body_text, state)
                if done in condition:
                    require(re.search(re.escape(done) + r"\s*==\s*1'b1", condition), "completion exit does not require done=1")
                else:
                    names = set(re.findall(r"\bap_block_state\w+", condition))
                    require(len(names) == 1, "ambiguous combined completion gate")
                    signal = next(iter(names))
                    require(re.search(r"1'b0\s*==\s*" + re.escape(signal), condition), "completion block is not required low")
                    predicate = block_assigning(blocks, signal)["text"]
                    expression = re.search(re.escape(signal) + r"\s*=\s*(.*?);", predicate, re.S)[1]
                    transformed = re.sub(r"\(grp_\w+_ap_done\s*==\s*1'b0\)", "D", expression)
                    require(re.fullmatch(r"\(?D(?:\s*\|\s*D)*\)?", transformed), "combined completion block is not OR(not-done)")
        check_phase_states(done_states, phase, fsm, instance + " done wait")
        for state in done_states:
            require(fsm["phases"][state] == phase, "completion outside phase")
        child = directory / (module + ".v")
        require(child.is_file(), "missing instantiated pipeline RTL")
        groups.append(dict(module=module, instance=instance, kind=kind, phase=phase,
                           start_signal=start, done_signal=done, launch_states=launch,
                           completion_wait_states=done_states, start_driver_line=driver["line"],
                           completion_controls=done_evidence, module_sha256=board._sha(child)))
    require(any(g["kind"] == "onchip_compute" for g in groups), "no identified onchip computation group")
    require(any(g["kind"] == "load" for g in groups) and any(g["kind"] == "store" for g in groups),
            "load/store subgroups missing")
    controls = []
    optimized_parameter_bus = not re.search(r"\bparameters_(?:ARVALID|RREADY)\b", text)
    if optimized_parameter_bus:
        require(not re.search(r"(?m)^\w+_parameters_m_axi\s", text), "optimized parameter bus still has an adapter")
        require(not any("parameters" in p for m in re.finditer(
            r"(?m)^pilot_kernel_(?:pilot_kernel_)?Pipeline_\w+\s+grp_\w+\s*\((.*?)\n\);", text, re.S)
            for p in re.findall(r"\.(m_axi_\w+)\(", m[1])), "optimized parameter bus still feeds a pipeline")
        for suffix in ("ARVALID", "AWVALID", "WVALID", "RREADY", "BREADY"):
            signal = "m_axi_parameters_" + suffix
            drivers = re.findall(r"\bassign\s+" + signal + r"\s*=\s*([^;]+);", text)
            require(drivers == ["1'b0"] and not any(re.search(r"\b"+signal+r"\s*(?:=|<=)", b["text"]) for b in blocks),
                    "unused parameter control must have exactly one continuous zero driver: " + signal)
            controls.append(dict(signal=signal, phase=None, states=[],
                                 status="OPTIMIZED_AWAY_TIED_OFF", control=f"assign {signal} = 1'b0;"))
    for signal, phase in [("inputs_ARVALID", 1), ("inputs_RREADY", 1),
                          ("parameters_ARVALID", 1), ("parameters_RREADY", 1),
                          ("outputs_AWVALID", 3), ("outputs_WVALID", 3), ("outputs_BREADY", 3)]:
        if optimized_parameter_bus and signal.startswith("parameters_"):
            continue
        block = block_assigning(blocks, signal)
        active = states(block["text"])
        check_phase_states(active, phase, fsm, signal)
        require(re.search(r"end\s+else\s+begin\s*"+re.escape(signal)+r"\s*=\s*1'b0;\s*end\s*end\s*$", block["text"]),
                signal + " has no unconditional zero fallback")
        controls.append(dict(signal=signal, phase=phase, states=active, line=block["line"], control=block["text"]))
    idle = fsm["markers"][0]
    idle_predecessors = [s for s, targets in fsm["graph"].items() if idle in targets]
    require(len(idle_predecessors) == 1, "idle has bypass or ambiguous predecessor")
    response_wait = idle_predecessors[0]
    response_body = fsm["state_bodies"][response_wait]
    response_condition, response_destination = completion_condition(response_body, response_wait)
    require(re.search(r"outputs_BVALID\s*==\s*1'b1", response_body) and
            response_destination == idle and fsm["graph"][response_wait] == sorted([response_wait, idle]),
            "idle can precede output response completion")
    adapter = directory / "pilot_kernel_outputs_m_axi.v"
    adapter_text = strip_comments(adapter.read_text(encoding="utf-8"))
    # These are deliberately exact templates. A new generator revision must be reviewed.
    adapter_needles = [".I_BVALID(outputs_BVALID)", ".I_BREADY(outputs_BREADY)",
                       ".I_AWVALID(outputs_AWVALID)", ".I_WVALID(outputs_WVALID)"]
    require(all(n in text for n in adapter_needles), "top output adapter wiring changed")
    compact = re.sub(r"\s+", "", adapter_text)
    response_needles = [".s_valid(in_BUS_BVALID)", ".m_valid(resp_valid)",
                       "assignout_HLS_BVALID=resp_valid&&(last_resp===1'b1);",
                       "assignursp_write=wrsp_valid&&(!wrsp_type||in_AXI_BVALID);",
                       ".if_empty_n(out_HLS_BVALID)", ".if_read(in_HLS_BREADY)"]
    require(all(n in compact for n in response_needles), "output response aggregation template changed")
    return dict(status="PASS_STATIC_RTL_STRUCTURE_REQUIRES_INTEGRATION", phase_markers=fsm["markers"],
                phase_marker_line=fsm["marker_line"], phase_valid_line=fsm["valid_line"],
                state_phase=fsm["phases"], successors=fsm["graph"], exploration=fsm["exploration"],
                groups=groups, transfer_controls=controls, output_response_wait_state=response_wait,
                output_response_aggregation="HLS response waits for adapter last bus response; adapter RTL template checked",
                rtl_files={p.name: board._sha(p) for p in sorted(directory.glob("*.v"))},
                dynamic_integrated_rtl_simulation="NOT_RUN",
                limitation="Conservative generated-FSM inspection and known adapter-template checks; not a general Verilog equivalence proof.")


def inspect_hwh(path, expected_clock_hz):
    # A reset output connection alone is insufficient: the reset controller can
    # hold every AXI slave in reset when an auxiliary input is tied active.
    reset_wiring = board._reset_wiring_identity(path)
    root = ET.parse(path).getroot()
    modules = {m.get("INSTANCE"): m for m in root.iter("MODULE")}
    def port(module, name):
        require(module in modules, "missing HWH module " + module)
        values = [p.attrib for p in modules[module].iter("PORT") if p.get("NAME") == name]
        require(len(values) == 1, f"missing/ambiguous HWH port {module}.{name}")
        return values[0]
    clock_sinks = [("ps", "pl_clk0"), ("core", "ap_clk"), ("timer", "clk"),
                   ("rst", "slowest_sync_clk"), ("ctrl", "aclk"), ("mem", "aclk"),
                   ("ps", "maxihpm0_fpd_aclk"), ("ps", "saxihpc0_fpd_aclk")]
    clock_sinks += [(name, "s_axi_aclk") for name in
                    ("timer_counts", "timer_sequence", "timer_identity")]
    clocks = [dict(module=m, **port(m, p)) for m, p in clock_sinks]
    require(len({p.get("SIGNAME") for p in clocks}) == 1 and clocks[0].get("SIGNAME"), "core and timer clock nets differ")
    require(all(int(p["CLKFREQUENCY"]) == expected_clock_hz for p in clocks), "HWH clock frequency differs from accepted shell")
    connections = []
    pairs = [("core", "phase", "timer", "phase"), ("core", "phase_ap_vld", "timer", "phase_valid"),
             ("rst", "peripheral_aresetn", "core", "ap_rst_n"), ("rst", "peripheral_aresetn", "timer", "aresetn"),
             ("ps", "pl_resetn0", "rst", "ext_reset_in")]
    pairs += [("timer", a, b, c) for a, b, c in [("compute_cycles", "timer_counts", "gpio_io_i"),
        ("transfer_cycles", "timer_counts", "gpio2_io_i"), ("total_cycles", "timer_sequence", "gpio_io_i"),
        ("sequence_count", "timer_sequence", "gpio2_io_i"), ("identity", "timer_identity", "gpio_io_i"),
        ("status", "timer_identity", "gpio2_io_i")]]
    for am, ap, bm, bp in pairs:
        a, b = port(am, ap), port(bm, bp)
        require(a.get("DIR") == "O" and b.get("DIR") == "I" and a.get("SIGNAME") and a["SIGNAME"] == b.get("SIGNAME"),
                f"HWH connection differs: {am}.{ap}->{bm}.{bp}")
        connections.append(dict(source=am+"."+ap, destination=bm+"."+bp, net=a["SIGNAME"]))
    for m, p in [("core", "ap_rst_n"), ("timer", "aresetn"), ("rst", "ext_reset_in")]:
        require(port(m, p).get("POLARITY") == "ACTIVE_LOW", "reset polarity mismatch")
    reset_params = {p.get("NAME"): p.get("VALUE") for p in modules["rst"].iter("PARAMETER")}
    require(reset_params.get("C_EXT_RESET_HIGH") == "0", "reset controller does not match active-low PS reset")
    return dict(status="PASS_HWH_WIRING", single_clock_domain=True, clock_ports=clocks,
                connections=connections, reset_wiring=reset_wiring,
                reset_parameters={name: reset_params[name] for name in ("C_EXT_RESET_HIGH", "C_AUX_RESET_HIGH")},
                hwh_sha256=board._sha(path))


def timer_witness(case, validation_path=None):
    if validation_path:
        candidates = [Path(validation_path).resolve()]
    else:
        base = Path(__file__).resolve().parents[2] / "results/hardware_measurement_pilot_20261012/timer_validation"
        candidates = sorted(base.glob("*/validation.json"), reverse=True)
    for path in candidates:
        report = board._load(path)
        if report.get("status") != "PASS_TIMER_RTL" or report.get("source_sha256") != case["source_files"]["phase_timer.v"]:
            continue
        testbench = path.parent / "tb.sv"
        require(board._sha(testbench) == report.get("testbench_sha256"), "timer testbench hash differs")
        commands = report.get("commands", [])
        require(commands and all(row.get("returncode") == 0 for row in commands), "timer simulator command failed")
        logs = [board._child(path.parent, row["log"]) for row in commands]
        require("PHASE_TIMER_RTL_PASS" in logs[-1].read_text(encoding="utf-8"), "timer assertions did not pass")
        files = [path, testbench] + logs
        return dict(status="PASS_BOUND_TIMER_RTL", validation_path=str(path), simulator=report["simulator"],
                    source_sha256=report["source_sha256"],
                    scope="TIMER_RTL_ONLY; does not simulate the integrated HLS AXI kernel",
                    files={str(p): board._sha(p) for p in files})
    raise ValueError("no successful standalone timer simulation matches the exact timer source")


def inspect_case(case_dir, rtl_dir=None, hwh=None, expected_clock_hz=187500000, timer_validation=None):
    root = Path(case_dir).resolve()
    case = board._case_identity(root)
    directory = Path(rtl_dir).resolve() if rtl_dir else root / RTL
    handoff = Path(hwh).resolve() if hwh else root / "vivado_shell/top.hwh"
    rtl = inspect_rtl(directory)
    wiring = inspect_hwh(handoff, expected_clock_hz)
    timer = strip_comments((root / "phase_timer.v").read_text(encoding="utf-8"))
    needed = ["observed_phase = phase_valid ? phase : latched_phase", "else if (phase_valid) latched_phase <= phase",
              "compute_cycles <= work_compute", "transfer_cycles <= work_transfer", "total_cycles <= work_total",
              "sequence_count <= sequence_count + 1"]
    require(all(n in timer for n in needed), "timer latch/commit contract changed")
    witness = timer_witness(case, timer_validation)
    return dict(schema="pilot_phase_structure_review_v1", status="STRUCTURE_PASS_REVIEW_REQUIRED",
                created_utc=datetime.now(timezone.utc).isoformat(), case_id=case["case_id"],
                case_sha256=case["case_sha256"], auditor_sha256=board._sha(__file__),
                source_files={name: case["source_files"][name] for name in ("kernel.cpp", "config.h", "phase_timer.v")},
                rtl_directory=str(directory), hwh_path=str(handoff), expected_clock_hz=expected_clock_hz,
                rtl=rtl, wiring=wiring, timer_validation=witness, phase_contract=board.PHASE_CONTRACT,
                admitted_scope="Phase-marked on-chip compute interval and phase-marked load-plus-store interval; includes phase-local controller overhead.",
                excluded_claims=["arithmetic-only cycles", "DDR bus occupancy", "full integrated dynamic RTL simulation", "formal cost table eligibility"],
                next_gate="Review this exact structure report, then bind the routed bitstream and five complete board correctness fixtures.")


def finalize(case_dir, structure_path, review_path, probe_dir):
    root, probe = Path(case_dir).resolve(), Path(probe_dir).resolve()
    case, metadata = board._case_identity(root), board._route_identity(root)
    structure, review = board._load(structure_path), board._load(review_path)
    require(structure.get("status") == "STRUCTURE_PASS_REVIEW_REQUIRED", "structure inspection not passed")
    require(structure.get("case_sha256") == case["case_sha256"], "structure case binding differs")
    require(structure.get("auditor_sha256") == board._sha(__file__), "auditor changed after structure review")
    require(review.get("status") == "APPROVED_RTL_PHASE_STRUCTURE" and review.get("reviewer") and
            review.get("case_sha256") == case["case_sha256"] and
            review.get("structure_sha256") == board._sha(structure_path), "exact structure report has no approval record")
    # Recompute from the final package, not the old external path in the report.
    fresh = inspect_case(root, expected_clock_hz=metadata["clock_hz"],
                         timer_validation=structure["timer_validation"]["validation_path"])
    require(json.loads(json.dumps(fresh["rtl"])) == structure["rtl"] and fresh["wiring"] == structure["wiring"] and
            fresh["source_files"] == structure["source_files"] and fresh["timer_validation"] == structure["timer_validation"],
            "reviewed RTL/HWH/source/timer witness differs from final package")
    summary, records = board._load(probe / "probe_summary.json"), board._load(probe / "correctness_fixtures.json")
    require(summary.get("status") == "PASS_BOARD_CORRECTNESS_PHASE_VALIDATION_PENDING" and
            summary.get("case_sha256") == case["case_sha256"] and summary.get("correctness_fixtures_passed") == 5 and
            summary.get("latency_claim") == "NONE", "probe case/correctness contract differs")
    require(summary.get("source_files") == fresh["source_files"], "probe source binding differs")
    require(summary.get("bitstream_sha256") == board._sha(root / "vivado_shell/top.bit") and
            summary.get("hls_synthesis_report_sha256") == board._sha(root / SYNTHESIS) and
            summary.get("board_probe_bundle_sha256") == board._sha(root / "board_probe_bundle.json"), "probe artifact binding differs")
    bundle = board._load(root / "board_probe_bundle.json")
    board._checked_files(root, bundle["files"])
    require(bundle.get("mode") == "probe" and bundle.get("case_sha256") == case["case_sha256"], "probe bundle identity differs")
    require(summary.get("timer_magic") == 0x50334D32, "probe timer ABI differs")
    clock = float(summary["clock_hz"])
    require(math.isfinite(clock) and abs(clock / metadata["clock_hz"] - 1) < .001 and
            abs(float(summary["clock_hz_end"]) / clock - 1) < .00001, "probe clock is wrong/changed")
    require([row.get("fixture_id") for row in records] == [f"test{i}" for i in range(5)], "five ordered probe fixtures required")
    for row, fixture in zip(records, case["fixtures"]):
        values = [row.get(k) for k in ("compute_cycles", "transfer_cycles", "total_active_cycles", "sequence", "sequence_before", "timer_status")]
        require(all(type(v) is int for v in values), "probe counters are not exact integers")
        compute, transfer, total, sequence, previous, status = values
        require(0 < compute < 2**32-1 and 0 < transfer < 2**32-1 and compute+transfer == total < 2**32-1,
                "probe count partition/overflow failed")
        require(sequence == previous+1 and status == 1, "probe sequence/valid/error status failed")
        require(row.get("phase_counts_verified") is False and row.get("clock_hz") == clock, "probe observation provenance differs")
        require(row.get("compared_elements") == fixture["expected_elements"] and row.get("mismatched_elements") == 0,
                "complete output comparison failed")
        expected = fixture["expected_output_sha256"]
        require(row.get("expected_output_sha256") == expected and row.get("observed_output_sha256") == expected and
                board._sha(probe / (row["fixture_id"] + "_board.bin")) == expected, "raw board output SHA differs")
        require(math.isclose(float(row["unvalidated_compute_ns"]), compute*1e9/clock, rel_tol=1e-12) and
                math.isclose(float(row["unvalidated_transfer_ns"]), transfer*1e9/clock, rel_tol=1e-12), "probe cycles/time conversion differs")
    dest = root / "phase_evidence"
    require(not dest.exists(), "preserve existing phase evidence; make a new case package")
    dest.mkdir()
    for source, name in [(Path(structure_path), "structure.json"), (Path(review_path), "review.json")]:
        shutil.copyfile(source, dest / name)
    for name in ["probe_summary.json", "correctness_fixtures.json"] + [f"test{i}_board.bin" for i in range(5)]:
        shutil.copyfile(probe / name, dest / name)
    shutil.copyfile(__file__, dest / "integrated_phase_audit_v2.py")
    timer_dir = dest / "timer_only_simulation"
    timer_dir.mkdir()
    for source, digest in fresh["timer_validation"]["files"].items():
        require(board._sha(source) == digest, "timer evidence changed during finalization")
        shutil.copyfile(source, timer_dir / Path(source).name)
    evidence = {str(p.relative_to(root)).replace("\\", "/"): board._sha(p) for p in dest.rglob("*") if p.is_file()}
    evidence.update({RTL+"/"+name: digest for name, digest in fresh["rtl"]["rtl_files"].items()})
    evidence.update({"vivado_shell/top.hwh": board._sha(root / "vivado_shell/top.hwh"),
                     "board_probe_bundle.json": board._sha(root / "board_probe_bundle.json")})
    result = dict(schema="pilot_integrated_phase_validation_v1", status="PASS_INTEGRATED_PHASE_VALIDATION",
        created_utc=datetime.now(timezone.utc).isoformat(), case_sha256=case["case_sha256"],
        bitstream_sha256=board._sha(root / "vivado_shell/top.bit"), phase_contract=board.PHASE_CONTRACT,
        phase_interface="ap_vld_latched", single_clock_domain=True, phases_disjoint=True,
        source_files=fresh["source_files"], synthesis_report={"path": SYNTHESIS, "sha256": board._sha(root / SYNTHESIS)},
        evidence_files=evidence, method="Reviewed static generated-FSM/adapter inspection plus HWH wiring and five bound board observations",
        dynamic_integrated_rtl_simulation="NOT_RUN", probe_board_id=summary["board_id"], actual_probe_clock_hz=clock,
        admitted_scope=fresh["admitted_scope"], excluded_claims=fresh["excluded_claims"], formal_cost_eligible=False)
    save_new(root / "phase_validation.json", result)
    board._phase_identity(root, case, root / "phase_validation.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--case-dir", required=True)
    inspect.add_argument("--rtl-dir")
    inspect.add_argument("--hwh")
    inspect.add_argument("--expected-clock-hz", type=int, default=187500000)
    inspect.add_argument("--timer-validation")
    inspect.add_argument("--output", required=True)
    finish = sub.add_parser("finalize")
    for name in ("case-dir", "structure", "review", "probe-dir"):
        finish.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if args.action == "inspect":
        result = inspect_case(args.case_dir, args.rtl_dir, args.hwh, args.expected_clock_hz, args.timer_validation)
        save_new(args.output, result)
    else:
        result = finalize(args.case_dir, args.structure, args.review, args.probe_dir)
    print(json.dumps({"status": result["status"], "case_sha256": result["case_sha256"]}))


if __name__ == "__main__":
    main()
