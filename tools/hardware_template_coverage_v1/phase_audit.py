"""P3T1: conservative RTL checks for repeated load/compute/store tiles.

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

from tools.hardware_template_coverage_v1 import board_runner as board

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
    pattern = r"\b" + re.escape(signal) + r"\s*" + re.escape(operator) + r"(?!=)\s*"
    found = [b for b in blocks if re.search(pattern, b["text"])]
    require(len(found) == 1, f"expected one driver block for {signal}, got {len(found)}")
    return found[0]


def completion_condition(body, state):
    match = re.fullmatch(r"\s*if\s*\((.*?)\)\s*begin\s*ap_NS_fsm\s*=\s*ap_ST_fsm_state(\d+);\s*"
                         r"end\s*else\s*begin\s*ap_NS_fsm\s*=\s*ap_ST_fsm_state(\d+);\s*end\s*end\s*",
                         body, flags=re.S)
    require(match and int(match[3]) == state and int(match[2]) != state, "unsupported completion branch polarity")
    return match[1], int(match[2])


def _literal_expression(expression, operator):
    """Accept only parenthesized literals joined by one explicit operator."""
    compact = re.sub(r"\s+", "", expression)
    position = 0

    def operand():
        nonlocal position
        require(position < len(compact), "missing control predicate operand")
        if compact[position] == "(":
            position += 1
            joined()
            require(position < len(compact) and compact[position] == ")", "unbalanced control predicate")
            position += 1
        else:
            require(compact[position] in "SBT", "unsupported control predicate literal")
            position += 1

    def joined():
        nonlocal position
        operand()
        while position < len(compact) and compact[position] == operator:
            position += 1
            operand()

    joined()
    require(position == len(compact), "unsupported control predicate operator or suffix")


def _state_or_condition(condition):
    found = []

    def replacement(match):
        found.append(int(match[1]))
        return "S"

    normalized = re.sub(r"\(1'b1\s*==\s*ap_CS_fsm_state(\d+)\)", replacement, condition)
    _literal_expression(normalized, "|")
    require(found and len(found) == len(set(found)), "phase predicate has repeated or missing state literals")
    return found


def _predicate_terms(condition):
    """Exact DNF for known one-bit equality predicates with AND/OR only."""
    atoms = {}

    def atom(value):
        token = "A"+str(len(atoms))
        atoms[token] = value
        return token

    compact = re.sub(r"\(1'b1\s*==\s*ap_CS_fsm_state(\d+)\)",
                     lambda m: atom("state:"+m[1]), condition)
    compact = re.sub(r"\((\w+)\s*==\s*1'[bd]([01])\)",
                     lambda m: atom("eq:"+m[1]+":"+m[2]), compact)
    compact = re.sub(r"\(1'[bd]([01])\s*==\s*(\w+)\)",
                     lambda m: atom("eq:"+m[2]+":"+m[1]), compact)
    compact = re.sub(r"\s+", "", compact)
    position = 0

    def operand():
        nonlocal position
        require(position < len(compact), "missing predicate operand")
        if compact[position] == "(":
            position += 1
            result = joined_or()
            require(position < len(compact) and compact[position] == ")", "unbalanced predicate")
            position += 1
            return result
        match = re.match(r"A\d+", compact[position:])
        require(match and match[0] in atoms, "unsupported predicate atom")
        position += len(match[0])
        return {frozenset([atoms[match[0]]])}

    def joined_and():
        nonlocal position
        result = operand()
        while position < len(compact) and compact[position] == "&":
            position += 1
            following = operand()
            result = {left | right for left in result for right in following}
        return result

    def joined_or():
        nonlocal position
        result = joined_and()
        while position < len(compact) and compact[position] == "|":
            position += 1
            result |= joined_and()
        return result

    result = joined_or()
    require(position == len(compact), "unsupported predicate suffix or operator")
    return result


def _simple_branches(body):
    match = re.fullmatch(r"\s*if\s*\((.*?)\)\s*begin\s*ap_NS_fsm\s*=\s*ap_ST_fsm_state(\d+);\s*"
                         r"end\s*else\s*begin\s*ap_NS_fsm\s*=\s*ap_ST_fsm_state(\d+);\s*end\s*end\s*",
                         body, flags=re.S)
    require(match, "unsupported guarded-marker FSM branches")
    return match[1],int(match[2]),int(match[3])


def _guard_counter_proof(text, blocks, state_bodies, graph, signal):
    """Track ZERO vs UNKNOWN for the guard's exact initialized loop register.

    The only pruned path is equality-to-a-positive-bound while the exact
    register is provably zero. Every later write is conservatively UNKNOWN.
    No assumed trip count, loop ordering or unexplored arithmetic is used.
    """
    pattern = (r"(?m)^assign\s+"+re.escape(signal)+r"\s*=\s*\(\((\w+)\s*==\s*(\d+)'d(\d+)\)"
               r"\s*\?\s*1'b1\s*:\s*1'b0\);$")
    found = list(re.finditer(pattern,text))
    require(len(found)==1, "guard is not a known equality-to-positive-loop-bound template: "+signal)
    counter,width,bound = found[0][1],int(found[0][2]),int(found[0][3])
    require(0 < bound < 2**width, "guard loop bound must be positive and representable")
    driver = block_assigning(blocks,counter,"<=")
    require(driver["text"].startswith("always @ (posedge ap_clk)"), "guard counter is not clocked by ap_clk")
    branches = re.findall(r"(?:if|else\s+if)\s*\((.*?)\)\s*begin\s*"+re.escape(counter)+r"\s*<=\s*([^;]+);",
                          driver["text"],re.S)
    require(branches and len(branches)==len(re.findall(r"\b"+re.escape(counter)+r"\s*<=",driver["text"])),
            "guard counter has unsupported assignments")
    initialized,writers,evidence = set(),set(),[]
    for predicate,value in branches:
        terms = _predicate_terms(predicate)
        require(len(terms)==1, "counter write predicate must be one conjunction")
        term = next(iter(terms))
        numbers = [int(a.split(":")[1]) for a in term if a.startswith("state:")]
        require(len(numbers)==1, "counter write must bind one explicit FSM state")
        state = numbers[0]
        require(state in graph and state not in writers, "multiple counter assignments in one state need review")
        writers.add(state)
        zero = bool(re.fullmatch(r"\d+'[bd]0",value.strip()))
        targets = []
        if zero:
            if term==frozenset(["state:"+str(state)]):
                targets=graph[state]
            else:
                condition,true_target,false_target=_simple_branches(state_bodies[state])
                require(_predicate_terms(condition)==terms and true_target!=false_target,
                        "counter zero initialization does not match an exact FSM branch")
                targets=[true_target]
            initialized.update((state,target) for target in targets)
        evidence.append(dict(state=state,predicate=predicate,value=value,zero_initialization=zero,
                             proven_zero_targets=targets))
    require(initialized, "guard counter has no proven zero-entry edge")
    return dict(signal=signal,counter=counter,width=width,positive_bound=bound,
                zero_init_edges=[list(e) for e in sorted(initialized)],write_states=sorted(writers),
                counter_definition_line=text.count("\n",0,found[0].start())+1,
                counter_definition=found[0][0],driver_line=driver["line"],driver=driver["text"],
                assignments=evidence,
                abstraction="ZERO after exact initialization edge; UNKNOWN after every other syntactic write; only ZERO equality-to-positive-bound true branches are pruned")


def fsm_analysis(text, blocks):
    phase_block = block_assigning(blocks, "phase")
    markers = {value: [] for value in (0, 1, 2, 3)}
    assignments = re.findall(r"(?:if|else\s+if)\s*\((.*?)\)\s*begin\s*phase\s*=\s*8'd([0-3]);",
                             phase_block["text"], re.S)
    events = {}
    marker_terms = set()
    for condition, value in assignments:
        terms = _predicate_terms(condition)
        marker_terms |= terms
        for term in terms:
            states_in_term=[int(a.split(":")[1]) for a in term if a.startswith("state:")]
            guards=[a for a in term if not a.startswith("state:")]
            require(len(states_in_term)==1 and len(guards)<=1, "unsupported phase marker predicate")
            state=states_in_term[0]
            require(state not in events, "one state assigns conflicting phase values")
            if guards:
                require(int(value) in (0,3) and re.fullmatch(r"eq:icmp_\w+:1",guards[0]),
                        "only positive loop-equality guards on idle/store markers are supported")
            markers[int(value)].append(state)
            events[state]=dict(value=int(value),guard=guards[0] if guards else None,condition=condition)
    markers = {value: sorted(numbers) for value, numbers in markers.items()}
    # The reviewed kernel has one parameters-load marker and one tiled-load
    # marker. HLS may combine both states under a single OR predicate.
    require({value: len(numbers) for value, numbers in markers.items()} == {0: 1, 1: 2, 2: 1, 3: 1},
            "P3T1 requires five marker states: two load, one compute, one store and one idle")
    marker_states = [state for numbers in markers.values() for state in numbers]
    require(len(marker_states) == len(set(marker_states)), "one state assigns conflicting phase values")
    require(len(re.findall(r"\bphase\s*=", phase_block["text"])) == len(assignments)+1 and
            len(re.findall(r"\bphase\s*=\s*'bx;", phase_block["text"])) == 1,
            "phase assignments include unsupported branches")
    valid_block = block_assigning(blocks, "phase_ap_vld")
    require(set(states(valid_block["text"])) == set(marker_states), "valid gating differs from marker states")
    require(len(re.findall(r"phase_ap_vld\s*=\s*1'b1", valid_block["text"])) == 1 and
            len(re.findall(r"phase_ap_vld\s*=\s*1'b0", valid_block["text"])) == 1 and
            len(re.findall(r"phase_ap_vld\s*=", valid_block["text"])) == 2,
            "unrecognized phase valid gating")
    valid_condition = re.search(r"if\s*\((.*?)\)\s*begin", valid_block["text"], re.S)[1]
    require(_predicate_terms(valid_condition) == marker_terms,
            "phase valid is not the exact OR of all five guarded marker predicates")
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
    proofs=[]
    for event in events.values():
        if event["guard"]:
            signal=event["guard"].split(":")[1]
            if signal not in {p["signal"] for p in proofs}:
                proofs.append(_guard_counter_proof(text,blocks,state_bodies,graph,signal))
    proof_index={proof["signal"]:i for i,proof in enumerate(proofs)}
    conditional_destinations={}
    for state,event in events.items():
        if event["guard"]:
            condition,true_target,false_target=_simple_branches(state_bodies[state])
            require(_predicate_terms(condition)==_predicate_terms(event["condition"]),
                    "guarded marker does not match the exact FSM exit predicate")
            require(true_target!=false_target, "guarded marker must have distinct true/false successors")
            conditional_destinations[state]=(true_target,false_target)
    phase_by_state, seen, queue = {}, set(), deque([(entry,0,tuple(False for _ in proofs))])
    marker_by_state = {state: phase for phase, numbers in markers.items() for state in numbers}
    allowed = {(0, 1), (1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (3, 1), (3, 0)}
    transitions=set()
    pruned=set()
    while queue:
        state,previous,zero_flags = queue.popleft()
        if (state,previous,zero_flags) in seen:
            continue
        seen.add((state,previous,zero_flags))
        event=events.get(state)
        choices=[]
        for target in graph[state]:
            emitted=event is not None
            if event and event["guard"]:
                true_target,false_target=conditional_destinations[state]
                emitted=target==true_target
                signal=event["guard"].split(":")[1]
                if emitted and zero_flags[proof_index[signal]]:
                    pruned.add((state,target,previous,signal))
                    continue
            current=event["value"] if emitted else previous
            if emitted:
                require((previous,current) in allowed,f"invalid tiled phase transition {previous}->{current} at {state}")
            following=list(zero_flags)
            for i,proof in enumerate(proofs):
                if state in proof["write_states"]:
                    following[i]=(list([state,target]) in proof["zero_init_edges"])
            choices.append((target,current,tuple(following)))
            phase_by_state.setdefault(state,set()).add(current)
            transitions.add((state,previous,target,current,event["value"] if emitted else -1))
        require(choices,"all FSM successors were removed by counter proof")
        queue.extend(choices)
    require(set(phase_by_state) == set(graph), "unreachable states need manual inspection")
    # Loop controller states can legitimately be visited after the initial
    # parameters load and after a tile's store. AXI and compute controls below
    # still require a single exact phase, so ambiguity cannot admit activity.
    phases = {s: sorted(v) for s, v in phase_by_state.items()}
    return dict(markers=markers,marker_events=events,guard_counter_proofs=proofs,
                pruned_zero_bound_true_edges=[dict(state=s,target=t,previous_phase=p,guard=g) for s,t,p,g in sorted(pruned)],
                context_transitions=[dict(state=s,previous_phase=p,target=t,current_phase=c,emitted_phase=e if e>=0 else None)
                                     for s,p,t,c,e in sorted(transitions)],
                graph=graph, state_bodies=state_bodies, phases=phases,
                marker_line=phase_block["line"], valid_line=valid_block["line"], reset_state=entry,
                exploration="all syntactic FSM successors x latched phase x exact-zero/unknown guarded loop registers; only documented zero-equality-to-positive-bound edges pruned")


def check_phase_states(numbers, expected, fsm, context):
    require(numbers, "no state controls found: " + context)
    require(all(fsm["phases"].get(s) == [expected] for s in numbers),
            f"{context} escapes phase {expected}: {numbers}")


def inspect_store_pipeline(path):
    """Bind the Vitis sequential pipeline drain and pending-BVALID stall.

    This recognizes the actual 2022.2 generated pipeline and full wrapper
    template. It proves the live done condition cannot advance while its
    pending response is unacknowledged; it is not arbitrary RTL equivalence.
    """
    path=Path(path)
    text=strip_comments(path.read_text(encoding="utf-8"));blocks=always_blocks(text)
    done=block_assigning(blocks,"ap_done_int")
    match=re.fullmatch(r"always @ \(\*\) begin\s*if\s*\((.*?)\)\s*begin\s*ap_done_int\s*=\s*1'b1;\s*"
                       r"end\s*else\s*begin\s*ap_done_int\s*=\s*ap_done_reg;\s*end\s*end",done["text"],re.S)
    require(match,"unsupported store pipeline done latch template")
    terms=_predicate_terms(match[1]);require(len(terms)==1,"pipeline done must be one conjunction")
    term=next(iter(terms))
    exits=[a for a in term if re.fullmatch(r"eq:ap_loop_exit_ready_pp0_iter\d+_reg:1",a)]
    require(len(term)==2 and len(exits)==1 and "eq:ap_block_pp0_stage0_subdone:0" in term,
            "pipeline done is not drained-exit AND unblocked-subdone")
    drain=int(re.search(r"iter(\d+)",exits[0])[1])
    stall=block_assigning(blocks,"ap_block_pp0_stage0_subdone")
    expression=re.search(r"ap_block_pp0_stage0_subdone\s*=\s*(.*?);",stall["text"],re.S)[1]
    stall_terms=_predicate_terms(expression)
    response=[t for t in stall_terms if "eq:m_axi_outputs_BVALID:0" in t]
    require(len(response)==1 and len(response[0])==3,"pipeline pending BVALID stall is missing or ambiguous")
    pending=response[0]-{"eq:m_axi_outputs_BVALID:0"}
    enables=[a for a in pending if re.fullmatch(r"eq:ap_enable_reg_pp0_iter\d+:1",a)]
    flags=[a for a in pending if re.fullmatch(r"eq:icmp_\w+_pp0_iter\d+_reg:1",a)]
    predicates=[a for a in pending if re.fullmatch(r"eq:ap_predicate_op\d+_writeresp_state\d+:1",a)]
    compound=None
    if len(enables)==1 and not flags and len(predicates)==1:
        predicate=predicates[0].split(":")[1]
        control=block_assigning(blocks,predicate)
        definition=re.search(re.escape(predicate)+r"\s*=\s*(.*?);",control["text"],re.S)[1]
        compound_terms=_predicate_terms(definition)
        require(len(compound_terms)==1,"compound response predicate must be a pure conjunction")
        atoms=next(iter(compound_terms))
        require(len(atoms)==2 and all(re.fullmatch(r"eq:icmp_\w+_pp0_iter\d+_reg:1",a) for a in atoms),
                "compound response predicate requires two exact staged last-section flags")
        flags=sorted(atoms)
        compound=dict(signal=predicate,line=control["line"],control=control["text"],atoms=flags)
    require(len(enables)==1 and len(flags) in (1,2),"unrecognized pending response token")
    final_stage=int(re.search(r"iter(\d+)",enables[0])[1])
    flag_stages=[int(re.search(r"pp0_iter(\d+)_reg",flag)[1]) for flag in flags]
    require(final_stage==drain+1 and all(stage==drain for stage in flag_stages),
            "exit sentinel and last-response token stages are not adjacent")
    transfer_stall=block_assigning(blocks,"ap_block_pp0_stage0_11001")
    transfer_expression=re.search(r"ap_block_pp0_stage0_11001\s*=\s*(.*?);",transfer_stall["text"],re.S)[1]
    require(_predicate_terms(transfer_expression)==stall_terms,"pipeline drain and data transfers have different stalls")
    bready=block_assigning(blocks,"m_axi_outputs_BREADY")
    condition=re.search(r"if\s*\((.*?)\)\s*begin\s*m_axi_outputs_BREADY\s*=\s*1'b1;",bready["text"],re.S)
    require(condition and _predicate_terms(condition[1])=={frozenset(pending|{"eq:ap_block_pp0_stage0_11001:0"})},
            "BREADY does not bind the same pending response and stall domain")
    require(re.search(r"end\s*else\s*begin\s*m_axi_outputs_BREADY\s*=\s*1'b0;\s*end\s*end$",bready["text"]),
            "pipeline BREADY has no zero fallback")
    # Every enable token advances under the very stall that guards done.
    enable_evidence=[]
    require(re.search(r"\bassign\s+ap_enable_reg_pp0_iter0\s*=\s*ap_start_int;",text),"pipeline token zero not driven by fresh start")
    for index in range(1,final_stage+1):
        signal=f"ap_enable_reg_pp0_iter{index}"
        driver=block_assigning(blocks,signal,"<=")
        compact=re.sub(r"\s+","",driver["text"])
        expected=("always@(posedgeap_clk)beginif(ap_rst==1'b1)begin"+signal+"<=1'b0;endelsebegin"
                  "if((1'b0==ap_block_pp0_stage0_subdone))begin"+signal+"<=ap_enable_reg_pp0_iter"+str(index-1)+";endendend")
        if index==1:
            expected=("always@(posedgeap_clk)beginif(ap_rst==1'b1)begin"+signal+"<=1'b0;endelsebegin"
                      "if(((1'b1==ap_CS_fsm_pp0_stage0)&(1'b0==ap_block_pp0_stage0_subdone)))begin"+signal+"<=ap_start_int;endendend")
        killed=("always@(posedgeap_clk)beginif(ap_rst==1'b1)begin"+signal+"<=1'b0;endelsebegin"
                "if((1'b1==ap_condition_exit_pp0_iter"+str(index-1)+"_stage0))begin"+signal+"<=1'b0;endelseif((1'b0==ap_block_pp0_stage0_subdone))begin"+
                signal+"<=ap_enable_reg_pp0_iter"+str(index-1)+";endendend")
        if index==1:
            killed=("always@(posedgeap_clk)beginif(ap_rst==1'b1)begin"+signal+"<=1'b0;endelsebegin"
                    "if((1'b1==ap_condition_exit_pp0_iter0_stage0))begin"+signal+"<=1'b0;endelseif(((1'b1==ap_CS_fsm_pp0_stage0)&(1'b0==ap_block_pp0_stage0_subdone)))begin"+
                    signal+"<=ap_start_int;endendend")
        require(compact in (expected,killed),"pipeline enable drain chain changed: "+signal)
        if compact==killed:
            kill_signal=f"ap_condition_exit_pp0_iter{index-1}_stage0"
            kill=block_assigning(blocks,kill_signal)
            predicate=re.search(r"if\s*\((.*?)\)\s*begin\s*"+kill_signal+r"\s*=\s*1'b1;",kill["text"],re.S)
            require(predicate,"sentinel-kill predicate not found")
            kill_terms=_predicate_terms(predicate[1]);require(len(kill_terms)==1,"sentinel kill must be one conjunction")
            kt=next(iter(kill_terms))
            base_terms={f"eq:ap_enable_reg_pp0_iter{index-1}:1","eq:ap_block_pp0_stage0_subdone:0"}
            if "eq:ap_CS_fsm_pp0_stage0:1" in kt:base_terms.add("eq:ap_CS_fsm_pp0_stage0:1")
            require(base_terms<=kt and len([a for a in kt if re.fullmatch(r"eq:icmp_\w+:1",a)])==1 and len(kt)==len(base_terms)+1,
                    "pipeline sentinel kill does not bind token, end flag and drain stall")
        enable_evidence.append(dict(signal=signal,line=driver["line"],driver=driver["text"]))
    # Exit sentinel propagates in the same stall domain, one stage before the
    # final data/response token. This is the known in-order HLS drain layout.
    exit_evidence=[]
    for index in range(1,drain+1):
        signal=f"ap_loop_exit_ready_pp0_iter{index}_reg"
        driver=block_assigning(blocks,signal,"<=")
        rhs="ap_loop_exit_ready" if index==1 else f"ap_loop_exit_ready_pp0_iter{index-1}_reg"
        require(re.findall(r"\b"+signal+r"\s*<=\s*([^;]+);",driver["text"])==[rhs],"exit sentinel chain changed")
        require("ap_block_pp0_stage0_11001" in driver["text"] and
                re.search(r"1'b0\s*==\s*ap_block_pp0_stage0_11001",driver["text"]),"exit sentinel does not share transfer stall")
        exit_evidence.append(dict(signal=signal,line=driver["line"],driver=driver["text"]))
    latch=block_assigning(blocks,"ap_done_reg","<=")
    latch_compact=re.sub(r"\s+","",latch["text"])
    expected_latch=("always@(posedgeap_clk)beginif(ap_rst==1'b1)beginap_done_reg<=1'b0;endelsebegin"
                    "if((ap_continue_int==1'b1))beginap_done_reg<=1'b0;endelseif(((ap_loop_exit_ready_pp0_iter"+str(drain)+
                    "_reg==1'b1)&(1'b0==ap_block_pp0_stage0_subdone)))beginap_done_reg<=1'b1;endendend")
    require(latch_compact==expected_latch,"pipeline done register has an unrecognized bypass")
    flow=path.parent/"pilot_kernel_flow_control_loop_pipe_sequential_init.v"
    flow_text=strip_comments(flow.read_text(encoding="utf-8"))
    compact_flow=re.sub(r"\s+","",flow_text)
    known_flow="f027d73f993dcaed0c3211a4764b13d2f91bfa4c2a628f2a660b5661dc04562b"
    require(hashlib.sha256(compact_flow.encode()).hexdigest()==known_flow,"full sequential-init flow-control wrapper template changed")
    # Exact full wrapper hash binds continue=1 (ap_done_reg forced zero), fresh
    # start cache suppression/clearing, and live ap_done_int -> external done.
    ports=dict(re.findall(r"\.(\w+)\(([^()]*)\)",re.search(r"pilot_kernel_flow_control_loop_pipe_sequential_init\s+\w+\s*\((.*?)\n\);",text,re.S)[1]))
    needed={"ap_clk":"ap_clk","ap_rst":"ap_rst","ap_start":"ap_start","ap_done":"ap_done",
            "ap_start_int":"ap_start_int","ap_done_int":"ap_done_int","ap_continue_int":"ap_continue_int",
            "ap_loop_exit_done":"ap_done_int","ap_loop_exit_ready":"ap_condition_exit_pp0_iter0_stage0"}
    require(all(ports.get(k)==v for k,v in needed.items()),"store pipeline flow-control wrapper connections changed")
    flag_chain=[]
    for root_flag in flags:
        flag=root_flag.split(":")[1];seen_flags=set()
        while True:
            require(flag not in seen_flags,"pending-response flag chain cycle")
            seen_flags.add(flag)
            candidates=[b for b in blocks if re.search(r"\b"+re.escape(flag)+r"\s*<=",b["text"])]
            if not candidates:
                definitions=re.findall(r"(?m)^assign\s+"+re.escape(flag)+r"\s*=\s*([^;]+);",text)
                require(len(definitions)==1,"pending-response flag source definition missing")
                flag_chain.append(dict(root_flag=root_flag,signal=flag,kind="source",definition=definitions[0]))
                break
            require(len(candidates)==1,"multiple response-flag clock drivers")
            driver=candidates[0]
            values=re.findall(r"\b"+re.escape(flag)+r"\s*<=\s*([^;]+);",driver["text"])
            require(len(values)==1 and re.fullmatch(r"\w+",values[0]),"unsupported response-flag assignment")
            require("ap_block_pp0_stage0_11001" in driver["text"] and
                    re.search(r"1'b0\s*==\s*ap_block_pp0_stage0_11001",driver["text"]),"response flag does not share transfer stall")
            flag_chain.append(dict(root_flag=root_flag,signal=flag,kind="clocked",source=values[0],line=driver["line"],driver=driver["text"]))
            flag=values[0]
    return dict(status="PASS_KNOWN_PIPELINE_PENDING_RESPONSE_DRAIN_TEMPLATE",module=path.name,
                module_sha256=board._sha(path),done_condition=match[1],done_line=done["line"],
                done_latch_line=latch["line"],done_latch=latch["text"],
                pending_response_term=sorted(response[0]),response_stage=final_stage,exit_sentinel_stage=drain,
                stall_line=stall["line"],stall_control=stall["text"],
                bready_line=bready["line"],bready_control=bready["text"],
                enable_chain=enable_evidence,exit_sentinel_chain=exit_evidence,
                pending_response_flag_chain=flag_chain,
                compound_response_predicate=compound,
                flow_control_sha256=board._sha(flow),flow_control_compact_sha256=known_flow,flow_control_ports=ports,
                fresh_start="full known wrapper masks and clears cached done on ap_start_int; ap_continue_int is constant one so done-register fallback is zero",
                limitation="known Vitis in-order pipeline token/stall/drain templates and last-response adapter; no general RTL equivalence or dynamic integrated simulation")


def output_response_barriers(fsm, write_states=(), store_groups=()):
    """Require a last-response handshake on every path out of each store phase.

    Only a recognized self-wait -> exit conditional requiring BVALID=1 is
    accepted. Edges which stay in that wait do not complete the barrier. The
    search starts anew at each store marker and checks both the next-tile
    load and the final idle marker; a previous tile cannot satisfy this tile.
    """
    write_states = set(write_states)
    if write_states:
        check_phase_states(sorted(write_states), 3, fsm, "output write issue states")
    verified = []
    for state, body in fsm["state_bodies"].items():
        if not re.search(r"\boutputs_BVALID\b", body):
            continue
        check_phase_states([state], 3, fsm, "output response completion wait")
        condition, destination = completion_condition(body, state)
        require(fsm["graph"][state] == sorted([state,destination]), "ambiguous output response wait successors")
        normalized, count = re.subn(r"\(outputs_BVALID\s*==\s*1'b1\)", "B", condition)
        require(count == 1, "output response exit does not require BVALID=1")
        normalized = re.sub(r"\(1'b1\s*==\s*ap_CS_fsm_state"+str(state)+r"\)", "T", normalized)
        _literal_expression(normalized, "&")
        # The only other accepted literals are that very wait state's active
        # one-hot bit; OR/negation or another state cannot weaken the handshake.
        require("S" not in normalized, "unsupported response completion literal")
        verified.append(dict(wait_state=state,exit_state=destination,condition=condition))
    for group in store_groups:
        witness=group.get("response_pipeline")
        require(witness and witness.get("status")=="PASS_KNOWN_PIPELINE_PENDING_RESPONSE_DRAIN_TEMPLATE",
                "store pipeline response drain has not been verified")
        for state in group["completion_wait_states"]:
            condition,destination=completion_condition(fsm["state_bodies"][state],state)
            require(re.search(re.escape(group["done_signal"])+r"\s*==\s*1'b1",condition),"store exit does not require verified pipeline done")
            verified.append(dict(wait_state=state,exit_state=destination,condition=condition,
                                 barrier="known_child_pending_response_drain_then_top_done",module=group["module"]))
    require(verified, "no verified output last-response completion wait")
    completed_edges = {(r["wait_state"],r["exit_state"]) for r in verified}
    marker_by_state = {state: phase for phase, numbers in fsm["markers"].items() for state in numbers}
    witnesses = []
    all_used = set()
    context=fsm.get("context_transitions",[])
    by_context={}
    for row in context:
        by_context.setdefault((row["state"],row["previous_phase"]),[]).append(row)
    guarded=bool(fsm.get("guard_counter_proofs"))
    for start in fsm["markers"][3]:
        if guarded:
            beginning=[row for row in context if row["state"]==start and row["emitted_phase"]==3]
            require(beginning,"store marker has no proven emitted-store transition")
            seen,queue,exits,used=set(),deque((row["target"],3,False) for row in beginning),[],set()
            while queue:
                state,previous,complete=queue.popleft()
                if (state,previous,complete) in seen:continue
                seen.add((state,previous,complete))
                following_rows=by_context.get((state,previous),[])
                require(following_rows,"missing predicate-aware store successor context")
                for row in following_rows:
                    target,current=row["target"],row["current_phase"]
                    complete_now=complete and state not in write_states
                    crossed=(state,target) in completed_edges
                    if crossed:used.add((state,target))
                    following=complete_now or crossed
                    if current!=3:
                        require(current in (0,1),"store exits to an unsupported guarded phase")
                        require(following,f"store phase has response-completion bypass {state}->{target}")
                        exits.append(dict(state=state,target_state=target,target_phase=current))
                    else:queue.append((target,current,following))
            require(exits and {r["target_phase"] for r in exits}=={0,1},"guarded store loop lacks verified next-load and final-idle exits")
            all_used.update(used)
            witnesses.append(dict(store_marker_state=start,explored_state_response_pairs=len(seen),
                                  exits=sorted(exits,key=lambda r:(r["state"],r["target_state"])),
                                  response_completion_edges=[list(e) for e in sorted(used)]))
            continue
        seen,queue,exits,used = set(),deque([(start,False)]),[],set()
        while queue:
            state,complete = queue.popleft()
            if (state,complete) in seen:
                continue
            seen.add((state,complete))
            require(3 in fsm["phases"].get(state,[]), "store traversal escaped latched store phase")
            # A response from before another write cannot finish that later
            # write. Revoke the witness whenever the controller may issue AW/W.
            complete = complete and state not in write_states
            for target in fsm["graph"][state]:
                crossed=(state,target) in completed_edges
                if crossed:
                    used.add((state,target))
                following=complete or crossed
                next_phase=marker_by_state.get(target,3)
                if next_phase != 3:
                    require(next_phase in (0,1), "store exits to an unsupported phase")
                    require(following, f"store phase has response-completion bypass {state}->{target}")
                    exits.append(dict(state=state,target_state=target,target_phase=next_phase))
                else:
                    queue.append((target,following))
        require(exits, "store marker has no verified next-load/final-idle exit")
        require({r["target_phase"] for r in exits} == {0,1}, "store loop lacks next-tile or final-idle branch")
        all_used.update(used)
        witnesses.append(dict(store_marker_state=start,explored_state_response_pairs=len(seen),
                              exits=sorted(exits,key=lambda r:(r["state"],r["target_state"])),
                              response_completion_edges=[list(e) for e in sorted(used)]))
    require(all_used == completed_edges, "unbound output response wait needs manual review")
    return dict(status="PASS_ALL_STORE_EXIT_PATHS_REQUIRE_LAST_RESPONSE",
                waits=verified,store_markers=witnesses,write_issue_states=sorted(write_states),
                exploration="all syntactic store-phase successors x per-tile response completion flag; wait self-loop never completes")


def inspect_output_response_adapter(top_text, adapter_text):
    """Match the reviewed adapter's complete last-bus-response aggregation."""
    top_needles = [".I_BVALID(outputs_BVALID)", ".I_BREADY(outputs_BREADY)",
                   ".I_AWVALID(outputs_AWVALID)", ".I_WVALID(outputs_WVALID)"]
    require(all(n in top_text for n in top_needles), "top output adapter wiring changed")
    compact = re.sub(r"\s+", "", adapter_text)
    response_needles = [".s_valid(in_BUS_BVALID)", ".m_valid(resp_valid)",
                       "assignout_HLS_BVALID=resp_valid&&(last_resp===1'b1);",
                       "assignursp_write=wrsp_valid&&(!wrsp_type||in_AXI_BVALID);",
                       ".if_empty_n(out_HLS_BVALID)", ".if_read(in_HLS_BREADY)"]
    require(all(n in compact for n in response_needles), "output response aggregation template changed")
    return dict(status="PASS_KNOWN_LAST_BUS_RESPONSE_ADAPTER_TEMPLATE",
                top_wiring_needles=top_needles,response_aggregation_needles=response_needles,
                limitation="exact known-template structural check; not dynamic AXI simulation")


def _shortest_fsm_distance(graph, origin, targets):
    queue,seen=deque([(origin,0)]),set()
    while queue:
        state,distance=queue.popleft()
        if state in seen:continue
        seen.add(state)
        if state in targets:return distance
        queue.extend((target,distance+1) for target in graph[state])
    raise ValueError("no next store launch reachable from completed store")


def _shortest_guarded_fsm_distance(fsm, origin, targets):
    """Clock-distance lower bound with the same exact ZERO-entry facts."""
    proofs=fsm["guard_counter_proofs"]
    proof_index={p["signal"]:i for i,p in enumerate(proofs)}
    queue,seen=deque([(origin,tuple(False for _ in proofs),0)]),set()
    while queue:
        state,zero_flags,distance=queue.popleft()
        if (state,zero_flags) in seen:continue
        seen.add((state,zero_flags))
        if state in targets:return distance
        event=fsm["marker_events"].get(state)
        for target in fsm["graph"][state]:
            if event and event["guard"]:
                _,true_target,_=_simple_branches(fsm["state_bodies"][state])
                signal=event["guard"].split(":")[1]
                if target==true_target and zero_flags[proof_index[signal]]:continue
            flags=list(zero_flags)
            for i,proof in enumerate(proofs):
                if state in proof["write_states"]:flags[i]=[state,target] in proof["zero_init_edges"]
            queue.append((target,tuple(flags),distance+1))
    raise ValueError("no proven next store launch reachable after response completion")


def inspect_local_compute_tree(path):
    """Bind all generated descendants of a recognized local compute child."""
    path=Path(path);queue=deque([path]);seen=set();rows=[]
    while queue:
        current=queue.popleft()
        if current.name in seen:continue
        seen.add(current.name)
        text=strip_comments(current.read_text(encoding="utf-8"))
        require(not re.search(r"\bm_axi_",text),"on-chip compute hierarchy contains an AXI interface")
        children=[];same_file=[]
        definitions=set(re.findall(r"(?m)^\s*(?:\(\*[^*]*\*\)\s*)?module\s+(pilot_kernel_\w+)\b",text))
        for module in set(re.findall(r"(?m)^(pilot_kernel_\w+)\s+(?:#\s*\(.*?\)\s*)?\w+\s*\(",text,re.S)):
            child=current.parent/(module+".v")
            if child.is_file():children.append(child.name);queue.append(child)
            else:
                # DSP wrappers generated by this Vitis version place their
                # DSP48 implementation module after the wrapper in one file.
                # That complete file already passed the no-AXI scan and hash.
                require(module in definitions,"missing on-chip compute descendant RTL")
                same_file.append(module)
        rows.append(dict(module=current.name,sha256=board._sha(current),generated_descendants=sorted(children),
                         generated_descendants_defined_in_same_file=sorted(same_file)))
    return dict(status="PASS_BOUND_LOCAL_COMPUTE_HIERARCHY_NO_AXI",modules=sorted(rows,key=lambda r:r["module"]))


def inspect_completion_block(expression, signal, done, state, start_driver, blocks, fsm):
    """Recognize pure all-done OR or an exact gated tail-lane finish wait."""
    terms=_predicate_terms(expression)
    if all(len(term)==1 and re.fullmatch(r"eq:grp_\w+_ap_done:0",next(iter(term))) for term in terms):
        require(frozenset(["eq:"+done+":0"]) in terms,"all-done block omits this helper's completion")
        return dict(kind="ALL_REQUESTED_HELPERS_DONE",expression=expression)
    require(len(terms)==1,"unsupported conditional completion disjunction")
    term=next(iter(terms))
    done_atom="eq:"+done+":0"
    flags=[a for a in term if re.fullmatch(r"eq:select_ln100_\d+_reg_\d+:1",a)]
    require(len(term)==2 and done_atom in term and len(flags)==1,
            "conditional completion is not exact registered tail-valid AND not-done")
    flag=flags[0]
    expected=frozenset([flag,"state:"+str(state),"eq:"+signal+":0"])
    writes=[]
    for block in blocks:
        assignment=re.search(r"if\s*\((.*?)\)\s*begin\s*(output_tile\w*_we1)\s*=\s*1'b1;",block["text"],re.S)
        if not assignment:continue
        if _predicate_terms(assignment[1])!={expected}:continue
        require(len(re.findall(r"\b"+assignment[2]+r"\s*=",block["text"]))==2 and
                re.search(r"end\s*else\s*begin\s*"+assignment[2]+r"\s*=\s*1'b0;\s*end\s*end$",block["text"]),
                "tail lane write has unsupported fallback")
        writes.append(dict(signal=assignment[2],line=block["line"],control=block["text"]))
    require(len(writes)==1,"conditional finish wait is not tied to exactly one guarded output bank write")
    launch=re.search(r"if\s*\(([^\n]+)\)\s*begin\s*\w+_ap_start_reg\s*<=\s*1'b1;",start_driver["text"],re.S)
    require(launch,"conditional finish launch predicate absent")
    candidates=[]
    for launch_term in _predicate_terms(launch[1]):
        numbers=[int(a.split(":")[1]) for a in launch_term if a.startswith("state:")]
        if flag not in launch_term or len(numbers)!=1:continue
        predecessor=numbers[0]
        if state not in fsm["graph"][predecessor]:continue
        check_phase_states([predecessor],2,fsm,"conditional tail finish request")
        candidates.append(dict(state=predecessor,atoms=sorted(launch_term)))
    require(len(candidates)==1,"tail-valid finish wait has no exact preceding request with the same flag")
    return dict(kind="REQUESTED_TAIL_LANE_DONE_BEFORE_GUARDED_BANK_WRITE",expression=expression,
                valid_flag=flag,wait_state=state,request=candidates[0],bank_write=writes[0],
                proof="bank write requires flag=1 AND block=0; block=flag AND !done, so every written valid lane requires done=1; flag=0 neither requests nor writes that bank")


def inspect_rtl(directory):
    directory = Path(directory)
    top = directory / "pilot_kernel.v"
    original = top.read_text(encoding="utf-8")
    text = strip_comments(original)
    blocks = always_blocks(text)
    fsm = fsm_analysis(text, blocks)
    groups = []
    # Only top pipeline groups. The separate memory/AXI adapters are handled below.
    for match in re.finditer(r"(?m)^(pilot_kernel_(?:pilot_kernel_)?(?:Pipeline_\w+|finish_r))\s+(grp_\w+)\s*\((.*?)\n\);", text, re.S):
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
            local_input=any(re.fullmatch(r"input_tile\w*_q[01]",name) for name in names)
            local_output=any(re.fullmatch(r"output_tile\w*_(?:d|we)[01]",name) for name in names)
            local_mac=(local_input and any(re.fullmatch(r"weights_local\w*_q[01]",name) for name in names) and
                       any(re.fullmatch(r"(?:add\w*|acc\w*)_out",name) for name in names))
            finish=(module=="pilot_kernel_finish_r" and {"acc","ap_return","p_read"}<=names)
            require((local_input and local_output) or local_mac or finish,"unrecognized non-AXI compute group")
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
        conditional_waits=[]
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
                    conditional_waits.append(inspect_completion_block(expression,signal,done,state,driver,blocks,fsm))
        check_phase_states(done_states, phase, fsm, instance + " done wait")
        for state in done_states:
            require(fsm["phases"][state] == [phase], "completion outside phase")
        child = directory / (module + ".v")
        require(child.is_file(), "missing instantiated pipeline RTL")
        entry=dict(module=module, instance=instance, kind=kind, phase=phase,
                           start_signal=start, done_signal=done, launch_states=launch,
                           completion_wait_states=done_states, start_driver_line=driver["line"],
                           completion_controls=done_evidence, conditional_completion_controls=conditional_waits,
                           module_sha256=board._sha(child))
        if kind=="store":
            entry["response_pipeline"]=inspect_store_pipeline(child)
            gaps=[]
            for wait in done_states:
                _,destination=completion_condition(fsm["state_bodies"][wait],wait)
                gap=_shortest_guarded_fsm_distance(fsm,destination,set(launch))
                require(gap>=entry["response_pipeline"]["response_stage"],
                        "insufficient inactive top-FSM clocks to flush previous store sentinel before reuse")
                gaps.append(dict(completion_wait_state=wait,exit_state=destination,
                                 next_launch_minimum_fsm_edges=gap,
                                 required_enable_exit_flush_clocks=entry["response_pipeline"]["response_stage"],
                                 proof="all FSM paths x exact-zero/unknown counters; initialized positive-bound loops cannot be skipped"))
            entry["response_pipeline"]["fresh_start_minimum_gap"]=gaps
        if kind=="onchip_compute":
            child_text=strip_comments(child.read_text(encoding="utf-8"))
            require(not re.search(r"\bm_axi_",child_text),"on-chip compute child has an AXI interface")
            entry["recognized_local_dataflow"]=("scalar_MAC_from_input_tile_and_local_weights" if local_mac else
                                                "integer_finish_scalar" if finish else "local_input_tile_to_output_tile")
            entry["local_compute_hierarchy"]=inspect_local_compute_tree(child)
        groups.append(entry)
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
    response_barriers = output_response_barriers(fsm, sorted({s for control in controls
        if control["signal"] in ("outputs_AWVALID", "outputs_WVALID") for s in control["states"]}),
        [g for g in groups if g["kind"]=="store"])
    adapter = directory / "pilot_kernel_outputs_m_axi.v"
    adapter_text = strip_comments(adapter.read_text(encoding="utf-8"))
    # These are deliberately exact templates. A new generator revision must be reviewed.
    adapter_review = inspect_output_response_adapter(text, adapter_text)
    return dict(status="PASS_STATIC_RTL_STRUCTURE_REQUIRES_INTEGRATION", phase_markers=fsm["markers"],
                phase_marker_line=fsm["marker_line"], phase_valid_line=fsm["valid_line"],
                state_phase=fsm["phases"], successors=fsm["graph"], exploration=fsm["exploration"],
                guarded_marker_events=fsm["marker_events"],guard_counter_proofs=fsm["guard_counter_proofs"],
                pruned_zero_bound_true_edges=fsm["pruned_zero_bound_true_edges"],
                context_transitions=fsm["context_transitions"],
                groups=groups, transfer_controls=controls,
                output_response_wait_states=[r["wait_state"] for r in response_barriers["waits"]],
                output_response_barriers=response_barriers,
                output_response_adapter=adapter_review,
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
        base = Path(__file__).resolve().parents[2] / "results/hardware_template_coverage_20261013/timer_validation"
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
    return dict(schema="t6_phase_structure_review_v1", status="STRUCTURE_PASS_REVIEW_REQUIRED",
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
    require(summary.get("timer_magic") == 0x50335431, "probe timer ABI differs")
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
    shutil.copyfile(__file__, dest / "phase_audit.py")
    timer_dir = dest / "timer_only_simulation"
    timer_dir.mkdir()
    for source, digest in fresh["timer_validation"]["files"].items():
        require(board._sha(source) == digest, "timer evidence changed during finalization")
        shutil.copyfile(source, timer_dir / Path(source).name)
    evidence = {str(p.relative_to(root)).replace("\\", "/"): board._sha(p) for p in dest.rglob("*") if p.is_file()}
    evidence.update({RTL+"/"+name: digest for name, digest in fresh["rtl"]["rtl_files"].items()})
    evidence.update({"vivado_shell/top.hwh": board._sha(root / "vivado_shell/top.hwh"),
                     "board_probe_bundle.json": board._sha(root / "board_probe_bundle.json")})
    result = dict(schema="t6_integrated_phase_validation_v1", status="PASS_INTEGRATED_PHASE_VALIDATION",
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
