"""Positive/negative tests for the tiled generated-FSM inspection contract.

Synthetic FSM tests exercise parser and conservative path checks. They do
not certify any generated HLS RTL or its integrated AXI/DDR behavior.
"""
import unittest
from pathlib import Path
import shutil
import tempfile

from .phase_audit import (always_blocks, block_assigning, check_phase_states, fsm_analysis,
                          _shortest_guarded_fsm_distance,
                          inspect_completion_block, inspect_local_compute_tree, inspect_output_response_adapter,
                          inspect_store_pipeline, output_response_barriers)


def generated_fsm(*, merged_load=False, bypass=False, response_one=True,
                  weak_response=False, extra_valid=False, wrong_phase=False,
                  hold_store_marker=False):
    predicates=[(13,0),(9,3),(7,2),(5,1),(2,1)]
    clauses=[]
    if merged_load:
        clauses=["if (((1'b1 == ap_CS_fsm_state2) | (1'b1 == ap_CS_fsm_state5))) begin\n"
                 "    phase = 8'd1;\nend"]
        predicates=[(13,0),(9,3),(7,2)]
    for state,value in predicates:
        keyword="else if" if clauses else "if"
        clauses.append(f"{keyword} ((1'b1 == ap_CS_fsm_state{state})) begin\n    phase = 8'd{value};\nend")
    phase="always @ (*) begin\n"+" ".join(clauses)+" else begin\n    phase = 'bx;\nend\nend\n"
    valid_numbers=[13,9,7,5,2]+([4] if extra_valid else [])
    valid_condition=" | ".join(f"(1'b1 == ap_CS_fsm_state{s})" for s in valid_numbers)
    valid=f"always @ (*) begin\nif ({valid_condition}) begin\n    phase_ap_vld = 1'b1;\nend else begin\n    phase_ap_vld = 1'b0;\nend\nend\n"
    bodies={1:"if (ap_start == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state2;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state1;\nend",
            2:"ap_NS_fsm = ap_ST_fsm_state3;",3:"ap_NS_fsm = ap_ST_fsm_state4;",
            4:"ap_NS_fsm = ap_ST_fsm_state5;",5:"ap_NS_fsm = ap_ST_fsm_state6;",
            6:"if (input_done == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state7;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state6;\nend",
            7:"ap_NS_fsm = ap_ST_fsm_state8;",
            8:"if (compute_done == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state9;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state8;\nend",
            9:"ap_NS_fsm = ap_ST_fsm_state10;",
            10:"if (store_done == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state11;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state10;\nend",
            11:"",12:"if (another_tile == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state4;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state13;\nend",
            13:"ap_NS_fsm = ap_ST_fsm_state1;"}
    equality="1'b1" if response_one else "1'b0"
    operation="|" if weak_response else "&"
    bodies[11]=f"if (((outputs_BVALID == {equality}) {operation} (1'b1 == ap_CS_fsm_state11))) begin\n    ap_NS_fsm = ap_ST_fsm_state12;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state11;\nend"
    if bypass:
        bodies[10]="if (bypass == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state12;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state11;\nend"
    if wrong_phase:
        bodies[12]=bodies[12].replace("ap_ST_fsm_state4","ap_ST_fsm_state7")
    if hold_store_marker:
        bodies[9]="if (hold == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state9;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state10;\nend"
    controller="always @ (*) begin\ncase (ap_CS_fsm)\n"
    for state,body in sorted(bodies.items()):
        controller+=f"ap_ST_fsm_state{state}: begin\n{body}\nend\n"
    controller+="default: begin\nap_NS_fsm = ap_ST_fsm_state1;\nend\nendcase\nend\n"
    reset="always @ (posedge ap_clk) begin\nif (ap_rst_n == 1'b0) begin\n    ap_CS_fsm <= ap_ST_fsm_state1;\nend else begin\n    ap_CS_fsm <= ap_NS_fsm;\nend\nend\n"
    return phase+valid+controller+reset


def parse(**kwargs):
    text=generated_fsm(**kwargs)
    return fsm_analysis(text,always_blocks(text))


def guarded_fsm(*, initialize=True, zero_bound=False, weaken_valid=False):
    text=generated_fsm()
    text=text.replace("ap_ST_fsm_state3: begin\nap_NS_fsm = ap_ST_fsm_state4;",
                      "ap_ST_fsm_state3: begin\nap_NS_fsm = ap_ST_fsm_state12;")
    guard="icmp_guard_fu_100_p2"
    conditional=f"(({guard} == 1'd1) & (1'b1 == ap_CS_fsm_state12))"
    text=text.replace("((1'b1 == ap_CS_fsm_state13)) begin\n    phase = 8'd0;",
                      "("+conditional+") begin\n    phase = 8'd0;")
    text=text.replace("(1'b1 == ap_CS_fsm_state13) |",conditional+" |")
    text=text.replace("if (another_tile == 1'b1) begin\n    ap_NS_fsm = ap_ST_fsm_state4;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state13;",
                      "if ("+conditional+") begin\n    ap_NS_fsm = ap_ST_fsm_state13;\nend else begin\n    ap_NS_fsm = ap_ST_fsm_state4;")
    if weaken_valid:
        old="if ("+conditional+" |"
        text=text.replace(old,"if ("+conditional.replace("1'd1","1'd0")+" |")
    bound=0 if zero_bound else 1
    text+=f"assign {guard} = ((loop_counter == 4'd{bound}) ? 1'b1 : 1'b0);\n"
    init="if ((1'b1 == ap_CS_fsm_state3)) begin\n    loop_counter <= 4'd0;\nend else " if initialize else ""
    text+="always @ (posedge ap_clk) begin\n"+init+"if ((1'b1 == ap_CS_fsm_state10)) begin\n    loop_counter <= update_unknown;\nend\nend\n"
    return text


class PhaseAuditTests(unittest.TestCase):
    def test_repeated_tiles_and_initial_load(self):
        fsm=parse()
        self.assertEqual(fsm["markers"],{0:[13],1:[2,5],2:[7],3:[9]})
        self.assertEqual(fsm["phases"][4],[1,3])
        check_phase_states([6],1,fsm,"load")
        check_phase_states([8],2,fsm,"compute")
        check_phase_states([10,11],3,fsm,"store")
        witness=output_response_barriers(fsm,[10])
        self.assertEqual(witness["waits"][0]["wait_state"],11)
        self.assertEqual({row["target_phase"] for row in witness["store_markers"][0]["exits"]},{0,1})

    def test_merged_or_load_marker_is_equivalent(self):
        merged=parse(merged_load=True)
        self.assertEqual(merged["markers"],parse()["markers"])
        output_response_barriers(merged,[10])

    def test_same_store_phase_marker_hold_is_allowed(self):
        output_response_barriers(parse(hold_store_marker=True),[10])

    def test_ambiguous_loop_state_cannot_control_axi(self):
        with self.assertRaisesRegex(ValueError,"escapes phase"):
            check_phase_states([4],1,parse(),"AXI launch")

    def test_extra_valid_state_fails(self):
        with self.assertRaisesRegex(ValueError,"valid gating differs"):
            parse(extra_valid=True)

    def test_and_instead_of_or_valid_fails(self):
        text=generated_fsm().replace("(1'b1 == ap_CS_fsm_state13) |", "(1'b1 == ap_CS_fsm_state13) &",1)
        with self.assertRaisesRegex(ValueError,"phase valid is not the exact OR"):
            fsm_analysis(text,always_blocks(text))

    def test_store_to_compute_without_load_fails(self):
        with self.assertRaisesRegex(ValueError,"invalid tiled phase transition 3->2"):
            parse(wrong_phase=True)

    def test_response_bypass_next_tile_and_idle_fails(self):
        with self.assertRaisesRegex(ValueError,"response-completion bypass"):
            output_response_barriers(parse(bypass=True),[10])

    def test_zero_bvalid_exit_fails(self):
        with self.assertRaisesRegex(ValueError,"does not require BVALID=1"):
            output_response_barriers(parse(response_one=False),[10])

    def test_or_bvalid_exit_fails(self):
        with self.assertRaisesRegex(ValueError,"control predicate"):
            output_response_barriers(parse(weak_response=True),[10])

    def test_prior_response_cannot_cover_a_later_write(self):
        fsm=parse()
        fsm["graph"][12]=[10,13]
        fsm["graph"][10]=[11,13]
        with self.assertRaisesRegex(ValueError,"response-completion bypass"):
            output_response_barriers(fsm,[10])

    def test_missing_completion_wait_fails(self):
        fsm=parse()
        fsm["state_bodies"][11]="ap_NS_fsm = ap_ST_fsm_state12;\nend\n"
        with self.assertRaisesRegex(ValueError,"no verified output"):
            output_response_barriers(fsm,[10])

    def test_wrong_wait_branch_polarity_fails(self):
        fsm=parse()
        fsm["state_bodies"][11]=fsm["state_bodies"][11].replace("ap_ST_fsm_state12","ap_ST_fsm_state11",1)
        with self.assertRaisesRegex(ValueError,"unsupported completion branch polarity"):
            output_response_barriers(fsm,[10])

    def test_unknown_response_predicate_fails(self):
        fsm=parse()
        fsm["state_bodies"][11]=fsm["state_bodies"][11].replace("(1'b1 == ap_CS_fsm_state11)","unreviewed_predicate")
        with self.assertRaisesRegex(ValueError,"unsupported control predicate"):
            output_response_barriers(fsm,[10])


class GuardedMarkerTests(unittest.TestCase):
    def test_guarded_idle_preserves_zero_entry_proof(self):
        text=guarded_fsm()
        fsm=fsm_analysis(text,always_blocks(text))
        self.assertEqual(fsm["guard_counter_proofs"][0]["zero_init_edges"],[[3,12]])
        self.assertEqual(fsm["guard_counter_proofs"][0]["positive_bound"],1)
        self.assertTrue(any(row["state"]==12 and row["previous_phase"]==1
                            for row in fsm["pruned_zero_bound_true_edges"]))
        self.assertEqual(fsm["phases"][12],[0,1,3])

    def test_missing_zero_initialization_fails(self):
        text=guarded_fsm(initialize=False)
        with self.assertRaisesRegex(ValueError,"no proven zero-entry edge"):
            fsm_analysis(text,always_blocks(text))

    def test_zero_loop_bound_cannot_remove_initial_idle_exit(self):
        text=guarded_fsm(zero_bound=True)
        with self.assertRaisesRegex(ValueError,"loop bound must be positive"):
            fsm_analysis(text,always_blocks(text))

    def test_valid_must_repeat_the_exact_guard(self):
        text=guarded_fsm(weaken_valid=True)
        with self.assertRaisesRegex(ValueError,"exact OR of all five guarded"):
            fsm_analysis(text,always_blocks(text))

    def test_fresh_start_gap_cannot_skip_initialized_nonempty_loop(self):
        text=guarded_fsm();fsm=fsm_analysis(text,always_blocks(text))
        # Raw state3->state12->state13 is the infeasible equality-to-bound
        # shortcut after counter=0. The real lower bound must visit a write.
        self.assertGreater(_shortest_guarded_fsm_distance(fsm,3,{13}),2)


class AdapterTemplateTests(unittest.TestCase):
    top="\n".join([".I_BVALID(outputs_BVALID)",".I_BREADY(outputs_BREADY)",
                    ".I_AWVALID(outputs_AWVALID)",".I_WVALID(outputs_WVALID)"])
    adapter="\n".join([".s_valid(in_BUS_BVALID)",".m_valid(resp_valid)",
                        "assign out_HLS_BVALID = resp_valid && (last_resp === 1'b1);",
                        "assign ursp_write = wrsp_valid && (!wrsp_type || in_AXI_BVALID);",
                        ".if_empty_n(out_HLS_BVALID)",".if_read(in_HLS_BREADY)"])

    def test_known_response_aggregation_template(self):
        self.assertEqual(inspect_output_response_adapter(self.top,self.adapter)["status"],
                         "PASS_KNOWN_LAST_BUS_RESPONSE_ADAPTER_TEMPLATE")

    def test_missing_last_response_guard_fails(self):
        with self.assertRaisesRegex(ValueError,"aggregation template changed"):
            inspect_output_response_adapter(self.top,self.adapter.replace(" && (last_resp === 1'b1)",""))

    def test_wrong_top_response_signal_fails(self):
        with self.assertRaisesRegex(ValueError,"adapter wiring changed"):
            inspect_output_response_adapter(self.top.replace(".I_BVALID(outputs_BVALID)",".I_BVALID(1'b1)"),self.adapter)

    def test_missing_hls_response_read_gate_fails(self):
        with self.assertRaisesRegex(ValueError,"aggregation template changed"):
            inspect_output_response_adapter(self.top,self.adapter.replace(".if_read(in_HLS_BREADY)",".if_read(1'b1)"))


class ActualPipelineMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        base=root/"results/hardware_template_coverage_20261013/attempt01/phase_prechecks/20261013T033736231856Z/cases/t6_body8_to_head16_3e96d4a01dc5_tiled/hls_project/solution1/syn/verilog"
        if not base.is_dir():raise unittest.SkipTest("actual HLS boundary snapshot not available")
        cls.base=base
        cls.pipeline=base/"pilot_kernel_pilot_kernel_Pipeline_VITIS_LOOP_208_13.v"
        cls.work=root/"results/hardware_template_coverage_20261013"

    def mutate(self,old,new,*,flow=False,error=""):
        with tempfile.TemporaryDirectory(prefix="phase_pipeline_negative_",dir=self.work) as temporary:
            directory=Path(temporary)
            for source in [self.pipeline,self.base/"pilot_kernel_flow_control_loop_pipe_sequential_init.v"]:
                shutil.copyfile(source,directory/source.name)
            target=directory/("pilot_kernel_flow_control_loop_pipe_sequential_init.v" if flow else self.pipeline.name)
            text=target.read_text(encoding="utf-8")
            self.assertIn(old,text)
            target.write_text(text.replace(old,new),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,error or "."):
                inspect_store_pipeline(directory/self.pipeline.name)

    def test_actual_pipeline_known_drain(self):
        result=inspect_store_pipeline(self.pipeline)
        self.assertEqual((result["response_stage"],result["exit_sentinel_stage"]),(10,9))
        self.assertGreater(len(result["pending_response_flag_chain"]),1)

    def test_pending_bvalid_must_block_done(self):
        self.mutate("(m_axi_outputs_BVALID == 1'b0)","(m_axi_outputs_BVALID == 1'b1)",error="pending BVALID stall")

    def test_done_requires_subdone_unblocked(self):
        self.mutate("(ap_loop_exit_ready_pp0_iter9_reg == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone)",
                    "(ap_loop_exit_ready_pp0_iter9_reg == 1'b1)",error="unblocked-subdone")

    def test_enable_chain_cannot_advance_while_pending_response(self):
        self.mutate("ap_enable_reg_pp0_iter10 <= ap_enable_reg_pp0_iter9;","ap_enable_reg_pp0_iter10 <= 1'b0;",error="enable drain chain changed")

    def test_exit_sentinel_chain_cannot_skip_stage(self):
        self.mutate("ap_loop_exit_ready_pp0_iter9_reg <= ap_loop_exit_ready_pp0_iter8_reg;",
                    "ap_loop_exit_ready_pp0_iter9_reg <= ap_loop_exit_ready_pp0_iter7_reg;",error="sentinel chain changed")

    def test_response_flag_cannot_bypass_transfer_stall(self):
        self.mutate("icmp_ln208_3_reg_748_pp0_iter9_reg <= icmp_ln208_3_reg_748_pp0_iter8_reg;",
                    "icmp_ln208_3_reg_748_pp0_iter9_reg <= unreviewed_expression | icmp_ln208_3_reg_748_pp0_iter8_reg;",error="response-flag assignment")

    def test_done_latch_fallback_requires_constant_continue(self):
        self.mutate("assign ap_continue_int = 1'b1;","assign ap_continue_int = 1'b0;",flow=True,error="full sequential-init")

    def test_cached_done_requires_fresh_start_suppression(self):
        self.mutate("(ap_done_cache == 1'b1) && (ap_start_int == 1'b0)","(ap_done_cache == 1'b1)",flow=True,error="full sequential-init")


class ActualCompoundResponseTests(unittest.TestCase):
    mutate=ActualPipelineMutationTests.mutate

    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        base=root/"results/hardware_template_coverage_20261013/attempt01/board_packages/t6_maxpool_c9ed9db2a44b_tiled/20261013T041902800762Z/case/hls_project/solution1/syn/verilog"
        if not base.is_dir():raise unittest.SkipTest("actual output20 pool package not available")
        cls.base=base;cls.pipeline=base/"pilot_kernel_pilot_kernel_Pipeline_VITIS_LOOP_208_15.v"
        cls.work=root/"results/hardware_template_coverage_20261013"
        cls.definition="ap_predicate_op128_writeresp_state11 = ((icmp_ln208_4_reg_706_pp0_iter9_reg == 1'd1) & (icmp_ln208_3_reg_702_pp0_iter9_reg == 1'd1));"

    def test_actual_last_row_and_last_logical_column_predicate(self):
        result=inspect_store_pipeline(self.pipeline)
        self.assertEqual(len(result["compound_response_predicate"]["atoms"]),2)
        self.assertEqual((result["response_stage"],result["exit_sentinel_stage"]),(10,9))

    def test_or_cannot_replace_compound_response_and(self):
        self.mutate(self.definition,self.definition.replace(" & "," | "),error="pure conjunction")

    def test_last_section_flags_must_have_same_token_stage(self):
        self.mutate(self.definition,self.definition.replace("icmp_ln208_4_reg_706_pp0_iter9_reg","icmp_ln208_4_reg_706_pp0_iter8_reg"),error="stages are not adjacent")

    def test_false_last_section_flag_is_not_accepted(self):
        self.mutate(self.definition,self.definition.replace("icmp_ln208_4_reg_706_pp0_iter9_reg == 1'd1","icmp_ln208_4_reg_706_pp0_iter9_reg == 1'd0"),error="exact staged last-section flags")


class DriverRecognitionTests(unittest.TestCase):
    def test_comparison_is_not_an_assignment_driver(self):
        text="always @ (*) begin\nif (predicate == 1'b1) begin\nother = 1'b1;\nend\nend\n"+\
             "always @ (*) begin\npredicate = actual_value;\nend\n"
        self.assertIn("predicate = actual_value",block_assigning(always_blocks(text),"predicate")["text"])

    def test_multiple_actual_drivers_are_rejected(self):
        text="always @ (*) begin\npredicate = one;\nend\nalways @ (*) begin\npredicate = two;\nend\n"
        with self.assertRaisesRegex(ValueError,"got 2"):
            block_assigning(always_blocks(text),"predicate")


class LocalComputeHierarchyTests(unittest.TestCase):
    def prepare(self,directory):
        top=directory/"pilot_kernel_compute.v"
        top.write_text("module pilot_kernel_compute();\nendmodule\npilot_kernel_dsp wrapper();\n",encoding="utf-8")
        child=directory/"pilot_kernel_dsp.v"
        child.write_text("(* use_dsp = \"yes\" *) module pilot_kernel_dsp_inner();\nendmodule\n"
                         "module pilot_kernel_dsp();\npilot_kernel_dsp_inner wrapped();\nendmodule\n",encoding="utf-8")
        return top,child

    def test_dsp_module_defined_in_same_file_is_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            top,child=self.prepare(Path(temporary))
            result=inspect_local_compute_tree(top)
            self.assertEqual(len(result["modules"]),2)
            row=next(r for r in result["modules"] if r["module"]==child.name)
            self.assertEqual(row["generated_descendants_defined_in_same_file"],["pilot_kernel_dsp_inner"])

    def test_undefined_descendant_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            top,child=self.prepare(Path(temporary))
            child.write_text("module pilot_kernel_dsp();\npilot_kernel_missing unknown();\nendmodule\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"missing on-chip compute descendant"):
                inspect_local_compute_tree(top)

    def test_nested_axi_in_same_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            top,child=self.prepare(Path(temporary))
            child.write_text(child.read_text().replace("pilot_kernel_dsp_inner();","pilot_kernel_dsp_inner(m_axi_hidden);"),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"contains an AXI interface"):
                inspect_local_compute_tree(top)


class ConditionalFinishTests(unittest.TestCase):
    def arguments(self):
        flag="select_ln100_2_reg_1747";done="grp_finish_r_fu_755_ap_done";block="ap_block_state39_on_subcall_done"
        expression=f"(({flag} == 1'd1) & ({done} == 1'b0))"
        write=f"always @ (*) begin\nif ((({flag} == 1'd1) & (1'b1 == ap_CS_fsm_state39) & (1'b0 == {block}))) begin\noutput_tile_1_we1 = 1'b1;\nend else begin\noutput_tile_1_we1 = 1'b0;\nend\nend"
        driver=dict(line=10,text=f"always @ (posedge ap_clk) begin\nif ((({flag} == 1'd1) & (1'b1 == ap_CS_fsm_state38) & ({done} == 1'b1))) begin\ngrp_finish_r_fu_755_ap_start_reg <= 1'b1;\nend\nend")
        fsm=dict(graph={38:[39],39:[39,40]},phases={38:[2],39:[2],40:[2]})
        return [expression,block,done,39,driver,always_blocks(write),fsm]

    def test_valid_lane_requires_done_before_bank_write(self):
        result=inspect_completion_block(*self.arguments())
        self.assertEqual(result["kind"],"REQUESTED_TAIL_LANE_DONE_BEFORE_GUARDED_BANK_WRITE")

    def test_wrong_bank_valid_flag_cannot_skip_completion(self):
        args=self.arguments();args[5][0]["text"]=args[5][0]["text"].replace("select_ln100_2_reg_1747","select_ln100_3_reg_1810")
        with self.assertRaisesRegex(ValueError,"exactly one guarded output bank"):
            inspect_completion_block(*args)

    def test_request_must_use_same_lane_flag(self):
        args=self.arguments();args[4]["text"]=args[4]["text"].replace("select_ln100_2_reg_1747","select_ln100_3_reg_1810")
        with self.assertRaisesRegex(ValueError,"same flag"):
            inspect_completion_block(*args)

    def test_request_cannot_run_under_store_phase(self):
        args=self.arguments();args[6]["phases"][38]=[3]
        with self.assertRaisesRegex(ValueError,"escapes phase"):
            inspect_completion_block(*args)

    def test_completion_cannot_be_or_valid_flag(self):
        args=self.arguments();args[0]=args[0].replace(" & "," | ")
        with self.assertRaisesRegex(ValueError,"conditional completion disjunction"):
            inspect_completion_block(*args)


if __name__ == "__main__":
    unittest.main()
