"""Bounded timer-only RTL simulation; never synthesize or access a board.

Run with the project Python: -m unittest tools.hardware_template_coverage_v1.test_phase_timer -v
If no supported simulator is installed, unittest records SKIP, never RTL PASS.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("phase_timer.v")
TESTBENCH = r"""`timescale 1ns / 1ps
module tb;
reg clk = 0, aresetn = 0, phase_valid = 0;
reg [7:0] phase = 0;
wire [31:0] compute_cycles, transfer_cycles, total_cycles, sequence_count, identity, status;
phase_timer dut(.clk(clk), .aresetn(aresetn), .phase(phase), .phase_valid(phase_valid),
 .compute_cycles(compute_cycles), .transfer_cycles(transfer_cycles),
 .total_cycles(total_cycles), .sequence_count(sequence_count), .identity(identity), .status(status));
always #2.5 clk = ~clk;
task step(input [7:0] value);
 begin @(negedge clk); phase = value; phase_valid = 1; @(posedge clk); #1; end
endtask
task hold_noise(input [7:0] value);
 begin @(negedge clk); phase = value; phase_valid = 0; @(posedge clk); #1; end
endtask
task check(input [31:0] c, t, total, seq, flags);
 begin
  if (compute_cycles !== c || transfer_cycles !== t || total_cycles !== total ||
      sequence_count !== seq || status !== flags || identity !== 32'h50335431)
   $fatal(1, "timer mismatch: c=%0d t=%0d total=%0d seq=%0d status=%0h, expected %0d %0d %0d %0d %0h",
    compute_cycles, transfer_cycles, total_cycles, sequence_count, status, c,t,total,seq,flags);
 end
endtask
task reset_timer;
 begin @(negedge clk); aresetn=0; phase=0; phase_valid=0; @(posedge clk); #1;
 check(0,0,0,0,0); @(negedge clk); aresetn=1; end
endtask
initial begin
 reset_timer();
 // Invalid output data does not constitute a phase write, including at idle.
 hold_noise(3); hold_noise(255); check(0,0,0,0,0);
 // Three load, five compute, two store cycles. First active cycle is counted.
 step(1); check(0,0,0,0,2);
 repeat(2) hold_noise(255); step(2); repeat(4) hold_noise(3); step(3); hold_noise(0);
 check(0,0,0,0,2); step(0); check(5,5,10,1,1);
 repeat(7) begin hold_noise(3); check(5,5,10,1,1); end
 // Published results remain stable while a new invocation is active.
 step(1); check(5,5,10,1,2); step(2); step(3); step(0); check(1,2,3,2,1);
 // One invocation has two complete tiles, committed only once at final idle.
 step(1); step(2); step(3); step(1); check(1,2,3,2,2);
 step(2); step(3); step(0); check(2,4,6,3,1);
 // Cross-tile holds accumulate and invalid data does not terminate a tile.
 step(1); hold_noise(0); step(2); hold_noise(1); step(3); hold_noise(0);
 step(1); step(2); step(3); step(0); check(3,6,9,4,1);
 // Missing load, missing compute, reverse transition, unknown phase all fail.
 step(2); step(3); step(0); check(1,1,2,5,8);
 step(1); step(3); step(0); check(0,2,2,6,8);
 step(1); step(2); step(1); step(3); step(0); check(1,3,4,7,8);
 step(4); step(0); check(0,0,1,8,8);
 // Early termination fails; reset during active work discards partial results.
 step(1); step(2); step(0); check(1,1,2,9,8);
 // An incomplete later tile cannot commit as a valid invocation.
 step(1); step(2); step(3); step(1); step(2); step(0); check(2,3,5,10,8);
 step(1); step(2); reset_timer(); step(0); check(0,0,0,0,0);
 // Inject near-overflow internal counters to test overflow without 2^32 cycles.
 step(1); dut.work_compute=32'hffffffff; dut.work_total=32'hffffffff;
 step(2); step(3); step(0); check(32'hffffffff,2,32'hffffffff,1,4);
 reset_timer(); step(1); dut.work_transfer=32'hffffffff;
 step(2); step(3); step(0); check(1,32'hffffffff,3,1,4);
 // Counter overflow belongs only to its invocation; subsequent run can pass.
 step(1); step(2); step(3); step(0); check(1,2,3,2,1);
 // Sequence overflow saturates and remains invalid until hardware reset.
 dut.sequence_count=32'hffffffff;
 step(1); step(2); step(3); step(0); check(1,2,3,32'hffffffff,16);
 step(1); step(2); step(3); step(0); check(1,2,3,32'hffffffff,16);
 reset_timer(); step(1); step(2); step(3); step(0); check(1,2,3,1,1);
 $display("PHASE_TIMER_RTL_PASS"); $finish;
end
initial begin #100000; $fatal(1,"testbench watchdog expired"); end
endmodule
"""


class PhaseTimerTests(unittest.TestCase):
    def test_directed_rtl(self):
        iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
        candidates = [Path(value) for value in [os.environ.get("VIVADO_BIN", ""),
            "F:/vivado/Vivado/2023.2/bin", "/opt/Xilinx/Vivado/2022.2/bin"] if value]
        xsim_on_path = shutil.which("xsim")
        if xsim_on_path:
            candidates.insert(0, Path(xsim_on_path).parent)
        suffix = ".bat" if os.name == "nt" else ""
        vivado = next((path for path in candidates if all((path / (name + suffix)).is_file()
                                                       for name in ("xvlog", "xelab", "xsim"))), None)
        if not (iverilog and vvp) and vivado is None:
            self.skipTest("iverilog/vvp and Vivado xvlog/xelab/xsim unavailable; RTL NOT_RUN")
        stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
        output = ROOT / "results/hardware_template_coverage_20261013/timer_validation" / stamp
        output.mkdir(parents=True, exist_ok=False)
        tb = output / "tb.sv"
        tb.write_text(TESTBENCH, encoding="utf-8")
        if iverilog and vvp:
            commands = [[iverilog, "-g2012", "-s", "tb", "-o", str(output / "timer.vvp"), str(SOURCE), str(tb)],
                        [vvp, str(output / "timer.vvp")]]
            simulator = "iverilog"
        else:
            commands = [[str(vivado / ("xvlog" + suffix)), "--sv", str(SOURCE), str(tb)],
                        [str(vivado / ("xelab" + suffix)), "tb", "-s", "phase_timer_test", "-mt", "2"],
                        [str(vivado / ("xsim" + suffix)), "phase_timer_test", "-runall"]]
            simulator = "vivado_xsim"
        records = []
        report = dict(status="FAIL", simulator=simulator, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      testbench_sha256=hashlib.sha256(tb.read_bytes()).hexdigest(), commands=records,
                      scope="TIMER_RTL_ONLY_HLS_INTEGRATION_AND_BOARD_NOT_RUN")
        try:
            for index, command in enumerate(commands):
                result = subprocess.run(command, cwd=output, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=120)
                (output / f"stage_{index}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
                records.append(dict(command=command, returncode=result.returncode, log=f"stage_{index}.log"))
                self.assertEqual(result.returncode, 0, msg=f"simulator failed; see {output}")
            self.assertIn("PHASE_TIMER_RTL_PASS", result.stdout + result.stderr, msg=f"RTL assertions did not pass; see {output}")
            report["status"] = "PASS_TIMER_RTL"
        finally:
            (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Timer RTL witness: {output / 'validation.json'}")

    def test_tcl_complete_and_helper_escaping(self):
        try:
            import tkinter
        except ImportError:
            self.skipTest("Tcl interpreter unavailable; shell script syntax NOT_RUN")
        interpreter = tkinter.Tcl()
        source = Path(__file__).with_name("build_shell.tcl").read_text(encoding="utf-8")
        self.assertEqual(int(interpreter.call("info", "complete", source)), 1)
        # Parse/evaluate only helpers, without invoking any Vivado command.
        helpers = source[source.index("proc read_text"):source.index("timed shell_generation")]
        interpreter.eval(helpers)
        for value in ["simple", "a\\b", 'a"b', "a\nb\tc"]:
            quoted = interpreter.call("json_quote", value)
            self.assertEqual(json.loads(quoted), value)
        # Verify the precise report regexes against representative Vivado rows.
        interpreter.eval(r'set sample "# of routable nets........ : 124"')
        interpreter.eval(r'set label "routable nets"')
        interpreter.eval(r'regexp "# of $label\\.+\\s*:\\s*(\[0-9\]+)" $sample ignored count')
        self.assertEqual(interpreter.getvar("count"), "124")


if __name__ == "__main__":
    unittest.main()
