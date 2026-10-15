"""Prepare only (do not launch) matched C/RTL numeric regression for real PTQ."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def build(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for name in ('h2_graph.cpp','tb_graph_descriptor.cpp','tb_graph_numeric.cpp'):
        shutil.copy2(Path(__file__).parent/name,out/name)
    shutil.copy2(ROOT/'tools/fpga_h2/numeric_v1.h',out/'numeric_v1.h')
    (out/'run.tcl').write_text('''open_project hls_project
set_top h2_graph
add_files h2_graph.cpp -cflags {-std=c++14 -I. -DH2_SMALL_COSIM}
add_files -tb tb_graph_numeric.cpp -cflags {-std=c++14 -I.}
open_solution solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5
config_interface -m_axi_addr64
config_schedule -enable_dsp_full_reg=false
csim_design -O
csynth_design
cosim_design -rtl verilog -trace_level none
exit
''')
    report=dict(status='PREPARED_NOT_EXECUTED',test_evaluated=False,
                scope='requant, wrap32, DW3/DW5 stride1/2, ReLU6, legacy descriptor regression',
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    (out/'regression_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(build(parser.parse_args().output),indent=2))
