"""Compare the generated serial GAP kernel against scalar integer averaging."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hls_lut_builder/scripts'))
from common import render_template

TB = r'''
#include "kernel.cpp"
#include <iostream>
int main() {
    int checked = 0, mismatches = 0;
    for (int frame = 0; frame < 3; ++frame) {
        hls::stream<input_axis_t> ins;
        hls::stream<output_axis_t> outs;
        weight_t weights[1] = {0}; acc_t biases[1] = {0};
        int sums[OUT_CH] = {};
        for (int pixel=0; pixel<IN_H*IN_W; ++pixel) {
            for (int p=0; p<PACKETS; ++p) {
                input_axis_t packet; packet.data = 0;
                for (int lane=0; lane<PACK_CH; ++lane) {
                    int c = p*PACK_CH+lane;
                    int v = frame == 0 ? -128 : (frame == 1 ? 127 : ((pixel*19+c*7)%256)-128);
                    if (c < OUT_CH) sums[c] += v;
                    packet.data.range(lane*8+7,lane*8) = ap_uint<8>(v);
                }
                packet.keep=-1; packet.strb=-1;
                packet.last=pixel==IN_H*IN_W-1 && p==PACKETS-1;
                ins.write(packet);
            }
        }
        test_kernel(ins,outs,weights,biases);
        for (int p=0; p<PACKETS; ++p) {
            if (outs.empty()) return 20;
            auto packet=outs.read();
            if (bool(packet.last)!=(p==PACKETS-1) || packet.keep != decltype(packet.keep)(-1) || packet.strb != decltype(packet.strb)(-1)) {
                std::cout << "framing mismatch frame=" << frame << " packet=" << p << "\n";
                ++mismatches;
            }
            for (int lane=0; lane<PACK_CH; ++lane) {
                int c=p*PACK_CH+lane;
                int expected=c<OUT_CH ? sums[c]/(IN_H*IN_W) : 0;
                data_t got=packet.data.range(lane*8+7,lane*8);
                if (int(got)!=expected) {
                    std::cout << "value mismatch frame=" << frame << " c=" << c << " got=" << int(got) << " expected=" << expected << "\n";
                    ++mismatches;
                }
                ++checked;
            }
        }
        if (!ins.empty() || !outs.empty()) return 21;
    }
    std::cout << "GAP_WITNESS checked=" << checked << " mismatches=" << mismatches << "\n";
    return mismatches ? 1 : 0;
}
'''

def run(output, vitis_include):
    output.mkdir(parents=True, exist_ok=True)
    template = ROOT / 'hls_lut_builder/templates/global_avg_pool_serial.cpp.tmpl'
    results = []
    for channels, resolution in [(1280, 7), (32, 3), (7, 3), (513, 1)]:
        folder = output / f'c{channels}_h{resolution}'
        folder.mkdir(exist_ok=True)
        params = dict(top_function='test_kernel', in_channels=channels, out_channels=channels,
                      feature_h=resolution, feature_w=resolution, array_partition_factor=4,
                      max_axis_bits=4096, pipeline_ii=1)
        (folder / 'kernel.cpp').write_text(render_template(template.read_text(), params))
        (folder / 'tb.cpp').write_text(TB)
        command = ['g++', '-std=c++14', '-O1', '-DAP_INT_MAX_W=8192', '-Wno-unknown-pragmas', '-I'+str(vitis_include),
                   '-I'+str(ROOT/'hls_lut_builder/include'), str(folder/'tb.cpp'), '-o', str(folder/'witness'), '-lgmp']
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (folder/'compile.log').write_text(compiled.stdout+compiled.stderr)
        if compiled.returncode:
            results.append(dict(case=folder.name, status='COMPILE_FAILED', command=command))
            break
        result = subprocess.run([str(folder/'witness')], capture_output=True, text=True, timeout=60)
        (folder/'witness.log').write_text(result.stdout+result.stderr)
        results.append(dict(case=folder.name, status='PASS' if result.returncode==0 else 'MISMATCH',
                            output=result.stdout.strip(), command=command))
    report = dict(status='PASS' if len(results)==4 and all(r['status']=='PASS' for r in results) else 'FAIL',
                  template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(), cases=results,
                  boundary='Integer semantics and AXI framing only; not RTL cosimulation or board validation.')
    (output/'status.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['status']=='PASS' else 1

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--vitis-include', type=Path, default=Path('/opt/Xilinx/Vitis_HLS/2022.2/include'))
    args=parser.parse_args()
    raise SystemExit(run(args.output, args.vitis_include))
