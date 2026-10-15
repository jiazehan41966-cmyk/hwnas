"""Bind a validated H2 parameter package to the serial-DDR FPGA executor.

Default remains historical synthetic graph A; real packages require the export
manifest. Tensor payloads are HWC; graph shape metadata remains CHW.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from prepare import plan_graph

MAGIC = 0x48324731
OP_CODES = {'conv': 0, 'add': 1, 'upsample_add': 2, 'requant': 4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_package(source):
    source = Path(source)
    graph = json.loads((source / 'graph.json').read_text())
    for filename in ('weights.bin', 'biases.bin', 'layers.bin'):
        if sha(source / filename) != graph['binary_hashes'][filename]:
            raise ValueError('parameter hash mismatch: ' + filename)
    for name in ('numeric_contract', 'interface_contract'):
        value = json.loads((ROOT / 'reports/fpga_h2_contract' / (name + '_v1.json')).read_text())
        expected = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if graph[name + '_hash'] != expected:
            raise ValueError('frozen H2 contract mismatch: ' + name)
    binding = None
    if graph.get('real_weights_bound'):
        if graph.get('synthetic_weights') is not False:
            raise ValueError('real package must explicitly reject synthetic weights')
        path = source / 'export_manifest.json'
        binding = json.loads(path.read_text())
        if binding.get('graph_sha256') != sha(source / 'graph.json'):
            raise ValueError('real package graph SHA mismatch')
        provenance = graph.get('provenance', {})
        for key in ('checkpoint_sha256', 'profile_sha256'):
            digest = provenance.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('missing real package provenance: ' + key)
        if binding.get('provenance') != provenance:
            raise ValueError('export manifest provenance disagrees with graph')
        binding = dict(export_manifest_sha256=sha(path), **binding)
    elif graph.get('synthetic_weights') is False:
        raise ValueError('unauthenticated real-weight package')
    return graph, binding


def make_descriptors(graph, plan, expected_offsets):
    """32-word little-endian ABI; op3 remains DMA, op4 is explicit requant."""
    rows = []
    for node in graph['nodes']:
        first = plan['allocations'][node['inputs'][0]]
        output = plan['allocations'][node['id']]
        second = plan['allocations'][node['inputs'][1]] if len(node['inputs']) == 2 else dict(offset=0, real_bytes=0)
        ci, h, w = first['shape']; co, oh, ow = output['shape']
        if max(ci, co) > 960 or max(h, w, oh, ow) > 640:
            raise ValueError('shape exceeds serial executor capacity')
        words = [0] * 32
        words[:11] = [OP_CODES[node['op']], ci, co, h, w, oh, ow,
                      node.get('kernel', 0), node.get('stride', 0), node.get('padding', 0), node.get('groups', 0)]
        words[11:14] = [first['offset'], second['offset'], output['offset']]
        if node['op'] == 'conv':
            count = int(np.prod(node['weight_shape']))
            if count > 153600 or node['bias_count'] > 960:
                raise ValueError('parameter slice exceeds serial executor capacity')
            words[14:16] = [count, node['bias_count']]
            words[16:20] = [node['multiplier'], node['shift'], {'linear': 0, 'relu': 1, 'relu6': 2}[node['activation']], node['q6']]
            words[27] = plan['parameters']['weights.bin']['offset'] + node['weight_offset']
            words[28] = plan['parameters']['biases.bin']['offset'] + node['bias_offset']
        elif node['op'] == 'requant':
            words[16:18] = [node['multiplier'], node['shift']]
        else:
            words[16:22] = node['numeric']
        words[22:27] = [plan['arena_required_bytes'], first['real_bytes'], second['real_bytes'], output['real_bytes'], MAGIC]
        words[29] = expected_offsets[node['id']]
        rows.append(words)
    return np.asarray(rows, dtype='<i4')


def _reference(source, image, real):
    if real:
        from tools.sonar_deploy.ptq_export import numpy_forward
        return numpy_forward(source, image)[1]
    sys.path.insert(0, str(ROOT / 'tools/fpga_h2'))
    from integer_reference import forward
    return forward(source, image)[1]


def build(output, package=None, input_path=None, values=None):
    source = Path(package or ROOT / 'artifacts/fpga_h2/20260916_v1/graph_A').resolve()
    graph, binding = validate_package(source)
    raw = {f: (source / f).read_bytes() for f in ('weights.bin', 'biases.bin', 'layers.bin')}
    plan = plan_graph(graph, {k: len(v) for k, v in raw.items()})
    image_path = Path(input_path or source / 'input.npy')
    image = np.load(image_path, allow_pickle=False)
    if image.dtype != np.int8 or list(image.shape) != graph['edges']['image']['shape']:
        raise ValueError('input must be quantized INT8 CHW matching graph')
    gold = values if values is not None else _reference(source, image, bool(binding))
    if not np.array_equal(gold['image'], image):
        raise ValueError('reference input mismatch')
    arena = bytearray(plan['arena_required_bytes'])
    for name, data in raw.items():
        offset = plan['parameters'][name]['offset']; arena[offset:offset + len(data)] = data
    data = image.transpose(1, 2, 0).copy().tobytes()
    offset = plan['allocations']['image']['offset']; arena[offset:offset + len(data)] = data
    expected = bytearray(); expected_offsets = {}; rows = []
    for node in graph['nodes']:
        value = np.asarray(gold[node['id']])
        if value.dtype != np.int8 or list(value.shape) != graph['edges'][node['id']]['shape']:
            raise ValueError('reference node shape/dtype mismatch: ' + node['id'])
        data = value.transpose(1, 2, 0).copy().tobytes()
        expected_offsets[node['id']] = len(expected); expected.extend(data)
        rows.append(dict(node=node['id'], operator=node['op'], output_sha256=hashlib.sha256(data).hexdigest()))
    desc = make_descriptors(graph, plan, expected_offsets)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out / 'arena.bin').write_bytes(arena)
    desc.tofile(out / 'descriptors.bin')
    (out / 'expected_layers.bin').write_bytes(expected)
    shutil.copy2(source / 'graph.json', out / 'graph.json')
    if binding: shutil.copy2(source / 'export_manifest.json', out / 'export_manifest.json')
    for name in ('h2_graph.cpp', 'tb_graph.cpp', 'pynq_graph_runner.py'):
        shutil.copy2(Path(__file__).parent / name, out / name)
    shutil.copy2(ROOT / 'tools/fpga_h2/numeric_v1.h', out / 'numeric_v1.h')
    (out / 'run_graph.tcl').write_text('''open_project hls_project
set_top h2_graph
add_files h2_graph.cpp -cflags {-std=c++14 -I.}
add_files -tb tb_graph.cpp -cflags {-std=c++14 -I.}
foreach name {arena.bin descriptors.bin expected_layers.bin} {add_files -tb $name}
open_solution solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5
config_interface -m_axi_addr64
config_schedule -enable_dsp_full_reg=false
csim_design -O
csynth_design
export_design -format ip_catalog -vendor hwnas.org -library hls -version 1.0
exit
''')
    bd = (Path(__file__).parent / 'build_dma_bd.tcl').read_text().replace('h2_dma', 'h2_graph')
    bd = bd.replace('scoped H2 DMA bring-up only; not full detector',
                    'serial DDR graph executor timing PASS; candidate board validation pending')
    bd = bd.replace('report_utilization -hierarchical',
                    'report_utilization -file [file join $root utilization.rpt]\nreport_utilization -hierarchical')
    (out / 'build_graph_bd.tcl').write_text(bd)
    plan['candidate_weights_bound'] = bool(binding)
    scope = 'trained PTQ serial DDR functional executor; not H3 mapping' if binding else 'synthetic H2 serial DDR functional witness; not a trained candidate'
    manifest = dict(scope=scope, graph_sha256=sha(source / 'graph.json'), plan=plan, layers=rows,
                    source_package_binary_hashes=graph['binary_hashes'], input_sha256=sha(image_path),
                    real_weights_bound=bool(binding), export_binding=binding, test_evaluated=False,
                    implementation=dict(mul='default', requested_clock_mhz=200, dsp_full_registers=False,
                        executor_version='serial_ddr_graph_v2_requant_timing02', requant_pipeline='off',
                        supported_opcodes=[0,1,2,3,4],
                        arena_capacity_bytes=plan['executor_arena_capacity_bytes']))
    manifest['files'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    manifest = build(args.output, args.package, args.input)
    print(json.dumps(dict(nodes=len(manifest['layers']), arena_bytes=manifest['plan']['arena_required_bytes'],
                         real_weights_bound=manifest['real_weights_bound'])))


if __name__ == '__main__':
    main()
