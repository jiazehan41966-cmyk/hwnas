"""Candidate-independent DDR/scheduler preparation; never programs a board."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERIAL_ARENA_CAPACITY = 16777216  # measured B1/C/D plans fit the existing 16 MiB limit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align(n, boundary=64):
    return (n + boundary - 1) // boundary * boundary


def plan_graph(graph, binary_sizes, arena_bytes=None):
    """Allocate relative offsets, retaining branches until their last consumer.

    This H2-v1 adapter accepts INT8 only. Offsets are NOT physical addresses.
    Hardware execution of a step requires a future matching kernel backend.
    """
    edges, nodes = graph['edges'], graph['nodes']
    if set(graph['inputs']) != {'image'}:
        raise ValueError('expected one image input')
    born = {'image': -1}
    last = {'image': -1}
    for i, node in enumerate(nodes):
        name = node['id']
        if name in born or name not in edges:
            raise ValueError('duplicate or missing edge')
        arity = {'conv': 1, 'add': 2, 'upsample_add': 2, 'requant': 1}.get(node['op'])
        if arity is None or len(node['inputs']) != arity:
            raise ValueError('unsupported operator/arity')
        for source in node['inputs']:
            if source not in born:
                raise ValueError('graph not topologically ordered')
            last[source] = i
        if node['op'] == 'requant':
            source = edges[node['inputs'][0]]
            if source['shape'] != edges[name]['shape']:
                raise ValueError('requant must preserve CHW shape')
            if node.get('input_exp') != source['exp'] or node.get('output_exp') != edges[name]['exp']:
                raise ValueError('requant exponent disagrees with edge')
            m, s = node.get('multiplier'), node.get('shift')
            if type(m) is not int or not 1 <= m < 2**31 or type(s) is not int or not 0 <= s <= 31:
                raise ValueError('requant multiplier/shift outside NUMERIC_CONTRACT_V1')
            if m * 2.0 ** (-s) != 2.0 ** (node['input_exp'] - node['output_exp']):
                raise ValueError('requant scale does not match exponent ratio')
        born[name] = last[name] = i
    for name in graph['outputs'].values():
        if name not in born:
            raise ValueError('unknown output')
        last[name] = len(nodes)  # preserve all heads until CPU consumes them

    allocations = {}
    high = 0
    for name, birth in born.items():
        shape = edges[name]['shape']
        if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('invalid CHW shape')
        if edges[name].get('bits', 8) != 8:
            raise ValueError('H2-v1 adapter is INT8 only')
        real = math.prod(shape)
        padded = align(real, 32)
        if padded >= 2**26:
            raise ValueError('DMA length exceeds shell 26-bit field')
        size = align(padded)
        occupied = sorted((a['offset'], a['offset'] + a['allocated_bytes'])
                          for a in allocations.values() if a['last_use'] >= birth)
        offset = 0
        for start, end in occupied:
            if offset + size <= start:
                break
            offset = max(offset, end)
        allocations[name] = dict(offset=offset, shape=shape, real_bytes=real,
            padded_bytes=padded, allocated_bytes=size, born=birth, last_use=last[name])
        high = max(high, offset + size)
    regions = {}
    cursor = align(high)
    for filename in ('weights.bin', 'biases.bin', 'layers.bin'):
        size = binary_sizes[filename]
        if type(size) is not int or size <= 0:
            raise ValueError('invalid binary size')
        regions[filename] = dict(offset=cursor, bytes=size, allocated_bytes=align(size))
        cursor += align(size)
    if arena_bytes is not None and (type(arena_bytes) is not int or cursor > arena_bytes):
        raise ValueError('caller-provided DMA arena is too small')
    if cursor > SERIAL_ARENA_CAPACITY:
        raise ValueError('serial graph executor arena capacity exceeded')

    steps = []
    for i, node in enumerate(nodes):
        params = {}
        if node['op'] == 'conv':
            for filename, key, count in (
                ('weights.bin', 'weight_offset', math.prod(node['weight_shape'])),
                ('biases.bin', 'bias_offset', node['bias_count'] * 4),
                ('layers.bin', None, 128)):
                offset = node[key] if key else node['layer_id'] * 128
                if offset < 0 or offset + count > binary_sizes[filename]:
                    raise ValueError('parameter slice outside binary')
                params[filename] = dict(offset=regions[filename]['offset'] + offset, bytes=count)
        steps.append(dict(index=i, node=node['id'], operator=node['op'],
            inputs=[dict(edge=s, **allocations[s]) for s in node['inputs']],
            output=dict(edge=node['id'], **allocations[node['id']]), parameters=params,
            backend='UNIMPLEMENTED', completion='kernel_done_and_all_transfers_done_no_errors',
            timeout_required=True))
    outputs = []
    levels = list(graph['outputs'])
    if tuple(levels) not in [('P3','P4','P5'), ('P2','P3','P4'), ('P2','P3','P4','P5')]:
        raise ValueError('unsupported output scale order')
    desc = {d['name']: d for d in graph['descriptors']}
    for level, name in graph['outputs'].items():
        a, d = allocations[name], desc[level]
        if a['shape'] != [d['channels'], d['height'], d['width']] or (
            a['real_bytes'], a['padded_bytes']) != (d['real_bytes'], d['padded_bytes']):
            raise ValueError('H2 descriptor mismatch')
        outputs.append(dict(level=level, edge=name, arena_offset=a['offset'],
            packed_export_offset=d['ddr_offset'], real_bytes=a['real_bytes'],
            padded_bytes=a['padded_bytes'], dma_beats=a['padded_bytes']//32,
            tail_zero_bytes=a['padded_bytes']-a['real_bytes']))
    return dict(status='HOST_PLAN_ONLY', precision='H2_V1_INT8',
        addressing='relative_offsets_not_physical_addresses', arena_required_bytes=cursor,
        executor_arena_capacity_bytes=SERIAL_ARENA_CAPACITY,
        activation_high_water_bytes=high, allocations=allocations, parameters=regions,
        steps=steps, outputs=outputs, shell_reservation='UNKNOWN',
        board_ready=False, candidate_weights_bound=False,
        cache_protocol=['CPU writes input and parameter packages; zero frame tail padding',
            'flush CPU-written input/parameters before device ownership',
            'clean/invalidate output arena before first device write; barrier',
            'device-owned intermediates: no CPU reads or writes between kernel steps',
            'wait with timeout for all DMA and kernel completions; reject errors',
            'invalidate output ranges after device completion before CPU read'],
        backend_requirements=['arm receive path before starting producer',
            'conv: matching parameter loader and H2 arithmetic',
            'add/upsample_add: two independently addressed input sources',
            'preserve low-resolution FPN source in DDR until consumed',
            '256-bit repacker with full masks and frame-end zero padding',
            'no use of fixed-784-word shell ABI as a dynamic descriptor ABI'])


def prepare(output):
    output = Path(output)
    contracts = ROOT / 'reports/fpga_h2_contract'
    hashes = {k: hashlib.sha256(json.dumps(json.loads((contracts/f'{k}_v1.json').read_text()),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for k in ('numeric_contract', 'interface_contract')}
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__),
        ROOT/'hardware/zcu104_dma_shell_v1/rtl/operator_benchmark_wrapper.v',
        ROOT/'hardware/zcu104_dma_shell_v1/vivado/build_bd.tcl',
        ROOT/'hardware/zcu104_dma_shell_v1/contracts/platform_contract.json']
    summaries = {}
    for letter in 'ABC':
        folder = ROOT / f'artifacts/fpga_h2/20260916_v1/graph_{letter}'
        path = folder/'graph.json'
        graph = json.loads(path.read_text())
        for key, digest in hashes.items():
            if graph[key+'_hash'] != digest:
                raise ValueError('H2 contract hash mismatch')
        for filename, digest in graph['binary_hashes'].items():
            if sha(folder/filename) != digest:
                raise ValueError('H2 binary hash mismatch')
        sizes = {f: (folder/f).stat().st_size for f in ('weights.bin','biases.bin','layers.bin')}
        plan = plan_graph(graph, sizes)
        plan['graph_sha256'] = sha(path)
        plan['binary_hashes'] = graph['binary_hashes']
        plan['contract_hashes'] = hashes
        (output/f'graph_{letter}_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        summaries[letter] = {k: plan[k] for k in ('arena_required_bytes','activation_high_water_bytes','status')}
        source_paths += [path] + [folder/f for f in sizes]
    manifest = dict(scope='candidate-independent preflight using synthetic H2 packages',
        graphs=summaries, sources={p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
        board_ready=False, hardware_execution_started=False, search_modified=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
