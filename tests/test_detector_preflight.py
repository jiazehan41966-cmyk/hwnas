"""Address lifetime and fail-closed checks, without FPGA or dataset access."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('detector_prepare', ROOT/'tools/detector_preflight/prepare.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load(letter):
    folder = ROOT/f'artifacts/fpga_h2/20260916_v1/graph_{letter}'
    graph = json.loads((folder/'graph.json').read_text())
    sizes = {f: (folder/f).stat().st_size for f in ('weights.bin','biases.bin','layers.bin')}
    return graph, sizes


@pytest.mark.parametrize('letter', list('ABC'))
def test_memory_replay_preserves_fpn_branches_and_all_outputs(letter):
    graph, sizes = load(letter)
    plan = module.plan_graph(graph, sizes)
    memory = bytearray(plan['arena_required_bytes'])
    tags = {'image': 1}

    def write(name, tag):
        a = plan['allocations'][name]
        memory[a['offset']:a['offset']+a['real_bytes']] = bytes([tag])*a['real_bytes']

    def check(name):
        a = plan['allocations'][name]
        assert memory[a['offset']:a['offset']+a['real_bytes']] == bytes([tags[name]])*a['real_bytes']

    write('image', 1)
    for i, node in enumerate(graph['nodes']):
        for source in node['inputs']:
            check(source)
        tags[node['id']] = i+2
        write(node['id'], i+2)
    for name in graph['outputs'].values():
        check(name)
    assert plan['activation_high_water_bytes'] < sum(a['allocated_bytes'] for a in plan['allocations'].values())
    assert all(a['offset'] % 64 == 0 for a in plan['allocations'].values())
    if 'P5' in graph['outputs']:
        assert plan['outputs'][-1]['tail_zero_bytes'] == 16
        assert plan['outputs'][-1]['dma_beats'] == 263
    for region in plan['parameters'].values():
        assert region['offset'] >= plan['activation_high_water_bytes']


@pytest.mark.parametrize('mutation', ['forward_reference', 'duplicate', 'precision', 'parameter', 'descriptor', 'dma_overflow'])
def test_reject_invalid_inputs(mutation):
    graph, sizes = load('C')
    graph = copy.deepcopy(graph)
    first = graph['nodes'][0]
    if mutation == 'forward_reference':
        first['inputs'] = [graph['nodes'][-1]['id']]
    elif mutation == 'duplicate':
        graph['nodes'][1]['id'] = first['id']
    elif mutation == 'precision':
        graph['edges']['image']['bits'] = 16
    elif mutation == 'parameter':
        first['weight_offset'] = sizes['weights.bin']
    elif mutation == 'descriptor':
        graph['descriptors'][0]['padded_bytes'] += 32
    else:
        graph['edges']['image']['shape'] = [1, 8192, 8192]
    with pytest.raises(ValueError):
        module.plan_graph(graph, sizes)


def test_reject_undersized_runtime_arena():
    graph, sizes = load('A')
    plan = module.plan_graph(graph, sizes)
    with pytest.raises(ValueError, match='too small'):
        module.plan_graph(graph, sizes, arena_bytes=plan['arena_required_bytes']-1)
    assert module.plan_graph(graph, sizes, arena_bytes=plan['arena_required_bytes'])['board_ready'] is False
