"""Cost-only graphs for the frozen 144-space, derived from H2 topology witnesses.

No weights, calibration, accuracy, or new architecture definition is introduced.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'configs/search_space_sonar_v0_1.yaml'
SPEC_SHA = 'ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4'
TEMPLATES = dict(P3_P4_P5='A', P2_P3_P4='B', P2_P3_P4_P5='C')
NODE_FIELDS = ('id','op','inputs','Cin','Cout','H','W','kernel','stride','padding','groups','activation')

def architectures():
    raw=SPEC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPEC_SHA:
        raise ValueError('Frozen search space changed')
    return deepcopy(_architectures(raw))

@lru_cache(maxsize=1)
def _architectures(raw):
    space = yaml.safe_load(raw)['architecture_space']
    result = {}
    for combo in space['output_combinations']:
        stages = [s for s in space['stages'] if s['op']=='mbconv' and s['index']<=combo['last_active_stage']]
        for ks in itertools.product(*(s['kernel_choices'] for s in stages)):
            kernels = {f"stage{s['index']}":k for s,k in zip(stages,ks)}
            aid = combo['id'] + '_' + '_'.join(f"s{s['index']}k{k}" for s,k in zip(stages,ks))
            result[aid] = dict(output_combination=combo['id'], kernels=kernels)
    if len(result)!=space['unique_architecture_count']:
        raise ValueError('Architecture cardinality mismatch')
    return result

def topology(graph):
    return dict(architecture=graph['architecture'],
        nodes=[{k:n[k] for k in NODE_FIELDS if k in n} for n in graph['nodes']],
        edges={k:dict(shape=v['shape']) for k,v in graph['edges'].items()},
        outputs=graph['outputs'], descriptors=graph['descriptors'])

@lru_cache(maxsize=3)
def _template(raw):
    return topology(json.loads(raw))

def candidate_graph(aid, hashes):
    arch = architectures().get(aid)
    if arch is None:
        raise ValueError('Architecture outside frozen V0.1')
    path = ROOT / f"artifacts/fpga_h2/20260916_v1/graph_{TEMPLATES[arch['output_combination']]}/graph.json"
    graph = deepcopy(_template(path.read_bytes()))
    graph['architecture'] = deepcopy(arch)
    for n in graph['nodes']:
        match = re.fullmatch(r'backbone.stages.(\d+).0.dw_conv', n['id'])
        if match:
            k=arch['kernels']['stage'+match[1]]
            n.update(kernel=k,padding=k//2)
    graph.update(hashes, graph_kind='cost_topology_only', search_space_sha256=SPEC_SHA)
    return graph

def identify(graph, hashes):
    arch=graph.get('architecture',{})
    kernels=arch.get('kernels',{})
    aid=arch.get('output_combination','')+'_'+'_'.join(
        f's{s}k{kernels[f"stage{s}"]}' for s in (1,2,4,5,6,7) if f'stage{s}' in kernels)
    if aid not in architectures():
        return None
    expected=candidate_graph(aid,hashes)
    # Numerical/weight packages must be registered separately. Do not accept
    # changed frozen weight graphs just because their topology is unchanged.
    return aid if graph == expected else None
