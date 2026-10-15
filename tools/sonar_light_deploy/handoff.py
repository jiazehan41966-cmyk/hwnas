"""Inspect existing software integer artifacts; never launch HLS or program a board.

This module validates provenance and the saved export, not model accuracy or
hardware correctness. Hardware reports cannot promote a stage using a PASS flag.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch

from tools.modnas_elastic.model import (
    SEARCH_SPACE_VERSION, StaticElasticNeckDetector, architecture_id, architecture_hash,
    validate_architecture,
)
from tools.modnas_elastic.cost_graph import build_cost_graph, canonical_hash
from tools.modnas_comparison.quantization import model_state_hash
from tools.modnas_comparison.quantization_mixed import _contracts
from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH
from hwnas_fpga.deploy.h2_ptq import ratio_from_exponents

PROFILES = ('W8A8', 'HEAD_W8A16', 'HEAD_W16A16')
MAPPINGS = ('reference_low', 'reference_high', 'reference_min_latency',
            'reference_min_dsp', 'reference_min_lut', 'reference_min_bram')
REQUIRED_EXPORTS = ('profile.json', 'numeric_graph.json', 'integer_parameters.pt',
                    'software_parity_witness.json', 'extrema.json')
HARDWARE_STAGES = {
    'hardware_parameter_package': 'New graph frame/loader/ABI integration and mapping binding required',
    'hls_integer_parity': 'Whole selected graph and profile need independent HLS numerical validation',
    'route': 'Full selected mapping needs routed resource and timing evidence',
    'board_parity': 'Selected checkpoint and bitstream need full-graph board numerical replay',
    'valid_accuracy': 'Frozen full-valid board evaluation and software accuracy comparison required',
    'latency': 'Repeated end-to-end latency measurements against frozen budget required',
    'power': 'Raw power traces, measurement scope, energy and budget verification required',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(reference, base_dir):
    require(isinstance(reference, dict) and reference.get('path'), 'missing artifact reference')
    path = (Path(base_dir) / reference['path']).resolve()
    require(path.is_file(), 'artifact missing: ' + str(path))
    require(sha(path) == reference.get('sha256'), 'artifact SHA256 mismatch: ' + str(path))
    return path


def _match(mapping, expected, context):
    for key, value in expected.items():
        require(mapping.get(key) == value, context + ' mismatch: ' + key)


def inspect_ptq_directory(directory):
    """Validate a saved existing PTQ package, including all node dtypes and parity.

    No inference or calibration is performed. Saved parity is verified for
    completeness and binding, not re-executed. A diagnostic package is acceptable
    for inspection only; final checkpoint identity is checked by prepare_handoff.
    """
    folder = Path(directory).resolve()
    manifest = read_json(folder / 'artifact_manifest.json')
    files = manifest['files']
    for name in REQUIRED_EXPORTS:
        require(name in files, 'missing export manifest entry: ' + name)
    for name, expected in files.items():
        require(Path(name).name == name, 'manifest traversal or nested export entry')
        require(sha(folder / name) == expected, 'export SHA256 mismatch: ' + name)
    profile, graph, witness = [read_json(folder / name) for name in
                               ('profile.json', 'numeric_graph.json', 'software_parity_witness.json')]
    q = profile['precision_profile']
    require(q in PROFILES, 'unsupported profile')
    arch = validate_architecture(profile['architecture'])
    expected_graph = build_cost_graph(arch, num_classes=profile['num_classes'])
    _match(graph, {'architecture': arch, 'architecture_sha256': architecture_hash(arch),
                   'num_classes': profile['num_classes'], 'input_shape': [1, 640, 640]}, 'numeric graph')
    require(profile['cost_graph_sha256'] == expected_graph['cost_graph_sha256'], 'cost topology mismatch')
    graph_body = deepcopy(graph)
    graph_hash = graph_body.pop('numeric_graph_sha256')
    require(canonical_hash(graph_body) == graph_hash == profile['numeric_graph_sha256'], 'numeric graph hash mismatch')
    contract = EXPECTED_HASH if q == 'W8A8' else _contracts(q)['candidate']['quant_contract_hash']
    require(profile['numeric_contract_hash'] == contract, 'numeric contract mismatch')
    _match(profile, {'rounding': 'nearest_even', 'zero_point': 0,
                     'overflow': 'twos_complement_modulo_2pow32'}, 'arithmetic contract')
    regions = None if q == 'W8A8' else _contracts(q)
    if regions is not None:
        require(profile['numeric_contract_regions'] == graph['numeric_contract_regions'] == regions,
                'mixed numeric/interface regions mismatch')
    calibration = deepcopy(profile['calibration'])
    calibration_hash = calibration.pop('calibration_sha256')
    require(canonical_hash(calibration) == calibration_hash, 'calibration manifest hash mismatch')
    _match(calibration, {'split': 'train', 'test_accessed': False, 'test_evaluated': False,
                         'valid_used_for_selection': False, 'augmentation': False}, 'calibration')
    require(len(calibration['entries']) == calibration['number_of_calibration_images'] == 64,
            'expected frozen train64 calibration')
    for entry in calibration['entries']:
        for field in ('image', 'annotation'):
            relative = Path(entry[field])
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'train',
                    'calibration path is not train-only')
            require(sha(Path(calibration['dataset_root']) / relative) == entry[field + '_sha256'],
                    'calibration source changed: ' + entry[field])
    for source, source_hash in profile['adapter_sources_sha256'].items():
        require(sha(source) == source_hash, 'PTQ adapter/contract source drift: ' + source)
    package = torch.load(folder / 'integer_parameters.pt', map_location='cpu', weights_only=True)
    require(package['graph'] == graph, 'parameter package graph mismatch')
    require(package['numeric_contract_hash'] == contract, 'parameter package numeric contract mismatch')
    nodes = {node['id']: node for node in graph['nodes']}
    require(len(nodes) == len(graph['nodes']), 'duplicate graph nodes')
    base_nodes = {node['id']: node for node in expected_graph['nodes']}
    boundaries = {name: node for name, node in nodes.items() if node['op'] == 'requantize'}
    require(set(nodes) - set(boundaries) == set(base_nodes), 'numeric topology adds/omits a primitive')
    require(len(boundaries) == (0 if q == 'W8A8' else 4), 'mixed head requires exactly four boundary nodes')
    require(graph['outputs'] == expected_graph['outputs'], 'detection heads changed or dropped')
    convs, frames, parameter_hash_rows = set(), [], {}
    available = {'image': (8, [1, 640, 640])}
    for name, node in nodes.items():
        is_head = name in graph['outputs'].values()
        bits = 16 if q != 'W8A8' and (is_head or name in boundaries) else 8
        require(node.get('output_bits', 8) == bits, 'output bitwidth mismatch: ' + name)
        for src in node['inputs']:
            require(src in available, 'invalid graph order or source: ' + name)
        if name not in boundaries:
            base = base_nodes[name]
            for key in ('op', 'output_shape', 'input_shapes', 'kernel', 'groups', 'stride', 'padding', 'activation'):
                require(node.get(key) == base.get(key), 'primitive geometry mismatch: ' + name + '.' + key)
            if q != 'W8A8' and is_head:
                require(len(node['inputs']) == 1 and node['inputs'][0] in boundaries, 'head missing conversion')
                boundary = boundaries[node['inputs'][0]]
                require(boundary['inputs'] == base['inputs'] and boundary['output_shape'] == base['input_shapes'][0],
                        'boundary source/geometry mismatch')
                require(node['input_exp'] == boundary['output_exp'], 'mixed boundary scale mismatch')
            else:
                require(node['inputs'] == base['inputs'], 'primitive connectivity mismatch: ' + name)
        else:
            require(node['input_bits'] == 8 and node['output_bits'] == 16, 'boundary bitwidth mismatch')
            require(len(node['inputs']) == 1 and available[node['inputs'][0]][0] == 8, 'boundary must consume body INT8')
            source_exp = graph['input_exp'] if node['inputs'][0] == 'image' else nodes[node['inputs'][0]]['output_exp']
            require(node['input_exp'] == source_exp, 'boundary source scale mismatch')
            require((node['multiplier'], node['shift']) == ratio_from_exponents(node['input_exp'], node['output_exp']),
                    'boundary multiplier/shift mismatch')
            require(node['numeric_contract_hash'] == regions['boundary_numeric_contract_hash'], 'boundary contract mismatch')
        if node['op'] == 'conv':
            convs.add(name)
            weight, bias = package['weights'][name], package['biases'][name]
            wb = 16 if q == 'HEAD_W16A16' and is_head else 8
            require(weight.dtype == (torch.int16 if wb == 16 else torch.int8), 'weight dtype mismatch: ' + name)
            require(bias.dtype == torch.int32 and list(bias.shape) == [node['Cout']], 'bias dtype/shape mismatch')
            require(list(weight.shape) == [node['Cout'], node['Cin'] // node['groups'], node['kernel'], node['kernel']],
                    'weight geometry mismatch: ' + name)
            require(node.get('weight_bits', 8) == wb and node.get('input_bits', 8) == available[node['inputs'][0]][0],
                    'convolution bitwidth mismatch: ' + name)
            require((node['multiplier'], node['shift']) == ratio_from_exponents(node['input_exp'] + node['weight_exp'], node['output_exp']),
                    'convolution multiplier/shift mismatch: ' + name)
            parameter_hash_rows[name] = {'weight_sha256': hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                                         'bias_sha256': hashlib.sha256(bias.numpy().tobytes()).hexdigest()}
        available[name] = bits, node['output_shape']
        frames.append({'node': name, 'shape_chw': node['output_shape'], 'bits': bits,
                       'bytes_per_image': prod(node['output_shape']) * bits // 8})
    require(set(package['weights']) == set(package['biases']) == convs, 'parameter node set mismatch')
    require(canonical_hash(parameter_hash_rows) == profile['integer_parameters_sha256'], 'integer parameter hash mismatch')
    _match(witness, {'status': 'PASS_SOFTWARE_INTEGER_FAKE_PARITY', 'mismatches': 0,
                     'numeric_graph_sha256': graph_hash, 'calibration_sha256': calibration_hash,
                     'original_model_state_sha256': profile['original_model_state_sha256'],
                     'integer_parameters_sha256': profile['integer_parameters_sha256'],
                     'test_evaluated': False}, 'software witness')
    require(type(witness['input_images']) is int and witness['input_images'] > 0, 'empty software witness')
    rows = {row['node']: row for row in witness['rows']}
    require(len(rows) == len(witness['rows']) and set(rows) == set(available), 'software witness missing nodes')
    for name, row in rows.items():
        require(row['mismatch_count'] == 0 and row['integer_sha256'] == row['fake_quantized_sha256'],
                'software parity mismatch: ' + name)
        require(row['element_count'] == prod(available[name][1]) * witness['input_images'], 'witness element count mismatch')
    return {'status': 'VERIFIED_SAVED_SOFTWARE_EXPORT', 'profile': q, 'architecture': arch,
            'architecture_id': architecture_id(arch), 'architecture_hash': architecture_hash(arch),
            'num_classes': profile['num_classes'], 'calibration_sha256': calibration_hash,
            'numeric_graph_sha256': graph_hash, 'original_model_state_sha256': profile['original_model_state_sha256'],
            'integer_parameters_sha256': profile['integer_parameters_sha256'],
            'manifest_sha256': sha(folder / 'artifact_manifest.json'), 'frames': frames,
            'frame_scope': 'logical tensors only; not a hardware ABI or allocated DDR plan',
            'software_parity_reexecuted': False, 'hardware_verified': False, 'formal_eligible': False}


def prepare_handoff(request, base_dir=Path('.')):
    """Produce a fail-closed readiness report from a versioned selected-candidate request."""
    arch = validate_architecture(request['architecture'])
    _match(request, {'schema_version': 1, 'space_version': SEARCH_SPACE_VERSION,
                     'architecture_id': architecture_id(arch), 'architecture_hash': architecture_hash(arch)}, 'request')
    require(request['profile'] in PROFILES and request['mapping_policy'] in MAPPINGS, 'unsupported profile/mapping')
    require(request['seed'] in (42, 43, 44) and request['budget'] == 80 and request['num_classes'] == 2,
            'final handoff requires registered seed, full 80 epoch budget and two classes')
    stages, blockers = {}, []
    artifacts = request.get('artifacts', {})
    state = None
    for stage in ('trained_checkpoint', 'software_export', 'confirmation'):
        try:
            if stage == 'trained_checkpoint':
                path = artifact(artifacts.get('checkpoint'), base_dir)
                receipt = read_json(artifact(artifacts.get('training_receipt'), base_dir))
                state = torch.load(path, map_location='cpu', weights_only=True)
                expected = {key: request[key] for key in ('architecture_id', 'seed', 'budget', 'num_classes')}
                _match(state, {**expected, 'architecture': arch, 'test_evaluated': False}, 'checkpoint')
                _match(receipt, {**expected, 'checkpoint_sha256': sha(path), 'test_evaluated': False}, 'training receipt')
                require(state.get('freeze_sha256') and state['freeze_sha256'] == receipt.get('freeze_sha256'), 'training freeze missing/mismatch')
                require(type(state['epoch']) is int and 1 <= state['epoch'] <= 80 and receipt['best_epoch'] == state['epoch'],
                        'selected epoch/receipt mismatch')
                require(receipt['mAP50_95'] == state['valid']['mAP50_95'], 'training validation score mismatch')
                require(request.get('evidence_kind') == 'trained_checkpoint', 'diagnostic or provisional checkpoint not admitted')
                model = StaticElasticNeckDetector(arch, seed=request['seed'], num_classes=request['num_classes'])
                model.load_state_dict(state['model_state_dict'], strict=True)
                source_hash = model_state_hash(model)
                stages[stage] = {'status': 'VERIFIED_BINDING', 'checkpoint_sha256': sha(path), 'model_state_sha256': source_hash}
            elif stage == 'software_export':
                reference = artifacts.get('ptq_directory', {})
                directory = reference.get('path')
                require(directory, 'PTQ directory missing')
                require(sha(Path(base_dir) / directory / 'artifact_manifest.json') == reference.get('sha256'),
                        'selected PTQ artifact manifest SHA256 mismatch')
                result = inspect_ptq_directory(Path(base_dir) / directory)
                _match(result, {key: request[key] for key in ('architecture_id', 'architecture_hash', 'profile', 'num_classes')}, 'PTQ export')
                require(stages.get('trained_checkpoint', {}).get('model_state_sha256') == result['original_model_state_sha256'],
                        'PTQ source state not bound to final checkpoint')
                require(request.get('calibration_sha256') == result['calibration_sha256'], 'selected calibration mismatch')
                profile = read_json(Path(base_dir) / directory / 'profile.json')
                require(state is not None, 'final checkpoint missing')
                _match(profile.get('budget_checkpoint_binding', {}), {
                    'architecture_id': request['architecture_id'], 'architecture_sha256': request['architecture_hash'],
                    'num_classes': request['num_classes'], 'seed': request['seed'], 'budget': request['budget'],
                    'freeze_sha256': state['freeze_sha256'], 'checkpoint_sha256': artifacts['checkpoint']['sha256'],
                    'calibration_sha256': request['calibration_sha256'], 'split': 'valid', 'test_evaluated': False},
                    'final PTQ budget/checkpoint binding')
                stages[stage] = result
            else:
                confirmation = read_json(artifact(artifacts.get('confirmation'), base_dir))
                _match(confirmation, {'status': 'PASS', 'evidence_kind': 'three_seed_quantized_confirmation',
                    'architecture_hash': architecture_hash(arch), 'profile': request['profile'],
                    'selected_checkpoint_sha256': artifacts['checkpoint']['sha256'],
                    'seeds': [42, 43, 44], 'accuracy_split': 'valid', 'test_evaluated': False}, 'confirmation')
                stages[stage] = {'status': 'PENDING_INDEPENDENT_GATE_REPLAY', 'binding_checked': True}
                blockers.append('confirmation: underlying three-seed admission replay required')
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            stages[stage] = {'status': 'BLOCKED', 'reason': str(error)}
            blockers.append(stage + ': ' + str(error))
    for name, reason in HARDWARE_STAGES.items():
        stages[name] = {'status': 'NOT_VERIFIED', 'reason': reason}
        blockers.append(name + ': ' + reason)
    if request.get('hardware_evidence') or request.get('board_parity') == 'PASS':
        blockers.append('UNSUPPORTED_HARDWARE_EVIDENCE: no registered verifier; PASS flags cannot certify deployment')
    return {'schema_version': 1, 'status': 'NOT_READY_FOR_DEPLOYMENT',
            'candidate': {key: request[key] for key in ('space_version', 'architecture_id', 'architecture_hash',
                        'architecture', 'profile', 'mapping_policy', 'num_classes', 'seed', 'budget')},
            'request_sha256': canonical_hash(request), 'stages': stages, 'blockers': blockers,
            'formal_eligible': False, 'launch_hardware': False, 'test_evaluated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--inspect-ptq', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.request) == bool(args.inspect_ptq):
        parser.error('supply exactly one of --request or --inspect-ptq')
    result = (prepare_handoff(read_json(args.request), args.request.resolve().parent)
              if args.request else inspect_ptq_directory(args.inspect_ptq))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
