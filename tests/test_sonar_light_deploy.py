"""Adversarial handoff tests using preserved real software exports as fixtures.

The fixtures originate from diagnostic weights; these tests do not confer
trained-model or board evidence on them.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch

from tools.sonar_light_deploy.handoff import (
    ROOT, PROFILES, architecture_id, architecture_hash, canonical_hash,
    inspect_ptq_directory, prepare_handoff, sha,
)

FIXTURES = ROOT / 'results/modnas_vs_ours_20261007/ptq_real_dataset_a/mixed_head_v1'


def request():
    arch = {'pan': False, 'td_kernels': [3, 3, 3], 'pan_kernels': []}
    return dict(schema_version=1, space_version='elastic_neck72_v1', architecture=arch,
                architecture_id=architecture_id(arch), architecture_hash=architecture_hash(arch),
                profile='HEAD_W8A16', mapping_policy='reference_low', num_classes=2,
                seed=42, budget=80, evidence_kind='synthetic_diagnostic', artifacts={})


def fixture(candidate, profile):
    path = FIXTURES / candidate / profile
    if not path.is_dir():
        pytest.skip('preserved diagnostic software exports not available')
    return path


def rewrite_manifest(folder):
    path = folder / 'artifact_manifest.json'
    data = json.loads(path.read_text())
    data['files'] = {name: sha(folder / name) for name in data['files']}
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.mark.parametrize('candidate', ['E0', 'E1'])
@pytest.mark.parametrize('profile', PROFILES)
def test_real_saved_exports_are_software_only(candidate, profile):
    result = inspect_ptq_directory(fixture(candidate, profile))
    assert result['status'] == 'VERIFIED_SAVED_SOFTWARE_EXPORT'
    assert result['hardware_verified'] is False and result['formal_eligible'] is False
    heads = [row for row in result['frames'] if row['node'].startswith('head.') and not row['node'].endswith('input_requant')]
    assert len(heads) == 4
    assert {row['bits'] for row in heads} == ({8} if profile == 'W8A8' else {16})


def test_missing_checkpoint_never_uses_diagnostic_export():
    data = request()
    folder = fixture('E0', 'HEAD_W8A16')
    data['artifacts']['ptq_directory'] = {'path': str(folder), 'sha256': sha(folder / 'artifact_manifest.json')}
    report = prepare_handoff(data)
    assert report['stages']['software_export']['status'] == 'BLOCKED'
    assert 'not bound to final checkpoint' in report['stages']['software_export']['reason']
    assert report['formal_eligible'] is False


def test_selected_export_manifest_identity_required():
    data = request()
    data['artifacts']['ptq_directory'] = {'path': str(fixture('E0', 'HEAD_W8A16')), 'sha256': '0' * 64}
    report = prepare_handoff(data)
    assert 'manifest SHA256 mismatch' in report['stages']['software_export']['reason']


def test_forged_board_pass_cannot_promote_stage():
    data = request()
    data['board_parity'] = 'PASS'
    data['hardware_evidence'] = {'board_parity': {'status': 'PASS', 'mismatches': 0}}
    result = prepare_handoff(data)
    assert result['status'] == 'NOT_READY_FOR_DEPLOYMENT'
    assert result['stages']['board_parity']['status'] == 'NOT_VERIFIED'
    assert any('UNSUPPORTED_HARDWARE_EVIDENCE' in row for row in result['blockers'])
    assert result['launch_hardware'] is False


def test_architecture_identity_mismatch_rejected():
    data = request()
    data['architecture_hash'] = '0' * 64
    with pytest.raises(ValueError, match='architecture_hash'):
        prepare_handoff(data)


def test_weight_dtype_rejected_even_after_manifest_rehash(tmp_path):
    folder = tmp_path / 'copied'
    shutil.copytree(fixture('E0', 'HEAD_W16A16'), folder)
    path = folder / 'integer_parameters.pt'
    package = torch.load(path, weights_only=True)
    head = next(name for name in package['weights'] if name.startswith('head.'))
    package['weights'][head] = package['weights'][head].to(torch.int8)
    torch.save(package, path)
    rewrite_manifest(folder)
    with pytest.raises(ValueError, match='weight dtype mismatch'):
        inspect_ptq_directory(folder)


def test_missing_parity_node_rejected_even_after_manifest_rehash(tmp_path):
    folder = tmp_path / 'copied'
    shutil.copytree(fixture('E0', 'W8A8'), folder)
    path = folder / 'software_parity_witness.json'
    data = json.loads(path.read_text())
    data['rows'].pop()
    path.write_text(json.dumps(data))
    rewrite_manifest(folder)
    with pytest.raises(ValueError, match='witness missing nodes'):
        inspect_ptq_directory(folder)


def test_mixed_boundary_cannot_be_relabeled_int8(tmp_path):
    folder = tmp_path / 'copied'
    shutil.copytree(fixture('E0', 'HEAD_W8A16'), folder)
    graph = json.loads((folder / 'numeric_graph.json').read_text())
    next(node for node in graph['nodes'] if node['op'] == 'requantize')['output_bits'] = 8
    graph.pop('numeric_graph_sha256')
    graph['numeric_graph_sha256'] = canonical_hash(graph)
    (folder / 'numeric_graph.json').write_text(json.dumps(graph))
    for name in ('profile.json', 'software_parity_witness.json'):
        data = json.loads((folder / name).read_text())
        data['numeric_graph_sha256'] = graph['numeric_graph_sha256']
        (folder / name).write_text(json.dumps(data))
    path = folder / 'integer_parameters.pt'
    package = torch.load(path, weights_only=True)
    package['graph'] = graph
    torch.save(package, path)
    rewrite_manifest(folder)
    with pytest.raises(ValueError, match='output bitwidth mismatch'):
        inspect_ptq_directory(folder)
