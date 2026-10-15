"""Regressions for the implementation audit; no training or hardware jobs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch
import torch.nn as nn

from hwnas_fpga.search.rl_searcher import RLSearcher
from hwnas_fpga.search.pareto import resolve_pareto_objectives
from hwnas_fpga.interfaces import SearchConstraints
from hwnas_fpga.hardware.mobilenetA_operator_cost import ZCU104MobileNetAOperatorCostEstimator
from hwnas_fpga.deploy.int8_reference import run_integer_reference

ROOT = Path(__file__).resolve().parents[1]


def reward_searcher():
    obj = RLSearcher.__new__(RLSearcher)
    obj._architecture_visit_key = lambda architecture: architecture
    obj.hardware_rank_weight = 0.05
    row = lambda value: dict.fromkeys(('latency_ms', 'energy_mj', 'dsp', 'bram', 'lut'), value)
    obj.raw_space_hardware_reference = {
        'pareto_rank_by_key': {'best': 0, 'second': 1},
        'metrics_by_key': {'best': row(1), 'second': row(2)},
        'max_pareto_rank': 1,
    }
    return obj, row


def test_unseen_architecture_gets_hardware_penalty_and_order_is_stable():
    obj, row = reward_searcher()
    assert obj._hardware_pareto_rank_penalty('unseen_worse', row(3)) == pytest.approx(.05)
    assert obj._hardware_pareto_rank_penalty('unseen_equal', row(2)) == pytest.approx(.025)
    assert obj._hardware_pareto_rank_penalty('unseen_better', row(0)) == 0
    assert obj._hardware_pareto_rank_penalty('unseen_worse', row(3)) == pytest.approx(.05)
    assert obj._hardware_pareto_rank_penalty('second') == pytest.approx(.025)
    with pytest.raises(ValueError, match='finite'):
        obj._hardware_pareto_rank_penalty('invalid', row(float('nan')))


@pytest.mark.parametrize('functional', [False, True])
@pytest.mark.parametrize('scale', [.1, .25, 1.0])
def test_integer_relu6_clamps_negative_and_upper_boundary(functional, scale):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU6()

        def forward(self, value):
            return torch.nn.functional.relu6(value) if functional else self.relu(value)

    package = {'schema_version': 2, 'input_scale': scale, 'layers': [{'name': 'no_weight_operator'}]}
    value = torch.tensor([-2., 0., 5., 6., 7., 10.])
    observed = run_integer_reference(Model(), value, package, input_kind='synthetic_unit_test')
    torch.testing.assert_close(observed.output_float, value.clamp(0, 6))


def test_explicit_pareto_axes_preserve_physical_flags():
    constraints = SearchConstraints(physical={'enabled': True, 'pareto_physical_risk': True,
                                              'pareto_early_expand_pressure': True})
    axes, directions = resolve_pareto_objectives(
        {'objectives': ['macro_f1', 'latency_ms'], 'directions': ['max', 'min']}, {},
        constraints, selection_metric='macro_f1',
    )
    assert axes == ['macro_f1', 'latency_ms', 'physical_risk', 'early_expand_pressure']
    assert directions == ['max', 'min', 'min', 'min']
    with pytest.raises(ValueError, match='physical objective'):
        resolve_pareto_objectives({'objectives': ['macro_f1', 'physical_risk'],
                                  'directions': ['max', 'max']}, {}, constraints,
                                 selection_metric='macro_f1')


def test_cost_uses_operating_clock_and_converts_bram_tiles():
    obj = ZCU104MobileNetAOperatorCostEstimator.__new__(ZCU104MobileNetAOperatorCostEstimator)
    obj.hardware_spec = SimpleNamespace(clock_mhz=200)
    obj.last_selected_row_keys = []
    obj._physical_constraints = lambda: {'board_max_bram18': 624, 'board_max_lutram': 101760}
    assert obj._operating_latency_ms({'latency_cycles_hls': 4899, 'latency_ms_hls': .01670559}) == pytest.approx(.024495)
    pressure = obj._zcu104_route_pressure_metrics(
        architecture=SimpleNamespace(head_conv_channels=1280), costs=[], total_lut=0, total_bram=400,
    )
    assert pressure['zcu104_route_bram18_pressure'] == pytest.approx(800/624)
    assert pressure['zcu104_route_overflow_pressure'] > 0


@pytest.fixture
def freezer():
    path = ROOT/'scripts/freeze_zcu104_mobilenetA_rl_pareto_candidates.py'
    spec = importlib.util.spec_from_file_location('freeze_regression', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_role_compatibility_helper_prefers_latency_before_energy(freezer):
    rows = [{'arch_id': name, 'metrics': dict(macro_f1=.8, top1=.8, latency_ms=latency,
             energy_mj=energy, dsp=1, bram=1, lut=1)}
            for name,latency,energy in [('A',1,10),('B',2,1)]]
    assert freezer.pick_roles(rows)['latency_energy_first'] == 'A'


def test_freezer_rejects_unrelated_config_before_freezing(tmp_path, freezer):
    results = tmp_path/'results'
    results.mkdir()
    (tmp_path/'config.yaml').write_text('search: {method: rl}\n')
    other = tmp_path/'other.yaml'
    other.write_text('search: {method: random}\n')
    with pytest.raises(ValueError, match='saved search run config'):
        freezer.validate_run_binding(results/'pareto_selection.json', other, tmp_path/'empty_lut.json')


def test_freezer_keeps_recorded_representatives_outside_topk(tmp_path, freezer):
    row = {'arch_id': 'A', 'encoding': {'id': 1}, 'metrics': {'macro_f1': .8}, 'pareto_front': True}
    pareto = {'selected_topk': [], 'ranked_candidates': [row], 'objectives': ['macro_f1'], 'directions': ['max']}
    reps = {'status': 'complete', 'objectives': ['macro_f1'], 'directions': ['max'],
            'roles': {'accuracy_first': row}}
    (tmp_path/'pareto_representatives.json').write_text(json.dumps(reps))
    selected, roles = freezer.recorded_selection(tmp_path, pareto)
    assert selected == [row]
    assert roles == {'accuracy_first': 'A'}
    reps['roles']['accuracy_first'] = {**row, 'encoding': {'id': 2}}
    (tmp_path/'pareto_representatives.json').write_text(json.dumps(reps))
    with pytest.raises(ValueError, match='identity'):
        freezer.recorded_selection(tmp_path, pareto)


def test_proxy_recipe_is_explicit_and_rejects_silently_ignored_knobs():
    from hwnas_fpga.training.proxy_config import resolve_proxy_training
    resolved = resolve_proxy_training({'lr': .02, 'batch_size': 4})
    assert resolved['lr'] == .02
    assert resolved['gradient_accumulation_steps'] == 1
    assert resolved['amp'] is False
    with pytest.raises(ValueError, match='does not support'):
        resolve_proxy_training({'amp': True})
    with pytest.raises(ValueError, match='unsupported'):
        resolve_proxy_training({'typo': 2})


def test_gap1280_measurement_config_is_isolated_and_renders():
    import sys
    sys.path.insert(0, str(ROOT/'hls_lut_builder/scripts'))
    from common import build_cases, load_config, load_template, render_template
    cases = build_cases(load_config(ROOT/'hls_lut_builder/configs/candidate_kernels_zcu104_gap1280_v3.yaml'))
    assert len(cases) == 1
    case = cases[0]
    rendered = render_template(load_template(case.template_path), {'top_function': case.top_function, **case.parameters})
    assert 'IN_CH = 1280' in rendered and 'OUT_CH = 1280' in rendered
    assert '__' not in rendered
    assert 'mobilenetA_gap1280_v3' in str(case.project_dir)
    assert case.target_clock_mhz == 200


@pytest.fixture
def frozen_run(tmp_path, freezer):
    """Synthetic provenance fixture; these costs are not hardware evidence."""
    import yaml
    from hwnas_fpga.hardware.mobilenetA_operator_cost import canonical_sha256
    from hwnas_fpga.runtime import build_search_space, build_constraints
    cfg = yaml.safe_load((ROOT/'configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml').read_text(encoding='utf-8'))
    space = build_search_space(cfg, image_size=224, input_channels=1, num_classes=8,
                               constraints=build_constraints(cfg))
    encoding = space.baseline_architecture().to_dict()
    results = tmp_path/'results'
    (results/'candidates').mkdir(parents=True)
    lut_path = tmp_path/'synthetic_lut.json'
    lut = dict(search_ready=True, search_space_version=cfg['search_space']['search_space_version'],
               hardware_lut_version='synthetic_test_only', rows=[{'synthetic': True}])
    lut['payload_sha256'] = canonical_sha256(lut)
    freezer.write_json(lut_path, lut)
    cfg['hardware']['operator_cost_lut_path'] = str(lut_path)
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    freezer.write_json(results/'lut_stats.json', dict(operator_cost_file_sha256=freezer.sha256_file(lut_path),
        payload_sha256=lut['payload_sha256'], hardware_lut_version=lut['hardware_lut_version']))
    row = dict(arch_id='synthetic_A', encoding=encoding, metrics={'macro_f1': .8}, pareto_front=True)
    freezer.write_json(results/'candidates/synthetic_A.json', {'candidate': row})
    pareto = dict(selected_topk=[row], ranked_candidates=[row], objectives=['macro_f1'], directions=['max'])
    freezer.write_json(results/'pareto_selection.json', pareto)
    freezer.write_json(results/'pareto_representatives.json', dict(status='complete', objectives=['macro_f1'],
        directions=['max'], roles={'accuracy_first': row}))
    freezer.write_json(results/'rl_search_metrics.json', {'synthetic': True})
    (results/'rl_search_curve.csv').write_text('episode,macro_f1\n0,0.8\n')
    return tmp_path, lut_path


def test_freeze_roundtrip_binds_published_bytes_and_preserves_original_roles(frozen_run, freezer, monkeypatch):
    import sys
    run, lut = frozen_run
    monkeypatch.setattr(sys, 'argv', ['freeze', '--run-dir', str(run),
        '--classification-setup-summary', str(run/'absent.json')])
    freezer.main()
    manifest = freezer.read_json(run/'frozen_candidates_v2/rl_frozen_mobilenetA_pareto_candidates.json')
    entry = manifest['frozen_candidates'][0]
    path = Path(entry['candidate_json']['path'])
    assert entry['candidate_json']['sha256'] == freezer.sha256_file(path)
    assert manifest['roles']['accuracy_first']['arch_id'] == 'synthetic_A'
    record = freezer.read_json(path)
    assert record['lut_binding']['sha256'] == freezer.sha256_file(lut)
    assert record['lut_binding']['cost_semantics_version'] == 'historical_unversioned'
    assert record['source_search_artifacts']['pareto_representatives']['sha256'] == freezer.sha256_file(run/'results/pareto_representatives.json')


def test_freezer_rejects_empty_and_substituted_lut(frozen_run, freezer):
    from hwnas_fpga.hardware.mobilenetA_operator_cost import canonical_sha256
    run, lut_path = frozen_run
    lut = freezer.read_json(lut_path)
    for rows, message in [([], 'ready operator evidence'), ([{'other': True}], 'consumed by this run')]:
        altered = {**lut, 'rows': rows}
        altered.pop('payload_sha256')
        altered['payload_sha256'] = canonical_sha256(altered)
        other = run/'other_lut.json'
        freezer.write_json(other, altered)
        with pytest.raises(ValueError, match=message):
            freezer.validate_run_binding(run/'results/pareto_selection.json', run/'config.yaml', other)


def test_freezer_validation_failure_publishes_no_candidates(frozen_run, freezer, monkeypatch):
    import sys
    run, _lut = frozen_run
    (run/'results/rl_search_curve.csv').unlink()
    monkeypatch.setattr(sys, 'argv', ['freeze', '--run-dir', str(run)])
    with pytest.raises(FileNotFoundError):
        freezer.main()
    assert not (run/'frozen_candidates_v2').exists()


def test_resume_reads_nested_protocol_and_rejects_old_state(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT))
    from run_search import _restore_rl_resume_state
    tracker = SimpleNamespace(results_dir=tmp_path/'results', checkpoints_dir=tmp_path/'checkpoints')
    tracker.results_dir.mkdir()
    tracker.checkpoints_dir.mkdir()
    (tracker.results_dir/'candidates.jsonl').write_text('')
    torch.save({'controller_state_dict': {}, 'optimizer_state_dict': {}, 'episode': 3},
               tracker.checkpoints_dir/'controller_latest.pt')
    protocol = {'schema_version': 2, 'reference_sha256': 'synthetic'}
    obj = SimpleNamespace(reward_protocol=protocol, evaluated_candidates=[], feasible_candidates=[],
        infeasible_candidates=[], baseline=0, controller=SimpleNamespace(load_state_dict=lambda state: None),
        controller_optimizer=SimpleNamespace(load_state_dict=lambda state: None))
    state_path = tracker.checkpoints_dir/'search_state.json'
    state_path.write_text(json.dumps({'extra': {'reward_protocol': protocol}}))
    assert _restore_rl_resume_state(obj, tracker, 'cpu') == 4
    for state in [{'extra': {}}, {'extra': {'reward_protocol': {**protocol, 'reference_sha256': 'changed'}}}]:
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match='Resume reward protocol'):
            _restore_rl_resume_state(obj, tracker, 'cpu')


def test_full_matrix_generator_uses_gap1280_and_new_workspace():
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from build_zcu104_mobilenetA_full_operator_cost_matrix import build_zcu104_config, BASE_CONFIG, MEASUREMENT_VERSION
    cfg = build_zcu104_config(BASE_CONFIG)
    assert cfg['operators']['global_avg_pool']['shape_refs'] == ['gap_7_1280']
    assert cfg['representative_shapes']['gap_7_1280']['in_channels'] == 1280
    assert all(MEASUREMENT_VERSION in value for value in cfg['workspace'].values())


def test_provider_and_preflight_agree_on_formal_lut_version():
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from audit_zcu104_mobilenetA_formal_lut_preflight import SUPPORTED_LUT_VERSIONS as preflight_versions
    from hwnas_fpga.hardware.mobilenetA_operator_cost import SUPPORTED_LUT_VERSIONS as provider_versions
    assert provider_versions == preflight_versions == {'mobilenetA_full_dmdc_cfem_v3_gap1280'}


def test_resume_rejects_changed_config_without_overwriting_evidence(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT))
    from run_search import _validate_rl_resume_config
    run = tmp_path/'original_run'
    run.mkdir()
    saved = run/'config.yaml'
    original = b'search: {episodes: 20}\n'
    saved.write_bytes(original)
    _validate_rl_resume_config(tmp_path, 'original_run', {'search': {'episodes': 20}})
    with pytest.raises(ValueError, match='differs from saved run'):
        _validate_rl_resume_config(tmp_path, 'original_run', {'search': {'episodes': 200}})
    assert saved.read_bytes() == original


def test_freeze_compares_shared_objectives_when_ranked_rows_have_extra_metrics(tmp_path, freezer):
    row = {'arch_id': 'A', 'encoding': {'id': 1}, 'metrics': {'macro_f1': .8}}
    ranked = {**row, 'metrics': {**row['metrics'], 'diagnostic_only': 3}, 'pareto_front': True}
    pareto = {'objectives': ['macro_f1'], 'directions': ['max'],
              'selected_topk': [row], 'ranked_candidates': [ranked]}
    freezer.write_json(tmp_path/'pareto_representatives.json', {
        'status': 'complete', 'objectives': ['macro_f1'], 'directions': ['max'],
        'roles': {'accuracy_first': ranked}})
    selected, _ = freezer.recorded_selection(tmp_path, pareto)
    assert selected == [row]
    row['metrics']['macro_f1'] = .9
    with pytest.raises(ValueError, match='objective metrics differ'):
        freezer.recorded_selection(tmp_path, pareto)
