"""Meaningful C01b guards and archive replays using synthetic, nonclaimable data."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_head_c01b import train as runner

IMAGES = [f'image_{index:03d}.png' for index in range(99)]


def metrics(value=.2):
    ap = {f'{q/100:.2f}': value for q in range(50, 96, 5)}
    raw = dict(mAP50_95=value, AP50=value, AP75=value, Precision=value, Recall=value,
               num_images=99, coco_details=dict(per_class_ap=dict(all={
                   label: deepcopy(ap) for label in runner.p.CLASS_NAMES})))
    return runner.t.enrich_metrics(raw)


def records():
    return [dict(image='valid/images/' + image, prediction=dict(boxes=[], labels=[], scores=[]),
                 target=dict(boxes=[[1., 2., 3., 4.]], labels=[int(i >= 70)]))
            for i, image in enumerate(IMAGES)]


def history(length=80):
    return [dict(epoch=epoch, valid=metrics(), batches=83, images=330,
                 augmented_batches_sha256=f'pixels-{epoch}', image_order_sha256=f'order-{epoch}',
                 valid_batches_sha256='valid', batch_sha256=[f'bytes-{epoch}-{index}' for index in range(83)],
                 losses=dict(loss=1., box_loss=.3, class_loss=.7, dfl_loss=0.))
            for epoch in range(1, length + 1)]


def head_info(channels=6):
    return dict(quality_package=dict(box='GIoU', distribution='disabled'), head_contract=dict(
        raw_output_channels_per_scale=[channels] * 4, reg_max=1 if channels == 6 else 16,
        scales=['P2', 'P3', 'P4', 'P5'], regression_hidden_channels=[64] * 4,
        classification_hidden_channels=[48] * 4, measurement='synthetic fixture; not measured run'))


def binding():
    return dict(backbone='E0', head='quality_aware', seed=42, mode='formal',
        execution_manifest_sha256='source', environment_fingerprint='environment',
        subset_contract_sha256='subset', test_evaluated=False,
        validation_target_sha256=runner.p.canonical_sha([
            dict(image=row['image'].split('/')[-1], target=row['target']) for row in records()]))


def test_registered_three_seed_protocol_and_complete_source_closure():
    cfg, _ = runner.load_config()
    assert cfg['seeds'] == [42, 43, 44]
    assert cfg['training']['epochs'] == 80
    assert cfg['C01_raw_output_channels_per_scale'] == [66] * 4
    assert cfg['C01b']['raw_output_channels_per_scale'] == [6] * 4
    assert not cfg['decision']['automatic_search']
    paths = runner.source_paths(cfg)
    for relative in ('tools/sonar_head_c01b/model.py', 'tools/sonar_head_c01b/train.py',
                     'tools/sonar_head_c01b/analysis.py', 'tests/test_sonar_head_c01b_train.py',
                     'results/sonar_head_c01b_20261013_01/PREREGISTRATION.md'):
        assert ROOT / relative in paths
    assert all(path.is_file() for path in paths)
    assert not any(path.name in ('baseline_resolution.json', 'independent_review.json') for path in paths)


def test_relative_cli_config_has_the_same_real_source_closure_and_hashes(monkeypatch):
    monkeypatch.chdir(ROOT)
    relative = Path('configs/sonar_head_c01b_20261013.yaml')
    cfg, _ = runner.load_config(relative)
    relative_paths = runner.source_paths(cfg, config=relative)
    absolute_paths = runner.source_paths(cfg, config=relative.resolve())
    assert relative_paths == absolute_paths
    assert all(path.is_absolute() for path in relative_paths)
    relative_hashes = runner.source_hashes(cfg, config=relative)
    assert relative_hashes == runner.source_hashes(cfg, config=relative.resolve())
    assert relative_hashes[relative.as_posix()] == runner.p.sha(relative.resolve())


@pytest.mark.parametrize('field,value', [
    ('seeds', [42, 43, 44, 45]), ('C01_reg_max', 8),
    ('C01_raw_output_channels_per_scale', [34] * 4), ('gpu_index', 0),
    ('training', {'epochs': 40}), ('C01_box_loss', 'GIoU'),
])
def test_drifted_protocol_is_rejected(tmp_path, field, value):
    cfg = yaml.safe_load(runner.CONFIG.read_text(encoding='utf-8'))
    cfg[field] = value
    path = tmp_path / 'changed.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='Unregistered C01b'):
        runner.load_config(path)


def authorization_fixture(root):
    cfg, _ = runner.load_config()
    runner.p.dump(root / 'execution_manifest.json', dict(config=cfg))
    runner.p.dump(root / 'smoke_gate.json', dict(status='PASS'))
    runner.p.dump(root / 'preflight_complete.json', dict(status='PASS',
        execution_manifest_sha256=runner.p.sha(root / 'execution_manifest.json')))
    runner.p.dump(root / 'baseline_resolution.json', dict(status='RESOLVED', choice=runner.CHOICE,
        user_authorization='Synthetic test authorization, not a user instruction',
        C01_execution_manifest_sha256=cfg['C01_execution_manifest_sha256'], C01_reg_max=16,
        C01_raw_output_channels_per_scale=[66] * 4))
    runner.p.dump(root / 'independent_review.json', dict(status='PASS', independent=True,
        scope='C01b_66_to_6_readiness', execution_manifest_sha256=runner.p.sha(root / 'execution_manifest.json'),
        smoke_gate_sha256=runner.p.sha(root / 'smoke_gate.json'),
        preflight_complete_sha256=runner.p.sha(root / 'preflight_complete.json')))


def test_unresolved_control_blocks_formal_before_smoke_or_process(tmp_path, monkeypatch):
    authorization_fixture(tmp_path)
    resolution = runner.read_json(tmp_path / 'baseline_resolution.json')
    resolution.update(status='UNRESOLVED', choice=None, user_authorization=None)
    runner.p.dump(tmp_path / 'baseline_resolution.json', resolution)
    monkeypatch.setattr(runner, 'require_smoke', lambda root: pytest.fail('Formal guard must stop before smoke'))
    with pytest.raises(ValueError, match='user-resolved'):
        runner.verify_formal_authorization(tmp_path, freeze=True)
    assert not (tmp_path / 'formal_authorization.json').exists()


@pytest.mark.parametrize('change', ['nonindependent', 'wrongscope', 'stalesource', 'nopass'])
def test_actual_independent_exact_manifest_review_required(tmp_path, monkeypatch, change):
    authorization_fixture(tmp_path)
    monkeypatch.setattr(runner, 'require_smoke', lambda root: {})
    review = runner.read_json(tmp_path / 'independent_review.json')
    if change == 'nonindependent':
        review['independent'] = False
    elif change == 'wrongscope':
        review['scope'] = 'static_initial_review'
    elif change == 'stalesource':
        review['execution_manifest_sha256'] = 'other'
    else:
        review['status'] = 'PENDING'
    runner.p.dump(tmp_path / 'independent_review.json', review)
    with pytest.raises(ValueError, match='Independent review'):
        runner.verify_formal_authorization(tmp_path, freeze=True)


@pytest.mark.parametrize('target', ['baseline_resolution.json', 'independent_review.json', 'smoke_gate.json',
                                 'preflight_complete.json'])
def test_frozen_authorization_rejects_later_control_digest_edits(tmp_path, monkeypatch, target):
    authorization_fixture(tmp_path)
    monkeypatch.setattr(runner, 'require_smoke', lambda root: {})
    auth = runner.verify_formal_authorization(tmp_path, freeze=True)
    assert runner.verify_formal_authorization(tmp_path) == auth
    # Even semantically equivalent JSON whitespace changes the frozen digest.
    path = tmp_path / target
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='digest'):
        runner.verify_formal_authorization(tmp_path)


@pytest.mark.parametrize('change', ['missing', 'failed', 'stalesource'])
def test_actual_preflight_completion_is_required_by_direct_formal_guard(tmp_path, monkeypatch, change):
    authorization_fixture(tmp_path)
    monkeypatch.setattr(runner, 'require_smoke', lambda root: {})
    path = tmp_path / 'preflight_complete.json'
    if change == 'missing':
        path.unlink()
        with pytest.raises(FileNotFoundError):
            runner.verify_formal_authorization(tmp_path, freeze=True)
    else:
        value = runner.read_json(path)
        if change == 'failed':
            value['status'] = 'FAILED'
        else:
            value['execution_manifest_sha256'] = 'other'
        runner.p.dump(path, value)
        with pytest.raises(ValueError, match='completed preflight'):
            runner.verify_formal_authorization(tmp_path, freeze=True)
    assert not (tmp_path / 'formal_authorization.json').exists()


@pytest.mark.parametrize('field', ['smoke_gate_sha256', 'preflight_complete_sha256'])
@pytest.mark.parametrize('value', [None, 'stale'])
def test_actual_review_must_bind_both_smoke_and_preflight_digests(tmp_path, monkeypatch, field, value):
    authorization_fixture(tmp_path)
    monkeypatch.setattr(runner, 'require_smoke', lambda root: {})
    review = runner.read_json(tmp_path / 'independent_review.json')
    if value is None:
        del review[field]
    else:
        review[field] = value
    runner.p.dump(tmp_path / 'independent_review.json', review)
    with pytest.raises(ValueError, match='digest'):
        runner.verify_formal_authorization(tmp_path, freeze=True)


def test_missing_formal_authorization_is_not_implicitly_created(tmp_path, monkeypatch):
    authorization_fixture(tmp_path)
    monkeypatch.setattr(runner, 'require_smoke', lambda root: {})
    with pytest.raises(ValueError, match='not been frozen'):
        runner.verify_formal_authorization(tmp_path)
    assert not (tmp_path / 'formal_authorization.json').exists()


def smoke_fixture(root, monkeypatch):
    runner.p.dump(root / 'execution_manifest.json', dict(candidate='synthetic'))
    runner.p.dump(root / 'environment_lock.json', dict(fingerprint='environment'))
    monkeypatch.setattr(runner, 'binding', lambda root, seed, mode: dict(seed=seed, mode=mode))
    monkeypatch.setattr(runner, 'reference_row', lambda root, seed: dict(
        history=history(), initialization=dict(seed=seed)))
    hashes = {}
    for seed in runner.SEEDS:
        folder = runner.result_dir(root, seed, 'smoke')
        runner.p.dump(folder / 'model_info.json', head_info())
        runner.p.dump(folder / 'smoke_report.json', dict(status='PASS', binding=dict(seed=seed, mode='smoke'),
            training=dict(batches=2, images=8, batch_sha256=history()[0]['batch_sha256'][:2], losses=dict(dfl_loss=0.)),
            valid=dict(num_images=4), valid_batches_sha256='valid4',
            initialization=dict(c01_reference=dict(seed=seed), shared_head_sha256='shared', c01_shared_head_sha256='shared'),
            model_info_sha256=runner.p.sha(folder / 'model_info.json'), initial_state_sha256='initial', final_state_sha256='changed'))
        hashes[str(seed)] = runner.p.sha(folder / 'smoke_report.json')
    gate = dict(status='PASS', execution_manifest_sha256=runner.p.sha(root / 'execution_manifest.json'),
                environment_fingerprint='environment', seeds=list(runner.SEEDS), smoke_report_sha256=hashes)
    runner.p.dump(root / 'smoke_gate.json', gate)
    return gate


def test_smoke_requires_all_three_seed_artifacts(tmp_path, monkeypatch):
    gate = smoke_fixture(tmp_path, monkeypatch)
    assert runner.require_smoke(tmp_path) == gate
    del gate['smoke_report_sha256']['44']
    runner.p.dump(tmp_path / 'smoke_gate.json', gate)
    with pytest.raises(ValueError, match='three-seed smoke'):
        runner.require_smoke(tmp_path)


@pytest.mark.parametrize('change', ['pixels', 'initialization', 'channels', 'valid4'])
def test_smoke_rejects_individual_seed_pairing_or_channel_drift(tmp_path, monkeypatch, change):
    gate = smoke_fixture(tmp_path, monkeypatch)
    folder = runner.result_dir(tmp_path, 43, 'smoke')
    report = runner.read_json(folder / 'smoke_report.json')
    if change == 'pixels':
        report['training']['batch_sha256'][0] = 'other'
    elif change == 'initialization':
        report['initialization']['c01_reference'] = dict(seed=42)
    elif change == 'valid4':
        report['valid_batches_sha256'] = 'other'
    else:
        info = head_info(66)
        runner.p.dump(folder / 'model_info.json', info)
        report['model_info_sha256'] = runner.p.sha(folder / 'model_info.json')
    runner.p.dump(folder / 'smoke_report.json', report)
    gate['smoke_report_sha256']['43'] = runner.p.sha(folder / 'smoke_report.json')
    runner.p.dump(tmp_path / 'smoke_gate.json', gate)
    with pytest.raises(ValueError):
        runner.require_smoke(tmp_path)


@pytest.mark.parametrize('field', list(runner.WITNESSES) + ['batch_sha256', 'epoch', 'dfl_loss'])
def test_real_time_pairing_rejects_every_data_or_dfl_drift(field):
    reference = history()[0]
    candidate = deepcopy(reference)
    runner.verify_paired_epoch(candidate, reference, 1)
    if field == 'dfl_loss':
        candidate['losses']['dfl_loss'] = .01
    elif field == 'epoch':
        candidate[field] = 2
    else:
        candidate[field] = 'different'
    with pytest.raises(ValueError):
        runner.verify_paired_epoch(candidate, reference, 1)


def test_pairing_requires_all_80_complete_histories():
    reference = dict(history=history())
    runner.verify_pairing_history(history(), reference)
    candidate = history()
    del candidate[29]['valid_batches_sha256']
    with pytest.raises(ValueError, match='data witness'):
        runner.verify_pairing_history(candidate, reference)
    with pytest.raises(ValueError, match='history'):
        runner.verify_pairing_history(history()[:-1] + [history()[78]], reference)


@pytest.mark.parametrize('channels', [6, 66])
def test_actual_head_channel_metadata_is_mandatory(channels):
    info = head_info(channels)
    runner.validate_head_info(info, channels)
    for change in ('missing', '34', 'hidden16'):
        altered = deepcopy(info)
        if change == 'missing':
            del altered['head_contract']
        elif change == '34':
            altered['head_contract']['raw_output_channels_per_scale'] = [34] * 4
        else:
            altered['head_contract']['regression_hidden_channels'] = [16] * 4
        with pytest.raises(ValueError, match='measured raw'):
            runner.validate_head_info(altered, channels)


def archive_fixture(root):
    folder = runner.result_dir(root, 42)
    folder.mkdir(parents=True)
    expected = binding()
    hist = history()
    for row in hist:
        if row['epoch'] in runner.STABLE_EPOCHS:
            row['prediction_archive'] = runner.t.save_stable_predictions(folder, row['epoch'], records(), expected, IMAGES)
    identity = dict(c01_reference=dict(policy='test'), shared_head_sha256='shared', c01_shared_head_sha256='shared')
    runner.p.dump(folder / 'model_info.json', head_info())
    runner.p.dump(folder / 'initialization_identity.json', identity)
    runner.p.dump(folder / 'epoch_log.json', hist)
    runner.p.dump(folder / 'best_valid_records.json', records())
    runner.p.dump(folder / 'terminal_valid_records.json', records())
    runner.p.dump(folder / 'best_valid_metrics.json', metrics())
    runner.p.dump(folder / 'terminal_valid_metrics.json', metrics())
    state = dict(binding=expected, epoch=80, history=hist, initialization=identity,
                 model_info_sha256=runner.p.sha(folder / 'model_info.json'), initial_state_sha256='test-initial')
    torch.save(state, folder / 'latest.pt')
    best = deepcopy(state)
    best.update(epoch=1, history=hist[:1])
    torch.save(best, folder / 'best.pt')
    runner.p.dump(root / 'execution_manifest.json', dict(inputs={}, comparison_contract_sha256='comparison'))
    return folder, hist, expected


def patch_archive_dependencies(monkeypatch, hist, expected):
    monkeypatch.setattr(runner, 'verify_manifest', lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(runner, 'verify_formal_authorization', lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, 'binding', lambda *args, **kwargs: expected)
    monkeypatch.setattr(runner, 'reference_row', lambda *args, **kwargs: dict(history=history(), initialization=dict(policy='test')))
    monkeypatch.setattr(runner.t, 'load_subset_contract', lambda *args, **kwargs: dict(full_images=IMAGES))
    monkeypatch.setattr(runner.p, 'metrics_from_records', lambda *args, **kwargs: {
        key: value for key, value in metrics().items() if key not in ('map', 'per_class_AP')})


def test_complete_run_replays_all_five_receipts_and_rejects_tamper(tmp_path, monkeypatch):
    folder, hist, expected = archive_fixture(tmp_path)
    patch_archive_dependencies(monkeypatch, hist, expected)
    report = runner.run_result(tmp_path, 42)
    assert report['epochs'] == 80 and report['formal_eligible']
    assert set(report['stable_prediction_paths']) == {'76', '77', '78', '79', '80'}
    assert not (folder / 'result.json').exists()
    path = folder / hist[77]['prediction_archive']['receipt']
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash drift'):
        runner.run_result(tmp_path, 42)


def test_full_completion_rejects_missing_stable_predictions(tmp_path, monkeypatch):
    folder, hist, expected = archive_fixture(tmp_path)
    patch_archive_dependencies(monkeypatch, hist, expected)
    path = folder / hist[79]['prediction_archive']['records']
    path.unlink()
    with pytest.raises(ValueError, match='missing file'):
        runner.run_result(tmp_path, 42)
