"""Tiny CPU checkpoint transactions; no training, dataset loading, or GPU use."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.p3_fusion_ablation import run as r


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    r.dump(tmp_path/'protocol.json', {'unit_test': 'CPU recovery only'})
    images = [f'valid/image_{i:03d}.png' for i in range(99)]
    monkeypatch.setattr(r, 'expected_validation_images', lambda: images)
    initial = {'backbone.weight': torch.zeros(2), 'neck.alpha': torch.tensor(1.)}
    shared = r.digest({'backbone.weight': initial['backbone.weight']})
    (tmp_path/'initializations').mkdir()
    r.atomic_save(tmp_path/'initializations'/'seed_42.pt', {
        'seed': 42, 'shared_sha256': shared, 'model_state_dict': initial,
        'torch_rng_after_build': torch.get_rng_state().clone()})
    d = r.folder('weighted', 42)
    d.mkdir(parents=True)
    b = r.baseline()
    cache = {}

    def data(correct):
        if correct not in cache:
            records = []
            for i, image in enumerate(images):
                label = i % 2
                pred = {'boxes': [[8., 8., 16., 16.]], 'scores': [.9], 'labels': [label]} if i < correct else {
                    'boxes': [], 'scores': [], 'labels': []}
                records.append({'image': image, 'prediction': pred,
                                'target': {'boxes': [[8., 8., 16., 16.]], 'labels': [label]}})
            valid = r.frozen.metrics_from_records(records, b)
            valid['validation_loss'] = .1
            cache[correct] = (valid, records)
        return deepcopy(cache[correct])

    def state(epoch, best_epoch, correct=40, best_correct=80, previous=None):
        normal, records = data(correct)
        best_metrics, best_records = data(best_correct)
        history = [{'epoch': i, 'valid': deepcopy(best_metrics if i == best_epoch else normal),
                    'alpha': 1., 'wall_seconds': .01} for i in range(1, epoch+1)]
        # A same-score later epoch must retain the earliest best snapshot.
        if best_epoch < epoch:
            history[best_epoch]['valid'] = deepcopy(best_metrics)
        def commit(h, preds, selected):
            return r.make_epoch_commit(run_binding=r.binding('weighted', 42), history=h,
                parents=[], initial_shared=shared, start_rng='starting_rng',
                model_state_dict={'backbone.weight': torch.full((2,), float(h[-1]['epoch'])),
                                  'neck.alpha': torch.tensor(1.)},
                optimizer_state_dict={'state': {0: {'exp_avg': torch.tensor([.25])}}, 'param_groups': []},
                rng={'torch': torch.get_rng_state().clone()}, records=preds, best_snapshot=selected,
                model_info={'total_params': 3, 'estimated_macs': 10})
        if previous is None:
            previous = commit(history[:best_epoch], best_records, None)['best_snapshot']
        return commit(history, records if epoch != best_epoch else best_records, previous)

    return d, state, data


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def test_epoch80_summary_repairs_stale79_log_and_wrong_exports(campaign):
    d, state, data = campaign
    committed = state(80, 37)
    r.atomic_save(d/'latest.pt', committed)
    latest_hash = r.sha(d/'latest.pt')
    r.dump(d/'epoch_log.json', committed['history'][:79])
    r.dump(d/'best_valid_predictions.json', data(2)[1])
    r.dump(d/'endpoint_valid_predictions.json', data(3)[1])
    r.dump(d/'best_valid_metrics.json', data(2)[0])
    r.dump(d/'model_info.json', {'total_params': -1, 'estimated_macs': -1})
    r.atomic_save(d/'best.pt', {'epoch': 79, 'binding': {'wrong': True}})
    rng_before = torch.get_rng_state().clone()

    result = r.summarize_one('weighted', 42)

    assert r.sha(d/'latest.pt') == latest_hash
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert result['best_epoch'] == 37
    assert result['best_valid']['mAP50_95'] == data(80)[0]['mAP50_95']
    assert result['endpoint_valid']['mAP50_95'] == data(40)[0]['mAP50_95']
    assert len(r.read(d/'epoch_log.json')) == 80
    assert r.read(d/'best_valid_predictions.json') == data(80)[1]
    assert r.read(d/'endpoint_valid_predictions.json') == data(40)[1]
    best = load(d/'best.pt')
    assert r.digest(best) == r.digest(committed['best_snapshot'])
    assert best['purpose'] == 'inference_only_not_resume'
    assert not {'optimizer_state_dict', 'rng', 'history'} & best.keys()
    assert torch.equal(best['model_state_dict']['backbone.weight'], torch.full((2,), 37.))
    assert r.read(d/'status.json')['status'] == 'COMPLETE'
    assert r.read(d/'status.json')['result_sha256'] == r.sha(d/'result.json')
    with pytest.raises(ValueError, match='latest training commit'):
        r.validate_committed(best)


def test_uncommitted_future_best_is_discarded_from_latest12(campaign):
    d, state, data = campaign
    committed = state(12, 8)
    future = state(13, 13, best_correct=98)
    r.atomic_save(d/'latest.pt', committed)
    latest_hash = r.sha(d/'latest.pt')
    r.atomic_save(d/'best.pt', future['best_snapshot'])
    r.dump(d/'best_valid_predictions.json', data(98)[1])
    r.dump(d/'best_valid_metrics.json', data(98)[0])
    r.dump(d/'epoch_log.json', future['history'])
    r.dump(d/'endpoint_valid_predictions.json', data(98)[1])
    r.dump(d/'result.json', {'uncommitted': True})
    r.dump(d/'status.json', {'status': 'COMPLETE', 'epochs': 80})

    selected = r.materialize_committed(load(d/'latest.pt'), d)

    assert r.sha(d/'latest.pt') == latest_hash
    assert selected['epoch'] == 8
    assert load(d/'best.pt')['epoch'] == 8
    assert r.read(d/'best_valid_predictions.json') == data(80)[1]
    assert len(r.read(d/'epoch_log.json')) == 12
    assert r.read(d/'progress.json')['epoch'] == 12
    assert r.read(d/'status.json')['status'] == 'TRAINING'
    assert not (d/'endpoint_valid_predictions.json').exists()
    assert not (d/'result.json').exists()
    next_commit = state(13, 8, previous=selected)
    assert next_commit['best_snapshot']['epoch'] == 8


def test_interruption_after_latest_commit_repairs_partial_materialization(campaign, monkeypatch):
    d, state, _ = campaign
    old = state(12, 8)
    r.atomic_save(d/'latest.pt', old)
    r.materialize_committed(old, d)
    new = state(13, 13, best_correct=98)
    r.atomic_save(d/'latest.pt', new)
    original = r._export_json
    def interrupted(path, value):
        if path.name == 'best_valid_metrics.json':raise OSError('simulated interruption after commit')
        return original(path, value)
    monkeypatch.setattr(r, '_export_json', interrupted)
    with pytest.raises(OSError, match='simulated interruption'):
        r.materialize_committed(new, d)
    assert len(r.read(d/'epoch_log.json')) == 12
    monkeypatch.setattr(r, '_export_json', original)
    r.materialize_committed(load(d/'latest.pt'), d)
    assert len(r.read(d/'epoch_log.json')) == 13
    assert load(d/'best.pt')['epoch'] == 13
    assert r.read(d/'best_valid_metrics.json') == new['best_snapshot']['valid']


@pytest.mark.parametrize('damage', ['history', 'predictions', 'best_epoch', 'alpha', 'binding', 'initialization'])
def test_malformed_commit_rejected_before_any_sidecar_repair(campaign, damage):
    d, state, _ = campaign
    bad = state(12, 8)
    if damage == 'history':bad['history'][-1]['epoch'] = 13
    elif damage == 'predictions':bad['validation_predictions'].pop()
    elif damage == 'best_epoch':bad['best_snapshot']['epoch'] = 9
    elif damage == 'alpha':bad['model_state_dict']['neck.alpha'].fill_(2.)
    elif damage == 'binding':bad['best_snapshot']['binding']['seed'] = 43
    elif damage == 'initialization':bad['best_snapshot']['initial_shared_sha256'] = 'wrong'
    r.atomic_save(d/'latest.pt', bad)
    with pytest.raises(ValueError):r.materialize_committed(bad, d)
    assert not (d/'best.pt').exists()
    assert not (d/'epoch_log.json').exists()


def test_commit_copies_live_tensors_and_keeps_rng_unchanged(campaign):
    _, _, data = campaign
    valid, records = data(40)
    weights = {'backbone.weight': torch.tensor([2., 3.]), 'neck.alpha': torch.tensor(1.)}
    avg = torch.tensor([.25])
    rng_before = torch.get_rng_state().clone()
    s = r.make_epoch_commit(run_binding=r.binding('weighted', 42),
        history=[{'epoch': 1, 'valid': valid, 'alpha': 1., 'wall_seconds': .01}],
        parents=[], initial_shared='initial', start_rng='starting_rng', model_state_dict=weights,
        optimizer_state_dict={'state': {0: {'exp_avg': avg}}, 'param_groups': []},
        rng={'torch': rng_before}, records=records, best_snapshot=None,
        model_info={'total_params': 3, 'estimated_macs': 10})
    weights['backbone.weight'].fill_(99.)
    avg.fill_(99.)
    records.clear()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(s['model_state_dict']['backbone.weight'], torch.tensor([2., 3.]))
    assert float(s['optimizer_state_dict']['state'][0]['exp_avg']) == .25
    assert len(s['validation_predictions']) == 99
    r.validate_committed(s)


@pytest.mark.parametrize('variant', ['equal_add', 'weighted'])
def test_real_model_profile_tuple_metadata_round_trips(campaign, variant):
    """Exercise real profile metadata, whose tuples caused the first server failure."""
    d, state, _ = campaign
    old_threads=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model=r.build_p3_fusion_candidate(variant).eval()
        with torch.random.fork_rng():
            info=r.frozen.get_model_info(model,(1,1,640,640))
    finally:
        torch.set_num_threads(old_threads)
    raw=state(80,37)
    committed=r.make_epoch_commit(run_binding=raw['binding'],history=raw['history'],
        parents=raw['parents'],initial_shared=raw['initial_shared_sha256'],
        start_rng=raw['train_start_rng_sha256'],model_state_dict=raw['model_state_dict'],
        optimizer_state_dict=raw['optimizer_state_dict'],rng=raw['rng'],
        records=raw['validation_predictions'],best_snapshot=raw['best_snapshot'],model_info=info)
    r.atomic_save(d/'latest.pt',committed)
    r.materialize_committed(committed,d)
    assert r.read(d/'model_info.json')==committed['model_info']
    assert committed['model_info']['total_params']==info['total_params']
    assert committed['model_info']['estimated_macs']==info['estimated_macs']
    result=r.summarize_one('weighted',42)
    assert result['params']==info['total_params']
