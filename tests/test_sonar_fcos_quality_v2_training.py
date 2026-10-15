"""CPU-only bounded replay witnesses; no research data, downloads, or GPU work."""
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_fcos_pilot import numerics
from tools.sonar_fcos_pilot.common import read_json, sha256_file, write_json
from tools.sonar_fcos_quality_v2 import train


class TinyDataset(Dataset):
    def __init__(self, path, split, image_size):
        self.split = split
        self.records = [{"image_id": index + 1, "boxes": [[0., 0., 1., 1.]], "labels": [1],
                         "width": 2, "height": 2} for index in range(3 if split == "train" else 2)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # All three RNGs affect updates, making the resume parity test sensitive
        # to missing Python, NumPy, or Torch checkpoint state restoration.
        image = torch.tensor([index + random.random() + float(np.random.random())]) + torch.rand(1)
        target = {"boxes": torch.tensor([[0., 0., 1., 1.]]),
                  "labels": torch.tensor([1]), "image_id": torch.tensor([index + 1])}
        return image, target


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.register_buffer("training_calls", torch.zeros((), dtype=torch.long))

    def forward(self, images, targets):
        self.training_calls.add_(1)
        loss = sum(((self.weight * image + torch.rand(1) * .1) - .5).square().sum() for image in images)
        return {"tiny_loss": loss}


def cpu_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(), "cuda": []}


def restore_cpu_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    assert state["cuda"] == []


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


@pytest.fixture
def tiny_campaign(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    source = {"fixture": "tiny_cpu_only", "pretrained_checkpoint_sha256": "fixture-hash"}
    environment = {"device": "cpu", "fixture": True}
    input_checks = []
    manifest_path = tmp_path / "manifest.json"
    manifest = {"classes": ["object"], "samples": []}
    write_json(manifest_path, manifest)
    monkeypatch.setattr(train, "require_manifest", lambda config: (manifest_path, manifest))
    monkeypatch.setattr(train, "source_binding", lambda config: source.copy())
    monkeypatch.setattr(train, "environment", lambda: environment.copy())
    monkeypatch.setattr(train, "verify_development_inputs", lambda value: input_checks.append(value))
    monkeypatch.setattr(train, "SonarDataset", TinyDataset)
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(train, "model_contract", lambda model: {"tiny_cpu_fixture": True,
                        "pretrained": {"checkpoint_sha256": "fixture-hash"}})
    monkeypatch.setattr(train, "seed_everything", seed_cpu)
    for module in (train, numerics):
        monkeypatch.setattr(module, "rng_state", cpu_rng_state)
        monkeypatch.setattr(module, "restore_rng", restore_cpu_rng)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("The bounded CPU witness must not call a CUDA runtime API")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden_cuda)

    def fake_predict(model, dataset, config, **kwargs):
        model.eval()
        predictions = [{"image_id": record["image_id"], "weight": float(model.weight.detach()),
                        "training_calls": int(model.training_calls)} for record in dataset.records]
        diagnostics = None
        if kwargs.get('trace'):
            rows = [{"image_id": record['image_id'], "gt_index": 0, "label": 1,
                     "mechanism_first_zero": "covered_final", "stages": {
                         stage: {"count": 1, "best_iou": 1., "good75_count": 1,
                                 "good85_count": 1, "good75_max_score": .8, "good85_max_score": .8}
                         for stage in train.STAGES}} for record in dataset.records]
            diagnostics = {"ground_truths": len(rows), "stage_rows": rows,
                           "nms_first_loss": 0, "raw_no_good_box": 0,
                           "stage_covered_gt": {stage: len(rows) for stage in train.STAGES}}
        return predictions, diagnostics

    def fake_evaluate(records, predictions, classes):
        value = 1.0 / (1.0 + abs(predictions[0]["weight"]))
        return {"map": value, "ap75": value * .8, "ap85": value * .6, "wrong_class_fp": 0}

    monkeypatch.setattr(train, "predict", fake_predict)
    monkeypatch.setattr(train, "evaluate_quality_predictions", fake_evaluate)

    def fake_checkpoint_evaluation(records, predictions, classes, rows, **kwargs):
        def workpoint(threshold):
            return {'threshold': threshold, 'counts': {'fppi': 0., 'wrong_class_fp': 0},
                    'diagnostics': {'status': 'AVAILABLE', 'nms_first_loss': 0,
                                    'raw_no_good_box': 0, 'final_no_good_box': 0}}

        return {'selection': kwargs['selection'], 'epoch': kwargs['epoch'],
                'metrics': train.evaluate_quality_predictions(records, predictions, classes),
                'diagnostics': {'ground_truths': len(rows), 'nms_first_loss': 0,
                                'raw_no_good_box': 0, 'final_no_good_box': 0, 'wrong_class_fp': 0},
                'fixed_score_025': workpoint(.25), 'workpoint': workpoint(.001)}

    monkeypatch.setattr(train, 'evaluate_checkpoint', fake_checkpoint_evaluation)

    def create(name):
        output = tmp_path / name
        config = {"variants": ["AAA", "QFL_DFL"], "seeds": [42, 43, 44],
            "output_dir": str(output), "checkpoint_root": str(output / "checkpoints"),
            "torch_threads": 1, "device": "cpu", "image_size": 640,
            "pretrained": False, "weight_cache": str(tmp_path / "unused"),
            "learning_rate": .01, "momentum": .9, "weight_decay": .0001,
            "amp": False, "amp_initial_scale": 128., "gradient_accumulation": 2,
            "batch_size": 1, "workers": 0, "learning_rate_drop_epochs": [20, 25],
            "warmup_epochs": 1, "warmup_start_factor": .1}
        execution_path = output / "execution_manifest.json"
        write_json(execution_path, {"source_binding": source, "environment": environment})
        write_json(output / "smoke_gate.json", {"status": "PASS", "source_binding": source,
            "environment": environment, "execution_manifest_sha256": sha256_file(execution_path)})
        return config

    yield create, input_checks
    torch.set_num_threads(previous_threads)


def checkpoint_path(config):
    return Path(config["checkpoint_root"]) / "QFL_DFL_seed42" / "last.pt"


def load_checkpoint(config):
    return torch.load(checkpoint_path(config), map_location="cpu", weights_only=False)


def assert_exact_nested(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact_nested(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_exact_nested(first, second)
    else:
        assert actual == expected


def test_two_epochs_equal_one_epoch_plus_resume_with_actual_optimizer_and_rng(tiny_campaign):
    create, input_checks = tiny_campaign
    uninterrupted, interrupted = create("uninterrupted"), create("interrupted")
    assert train.train_one(uninterrupted, "QFL_DFL", 42, epoch_budget=2) == {"status": "PARTIAL", "completed_epochs": 2}
    assert train.train_one(interrupted, "QFL_DFL", 42, epoch_budget=1) == {"status": "PARTIAL", "completed_epochs": 1}
    before = load_checkpoint(interrupted)
    assert before["model"]["training_calls"] == 3
    # Deliberately perturb all RNGs between calls; the checkpoint must override.
    random.random()
    np.random.random(5)
    torch.rand(7)
    assert train.train_one(interrupted, "QFL_DFL", 42, epoch_budget=1) == {"status": "PARTIAL", "completed_epochs": 2}
    actual, expected = load_checkpoint(interrupted), load_checkpoint(uninterrupted)
    for key in ["model", "optimizer", "scaler", "rng", "best_map", "best_epoch", "epoch", "binding"]:
        assert_exact_nested(actual[key], expected[key])
    for checkpoint in (actual, expected):
        for epoch in checkpoint["history"]:
            epoch.pop("seconds")
    assert_exact_nested(actual["history"], expected["history"])
    assert actual["model"]["training_calls"] == 6
    assert [row["optimizer_steps"] for row in actual["history"]] == [2, 2]
    assert len(input_checks) == 2  # Only fresh runs reread all development inputs.
    receipt = read_json(Path(interrupted["output_dir"]) / "runs/QFL_DFL_seed42/run_manifest.json")
    assert receipt["epochs"] == 30
    assert receipt["test_evaluated"] is False


def test_pause_before_next_epoch_preserves_last_checkpoint(tiny_campaign):
    create, _ = tiny_campaign
    config = create("paused")
    train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
    before = sha256_file(checkpoint_path(config))
    pause = Path(config["output_dir"]) / "pause.request"
    pause.write_text("primary owns GPU", encoding="utf-8")
    with pytest.raises(train.YieldToPrimary):
        train.train_one(config, "QFL_DFL", 42, epoch_budget=1, pause_file=pause)
    assert sha256_file(checkpoint_path(config)) == before
    status = read_json(Path(config["output_dir"]) / "runs/QFL_DFL_seed42/status.json")
    assert status["status"] == "PAUSED_FOR_PRIMARY"
    assert status["completed_epochs"] == 1


def test_failed_or_stale_smoke_gate_rejects_before_training(tiny_campaign):
    create, input_checks = tiny_campaign
    for name, field, value in [("failed", "status", "FAIL"),
                                ("stale", "execution_manifest_sha256", "not-current")]:
        config = create(name)
        path = Path(config["output_dir"]) / "smoke_gate.json"
        gate = read_json(path)
        gate[field] = value
        write_json(path, gate)
        with pytest.raises(RuntimeError, match="smoke"):
            train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
        assert not checkpoint_path(config).exists()
    assert input_checks == []


def test_resume_rejects_checkpoint_from_different_binding(tiny_campaign):
    create, _ = tiny_campaign
    config = create("wrong_binding")
    train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
    saved = load_checkpoint(config)
    saved["binding"]["seed"] = 43
    torch.save(saved, checkpoint_path(config))
    before = sha256_file(checkpoint_path(config))
    with pytest.raises(RuntimeError, match="Resume checkpoint binding changed"):
        train.train_one(config, "QFL_DFL", 42, epoch_budget=1)
    assert sha256_file(checkpoint_path(config)) == before


def test_all_late_epochs_retained_even_when_best_is_epoch_one(tiny_campaign, monkeypatch):
    create, _ = tiny_campaign
    config = create('complete_nonbest_late')

    def decreasing_metrics(records, predictions, classes):
        value = 100. / (1 + predictions[0]['training_calls'])
        return {'map': value, 'ap75': value * .8, 'ap85': value * .6, 'wrong_class_fp': 0}

    monkeypatch.setattr(train, 'evaluate_quality_predictions', decreasing_metrics)
    # Split at the retention boundary to verify the persisted artifact index.
    partial = train.train_one(config, 'QFL_DFL', 42, epoch_budget=26)
    assert partial == {'status': 'PARTIAL', 'completed_epochs': 26}
    assert set(load_checkpoint(config)['retained_epochs']) == {'26'}
    result = train.train_one(config, 'QFL_DFL', 42, epoch_budget=4)
    assert result['status'] == 'COMPLETE'
    assert result['best_epoch'] == 1
    assert set(result['retained_epochs']) == {'26', '27', '28', '29', '30'}
    assert set(result['evaluation_manifests']) == {'best', 'epoch30', 'epochs26_30_mean'}
    for epoch in range(26, 31):
        artifacts = result['retained_epochs'][str(epoch)]
        assert set(artifacts) == {'checkpoint', 'predictions', 'metrics', 'stage_rows', 'evaluation'}
        train.verify_artifacts(artifacts)
        checkpoint = torch.load(artifacts['checkpoint']['path'], map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == epoch - 1
        assert int(checkpoint['model']['training_calls']) == epoch * 3
        predictions = read_json(artifacts['predictions']['path'])
        assert [row['image_id'] for row in predictions] == [1, 2]
        assert all(row['training_calls'] == epoch * 3 for row in predictions)
        rows = read_json(artifacts['stage_rows']['path'])
        assert len(rows) == 2
        assert rows[0]['stages']['final']['good75_max_score'] == .8
    assert result['endpoints']['best']['epoch'] == 1
    assert result['endpoints']['epoch30']['epoch'] == 30
    for endpoint in ['best', 'epoch30']:
        assert result['endpoints'][endpoint]['diagnostics']['final_no_good_box'] == 0
        assert result['endpoints'][endpoint]['diagnostics']['ground_truths'] == 2
    expected_mean = sum(100. / (1 + epoch * 3) for epoch in range(26, 31)) / 5
    assert result['endpoints']['epochs26_30_mean']['metrics']['map'] == expected_mean
    assert result['endpoints']['best']['metrics']['map'] > result['endpoints']['epoch30']['metrics']['map']
    # A completed run rechecks every endpoint artifact instead of retraining.
    before = sha256_file(checkpoint_path(config))
    assert train.train_one(config, 'QFL_DFL', 42)['status'] == 'COMPLETE'
    assert sha256_file(checkpoint_path(config)) == before


def test_stage_score_maxima_retain_good_box_quality_not_high_bad_box_score():
    from tools.sonar_fcos_quality.diagnostics import stage_rows_for_image
    record = {'image_id': 1, 'boxes': [[0., 0., 10., 10.]], 'labels': [1]}
    boxes = np.array([[0., 0., 10., 10.], [0., 0., 20., 20.], [1., 0., 11., 10.]])
    scores = np.array([.2, .95, .6])
    raw_scores = np.stack([np.zeros(3), scores], axis=1)
    labels = np.ones(3, dtype=int)
    rows = stage_rows_for_image(record, boxes, raw_scores, boxes, labels, [0, 1], [0, 1], .001)
    rows = train.add_stage_score_maxima(rows, record, boxes, raw_scores, boxes, scores,
                                       labels, [0, 1], [0, 1], .001)
    assert rows[0]['stages']['raw']['good75_max_score'] == .6
    assert rows[0]['stages']['topk']['good75_max_score'] == .6
    assert rows[0]['stages']['nms']['good75_max_score'] == .2
    assert rows[0]['stages']['final']['good75_max_score'] == .2
    # The .95 candidate has insufficient IoU and cannot rescue quality coverage.
    assert rows[0]['stages']['final']['good75_max_score'] < .5


def test_supplementary_analysis_preserves_all_training_rng_states(tiny_campaign, monkeypatch, tmp_path):
    original = train.evaluate_checkpoint

    def consuming_analysis(*args, **kwargs):
        random.random()
        np.random.random(4)
        torch.rand(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(train, 'evaluate_checkpoint', consuming_analysis)
    dataset, model = TinyDataset(None, 'val', 640), TinyModel()
    predictions, diagnostics = train.predict(model, dataset, {}, trace=True)
    before = cpu_rng_state()
    train.save_evaluation(tmp_path / 'analysis_rng', dataset.records, predictions, 1,
                          diagnostics, selection='epoch26', epoch=26)
    assert_exact_nested(cpu_rng_state(), before)


@pytest.mark.parametrize('mode', ['baseline_centerness', 'quality_sigmoid'])
def test_additive_tracer_matches_v1_native_output_and_stage_counts(mode):
    import types
    from torchvision.models.detection import _utils
    from torchvision.models.detection.fcos import FCOS
    from tools.sonar_fcos_quality.model import QualityFCOS

    model = nn.Module()
    model._sonar_quality_score_mode = mode
    model.box_coder = _utils.BoxLinearCoder(normalize_by_size=True)
    model.score_thresh, model.topk_candidates = .001, 1000
    model.nms_thresh, model.detections_per_img = .6, 100
    method = QualityFCOS.postprocess_detections if mode == 'quality_sigmoid' else FCOS.postprocess_detections
    model.postprocess_detections = types.MethodType(method, model)
    record = {'image_id': 1, 'boxes': [[0., 0., 8., 8.]], 'labels': [1], 'width': 32, 'height': 32}
    target = {'image_id': torch.tensor([1]), 'resize_scale': torch.tensor([1., 1.]),
              'pad_offset': torch.tensor([0., 0.]), 'original_size': torch.tensor([32, 32])}
    logits = torch.tensor([[[-12., torch.logit(torch.tensor(.3))],
                            [-12., torch.logit(torch.tensor(.7))]]])
    heads = {'cls_logits': [logits], 'bbox_regression': [torch.full((1, 2, 4), .5)],
             'bbox_ctrness': [torch.zeros(1, 2, 1)]}
    anchors = [[torch.tensor([[0., 0., 8., 8.], [12., 0., 20., 8.]])]]
    with train.StageDiagnosticTracer(model) as reference:
        reference.begin_batch([target], [record])
        original = model.postprocess_detections(heads, anchors, [(32, 32)])
    with train.ScoreStageTracer(model) as additive:
        additive.begin_batch([target], [record])
        actual = model.postprocess_detections(heads, anchors, [(32, 32)])
    assert_exact_nested(actual, original)
    assert additive.summary([record]) == reference.summary([record])
    for stage in train.STAGES:
        for key, value in reference.rows[0]['stages'][stage].items():
            assert additive.rows[0]['stages'][stage][key] == value
        expected_score = .3 if mode == 'quality_sigmoid' else float(np.sqrt(.3 * .5))
        assert additive.rows[0]['stages'][stage]['good75_max_score'] == pytest.approx(expected_score)


def test_six_completed_tiny_runs_summarize_all_endpoints_workpoints_and_cached_ci(tiny_campaign, monkeypatch):
    from tools.sonar_fcos_quality_v2 import analysis, campaign
    create, _ = tiny_campaign
    config = create('six_run_summary')
    output = Path(config['output_dir'])
    prereg_path = output / 'synthetic_preregistration.json'
    write_json(prereg_path, {'claim_boundaries': ['synthetic CPU fixture; no scientific conclusion']})
    config.update(preregistration_path=str(prereg_path), historical_reference_only={'fixture': True})
    manifest_path, manifest = train.require_manifest(config)
    campaign_manifest = {**manifest, 'samples': [{**record, 'split': 'val'}
                                               for record in TinyDataset(None, 'val', 640).records]}
    monkeypatch.setattr(campaign, 'source_binding', train.source_binding)
    monkeypatch.setattr(campaign, 'require_manifest', lambda cfg: (manifest_path, campaign_manifest))
    bootstrap_calls = []

    def fake_bootstrap(records, predictions_by_run, num_classes, *, n_resamples, random_seed):
        assert n_resamples == 2000 and random_seed == 20261012
        assert [record['image_id'] for record in records] == [1, 2]
        assert set(predictions_by_run) == {(variant, seed) for variant in ['AAA', 'QFL_DFL']
                                          for seed in [42, 43, 44]}
        values = {}
        widths = []
        for key, predicted in predictions_by_run.items():
            checkpoints = predicted if isinstance(predicted[0], list) else [predicted]
            widths.append(len(checkpoints))
            values[key] = sum(train.evaluate_quality_predictions(records, predictions, num_classes)['ap75']
                              for predictions in checkpoints) / len(checkpoints)
        deltas = {str(seed): values['QFL_DFL', seed] - values['AAA', seed] for seed in [42, 43, 44]}
        point = sum(deltas.values()) / 3
        bootstrap_calls.append(widths)
        return {'mean_paired_ap75_gain_pp': point, 'ci95_low_pp': point, 'ci95_high_pp': point,
                'seed_signs': {seed: 'positive' if value > 0 else 'negative' if value < 0 else 'zero'
                               for seed, value in deltas.items()}, 'synthetic_not_statistical_evidence': True}

    monkeypatch.setattr(analysis, 'bootstrap_suite', fake_bootstrap)
    for seed in [42, 43, 44]:
        for variant in ['AAA', 'QFL_DFL']:
            completed = train.train_one(config, variant, seed, epoch_budget=30)
            assert completed['status'] == 'COMPLETE'
    result = campaign.summarize(config)
    assert result['completed_runs'] == 6
    assert len(result['rows']) == 18
    assert {(row['view'], row['variant'], row['seed']) for row in result['rows']} == {
        (view, variant, seed) for view in ['best', 'epoch30', 'mean26_30']
        for variant in ['AAA', 'QFL_DFL'] for seed in [42, 43, 44]}
    assert len(result['workpoints']) == 36
    assert len({(row['view'], row['variant'], row['seed']) for row in result['workpoints']}) == 36
    assert {row['view'] for row in result['workpoints']} == {'best', 'epoch30', 'epoch26', 'epoch27', 'epoch28', 'epoch29'}
    assert set(result['confidence_intervals']) == {'best', 'epoch30', 'mean26_30'}
    assert bootstrap_calls == [[1] * 6, [1] * 6, [5] * 6]
    report = (output / '支线结果表.md').read_text(encoding='utf-8')
    assert all(view in report for view in ['best', 'epoch30', 'mean26_30'])
    assert report.count('图像配对bootstrap 95%区间') == 3
    assert 'FCOS权重只归档' in report and 'C01c暂不执行' in report
    workpoint_report = (output / '工作点诊断表.md').read_text(encoding='utf-8')
    assert workpoint_report.count('|固定0.25|') == 36
    assert workpoint_report.count('|目标FPPI0.5|') == 36
    exported_manifest = read_json(output / 'pretrained/pretrained_manifest.json')
    exported_path = Path(exported_manifest['checkpoint'])
    assert sha256_file(exported_path) == exported_manifest['sha256']
    assert exported_manifest['transfer_status'] == 'FCOS_ARCHIVE_ONLY_NOT_E0_PRETRAINING'
    selected_summary = read_json(output / 'runs' /
        f"{exported_manifest['variant']}_seed{exported_manifest['seed']}" / 'summary.json')
    assert exported_manifest['sha256'] == selected_summary['best_artifacts']['checkpoint']['sha256']
    cached_hashes = {view: sha256_file(output / 'confidence_intervals' / f'{view}.json')
                     for view in ['best', 'epoch30', 'mean26_30']}

    def should_use_cache(*args, **kwargs):
        raise AssertionError('Unchanged summaries must reuse the identity-bound bootstrap cache')

    monkeypatch.setattr(analysis, 'bootstrap_suite', should_use_cache)
    repeated = campaign.summarize(config)
    for key in ['rows', 'decision', 'workpoints', 'confidence_intervals']:
        assert repeated[key] == result[key]
    assert cached_hashes == {view: sha256_file(output / 'confidence_intervals' / f'{view}.json') for view in cached_hashes}
    assert sha256_file(exported_path) == exported_manifest['sha256']
