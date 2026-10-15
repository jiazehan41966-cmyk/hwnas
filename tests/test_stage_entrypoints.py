"""Actual PTQ/evaluation/bundle entrypoints on synthetic, untrained fixtures.

These tests never optimize weights or read research datasets. Their temporary
receipts explicitly identify initialized weights and nonformal calibration.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image
import pytest
import torch
import yaml

import experiment
from experiment import canonical_hash, create_run, save_json, sha_file
from models import build_model
from quant import Q
from quant import executor
from scripts import deploy, quantize, tables
from scripts.common import baseline_config
from space import Genotype
from train import DenseAnchorDetectionLoss, evaluate, loss_config_for_model
from train.data import bind_dataset, make_loader
from train.training import _checkpoint


def _dataset(folder):
    folder.mkdir()
    (folder / "data.yaml").write_text("names: [cylider, manta]\n", encoding="utf-8")
    for split, brightness in (("train", 127), ("valid", 96)):
        image = folder / split / "images" / "fixture.png"
        label = folder / split / "labels" / "fixture.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        Image.new("L", (8, 8), brightness).save(image)
        label.write_text("", encoding="utf-8")  # explicit negative; no missing annotations


def test_actual_quantize_deploy_and_tables_without_hardware_measurement(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        data = tmp_path / "synthetic_dataset"
        _dataset(data)
        source_config = experiment.ROOT / "configs/search.yaml"
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        config["dataset"]["root"] = str(data)
        config["dataset"]["id"] = "SYNTHETIC_UNTRAINED_ENTRYPOINT_FIXTURE"
        config_path = tmp_path / "fixture_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        training_run = tmp_path / "initialized_checkpoint_run"
        _, run = create_run(config_path, training_run,
                            command="pytest initialized-checkpoint fixture; no training", require_clean=False)
        # Preserve real source/config hashes and all artifact checks. Only the
        # dirty-source execution gate is bypassed inside this temporary fixture.
        verify_run = experiment.verify_run
        def fixture_verify_run(path, *, current_source=False):
            return verify_run(path, current_source=False)
        monkeypatch.setattr(quantize, "verify_run", fixture_verify_run)
        monkeypatch.setattr(deploy, "verify_run", fixture_verify_run)
        real_calibration = executor.TrainCalibration
        class OneSyntheticTrainImage(real_calibration):
            def __init__(self, root, entries, **kwargs):
                assert kwargs == {"num_images": 64}
                super().__init__(root, entries, num_images=1, formal=False)
        monkeypatch.setattr(executor, "TrainCalibration", OneSyntheticTrainImage)
        baseline = baseline_config(config, config_path=config_path)
        genotype = Genotype(False, (3, 3, 3))
        model = build_model(genotype, seed=42).eval()
        train = make_loader(baseline, data, "train", 42, 0)
        valid = make_loader(baseline, data, "valid", 42, 0)
        datasets = dict(train=bind_dataset(train), valid=bind_dataset(valid))
        binding = dict(genotype=genotype.to_dict(), arch_hash=genotype.arch_hash, seed=42, budget=40,
                       config_sha256=canonical_hash(baseline), dataset_sha256=canonical_hash(datasets),
                       test_evaluated=False)
        # A real evaluation supplies the fixture receipt's FP32 metric. There is
        # no train_epoch call and no optimizer in this test.
        metrics, _ = evaluate(model, valid, DenseAnchorDetectionLoss(
            loss_config_for_model(model, baseline)), baseline, "cpu")
        checkpoint = training_run / "training" / genotype.arch_hash / "seed_42" / "initialized_fixture.pt"
        checkpoint.parent.mkdir(parents=True)
        _checkpoint(checkpoint, dict(binding=binding, epoch=0, model_state_dict=model.state_dict(),
                                    valid=metrics, fixture_untrained=True), binding)
        receipt = dict(binding=binding, best_epoch=0, mAP50_95=metrics["mAP50_95"],
                       checkpoint=checkpoint.name, checkpoint_sha256=sha_file(checkpoint),
                       status="SYNTHETIC_INITIALIZED_WEIGHTS_FIXTURE_NO_TRAINING", test_evaluated=False)
        receipt_path = training_run / "retrain_fixture.json"
        save_json(receipt_path, receipt)
        quantized = tmp_path / "quantized"
        result = quantize.execute(training_run, quantized, profile="W8A8", device="cpu")
        assert math.isfinite(result["mAP50_95"])
        assert result["software_integer_parity"] == "PASS_SOFTWARE_INTEGER_FAKE_PARITY"
        assert result["arch_hash"] == genotype.arch_hash
        assert result["q_hash"] == Q(genotype).q_hash
        calibration = json.loads((quantized / "calibration.json").read_text(encoding="utf-8"))
        assert calibration["formal_calibration_count"] is False
        assert calibration["number_of_calibration_images"] == 1
        assert run["config_hash"] == json.loads((quantized / "run.json").read_text(encoding="utf-8"))["config_hash"]
        bundle = tmp_path / "bundle"
        monkeypatch.setattr(sys, "argv", ["scripts.deploy", "--config", str(config_path),
                            "--quantized-run", str(quantized), "--output", str(bundle)])
        deploy.main()
        readiness = json.loads((bundle / "deployment_readiness.json").read_text(encoding="utf-8"))
        assert readiness["arch_hash"] == genotype.arch_hash
        assert readiness["q_hash"] == Q(genotype).q_hash
        assert readiness["hardware"]["status"] == "NEEDS_MEASUREMENT"
        assert readiness["deployment_claim"] is False
        assert all(v is None for v in readiness["hardware"]["metrics"].values())
        executor.verify_saved_artifacts(bundle / "integer", expected_arch_hash=genotype.arch_hash,
                                        expected_q_hash=Q(genotype).q_hash)
        table_folder = tmp_path / "tables"
        report = tables.generate(bundle, table_folder)
        assert report["accuracy_rows"] == 1 and report["hardware_rows"] == 1
        assert report["pareto_rows"] == 0
        with (table_folder / "table1_accuracy.csv").open(encoding="utf-8-sig", newline="") as f:
            score = list(csv.DictReader(f))[0]
        assert score["status"] == "EVALUATED"
        assert float(score["mAP50_95"]) == result["mAP50_95"]
        with (table_folder / "table2_hardware.csv").open(encoding="utf-8-sig", newline="") as f:
            hardware = list(csv.DictReader(f))[0]
        assert hardware["status"] == "NEEDS_MEASUREMENT"
        assert all(hardware[k] == "" for k in ("latency_us", "lut", "dsp", "bram18", "power_w"))
        # Exercise an actual CLI rejection after the immutable leaf changes.
        graph_path = quantized / "ptq" / "numeric_graph.json"
        graph_bytes = graph_path.read_bytes()
        graph_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["scripts.deploy", "--config", str(config_path),
                            "--quantized-run", str(quantized), "--output", str(tmp_path / "rejected_bundle")])
        with pytest.raises(ValueError, match="SHA256 drift"):
            deploy.main()
        assert not (tmp_path / "rejected_bundle").exists()
        graph_path.write_bytes(graph_bytes)
        duration = time.monotonic() - started
        save_json(tmp_path / "stage_entrypoints_report.json", dict(
            fixture="SYNTHETIC_INITIALIZED_WEIGHTS_NO_TRAINING", elapsed_seconds=duration,
            config_hash=run["config_hash"], arch_hash=genotype.arch_hash, q_hash=Q(genotype).q_hash,
            formal_calibration=False, test_evaluated=False, deployment_claim=False,
            software_integer_parity=result["software_integer_parity"],
            hardware_status=readiness["hardware"]["status"], table_rows=report))
    finally:
        torch.set_num_threads(previous_threads)
        print(f"Synthetic untrained quantize/deploy/tables seconds: {time.monotonic()-started:.3f}")
