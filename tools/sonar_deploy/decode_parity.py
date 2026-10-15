"""Independent same-integer-head CPU/CUDA decode audit on frozen valid only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools/sonar_analysis"), str(ROOT / "scripts")]
from tools.sonar_deploy.ptq_export import IntegerGraphModel
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss, decode_detection_predictions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def check_decode_device(package, output, device="cuda:0"):
    from phase5a_quantization.common import build_network, baseline
    import baseline as frozen
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for matched decode-device diagnostic")
    package, output = Path(package), Path(output)
    graph = json.loads((package / "graph.json").read_text())
    provenance = graph["provenance"]
    template, aid = build_network(provenance["network"])
    if aid != provenance["architecture_id"]:
        raise ValueError("candidate architecture mismatch")
    config = baseline(); proto = config["evaluation_protocol"]
    frozen.seed_all(int(provenance["seed"]))
    integer = IntegerGraphModel(package, template).to(device).eval()
    criterion = DenseAnchorDetectionLoss(template.loss_config(config)).cpu()
    gpu_predictions = []

    class CPUHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.integer = integer

        def forward(self, image):
            heads = self.integer(image)
            gpu = decode_detection_predictions(heads, config=criterion.config,
                score_threshold=proto["ap_decode_score_threshold"],
                nms_iou_threshold=proto["nms_iou_threshold"],
                pre_nms_topk=proto["pre_nms_topk"],
                max_detections=proto["post_nms_max_detections_per_image"])
            gpu_predictions.extend({key: item[key].cpu().tolist() for key in ("boxes", "scores", "labels")} for item in gpu)
            return {"predictions": tuple(t.cpu() for t in heads["predictions"])}

    loader = frozen.make_loader(config, frozen.resolve(config["dataset"]["data_dir"]), "valid", 42, 0)
    if len(loader.dataset) != 99:
        raise ValueError("expected frozen valid99")
    actual_split = frozen.bind_dataset(loader)
    frozen_split = json.loads((ROOT / "reports/baseline_sonar_frozen/freeze_manifest.json").read_text())["datasets"]["valid"]
    expected_entries = [dict(row, image=row["image"].replace("\\", "/"), annotation=row["annotation"].replace("\\", "/"))
                        for row in frozen_split["entries"]]
    if actual_split["entries"] != expected_entries or provenance["split_manifest_sha256"] != frozen_split["sha256"]:
        raise ValueError("valid identity differs from frozen package binding")
    started = time.perf_counter()
    cpu_metrics, cpu_records = frozen.evaluate(CPUHeads().eval(), loader, criterion, config, device)
    if len(cpu_records) != len(gpu_predictions) or len(cpu_records) != 99:
        raise ValueError("decode record count mismatch")
    gpu_records = [dict(row, prediction=prediction) for row, prediction in zip(cpu_records, gpu_predictions)]
    gpu_metrics = frozen.metrics_from_records(gpu_records, config)
    keys = ("AP50", "AP75", "mAP50_95", "Precision", "Recall")
    metric_deltas = {k: float(cpu_metrics[k] - gpu_metrics[k]) for k in keys}
    rows = []
    for cpu, gpu in zip(cpu_records, gpu_records):
        a, b = cpu["prediction"], gpu["prediction"]
        row = dict(image=cpu["image"], cpu_detections=len(a["scores"]), gpu_detections=len(b["scores"]), identical=a == b)
        if len(a["scores"]) == len(b["scores"]) and a["scores"]:
            row.update(boxes_max_abs_error=float(np.max(np.abs(np.asarray(a["boxes"])-np.asarray(b["boxes"])))),
                       scores_max_abs_error=float(np.max(np.abs(np.asarray(a["scores"])-np.asarray(b["scores"])))),
                       different_label_positions=int(np.count_nonzero(np.asarray(a["labels"])-np.asarray(b["labels"]))))
        rows.append(row)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("cpu_predictions.json", cpu_records), ("cuda_predictions.json", gpu_records)):
        (output / name).write_text(json.dumps(records, indent=2), encoding="utf-8")
    result = dict(status="METRICS_IDENTICAL" if not any(metric_deltas.values()) else "DECODE_DEVICE_METRIC_DIFFERENCE",
                  scope="diagnostic_only_same_integer_heads_no_protocol_replacement",
                  cpu_metrics=cpu_metrics, cuda_metrics=gpu_metrics, cpu_minus_cuda=metric_deltas,
                  prediction_identical_images=sum(r["identical"] for r in rows), num_images=99, per_image=rows,
                  graph_sha256=sha(package / "graph.json"), export_manifest_sha256=sha(package / "export_manifest.json"),
                  baseline_sha256=sha(ROOT / "configs/baseline_sonar_frozen.yaml"),
                  evaluation_source_sha256=sha(ROOT / "tools/sonar_analysis/baseline.py"),
                  decode_source_sha256=sha(ROOT / "src/hwnas_fpga/training/detection_trainer.py"),
                  source_sha256=sha(__file__), provenance=provenance, device=device,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  elapsed_seconds=time.perf_counter()-started, split="valid", test_evaluated=False)
    (output / "DECODE_DEVICE_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); torch.set_num_threads(2)
    result = check_decode_device(args.package, args.output, args.device)
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
