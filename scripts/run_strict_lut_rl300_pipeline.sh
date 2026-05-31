#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/hwnas/hwnas}"
PYTHON="${PYTHON:-${REPO_DIR}/.venv/bin/python}"
CONFIG="configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_av7k325.yaml"
SEARCH_RUN="nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_v1"
SEARCH_DIR="results/${SEARCH_RUN}"
RETRAIN_RUN="nas_board_lut_strict_current84_arch84_full_best_retrain150_cuda_v1"
RETRAIN_DIR="results/${RETRAIN_RUN}"
EXPORT_DIR="${RETRAIN_DIR}/export"
FINAL_CKPT="${RETRAIN_DIR}/checkpoints/final_best_model.pt"
FINAL_EVAL="${RETRAIN_DIR}/results/final_eval_summary.json"
FINAL_REPORT="${RETRAIN_DIR}/results/final_report.json"

cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[pipeline] started_at=$(date --iso-8601=seconds)"
echo "[pipeline] repo=${REPO_DIR}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[pipeline] validating strict LUT pytest"
"${PYTHON}" -m pytest tests/test_nas_board_lut_strict_current84_arch84.py

echo "[pipeline] validating config"
"${PYTHON}" - <<'PY'
from hwnas_fpga.runtime import load_config
cfg = load_config("configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_av7k325.yaml")
assert cfg["project"]["run_name"] == "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_v1"
assert cfg["search"]["episodes"] == 300
assert cfg["search"]["eval_epochs"] == 10
assert cfg["search"]["eval_early_stopping_patience"] == 3
assert cfg["search"]["pareto"]["topk"] == 8
assert cfg["training"]["epochs"] == 150
assert cfg["training"]["batch_size"] == 32
assert cfg["training"]["early_stopping_patience"] is None
assert cfg["search"]["selection_metric"] == "macro_f1"
assert cfg["hardware"]["strict_formal_lut"] is True
assert cfg["hardware"]["lut_allow_shape_only_match"] is False
assert cfg["hardware"]["lut_enable_interpolation"] is False
assert cfg["hardware"]["use_dummy_lut"] is False
print("CONFIG_OK")
PY

if [[ -f "${SEARCH_DIR}/results/summary.json" ]]; then
  SEARCH_STATUS="$("${PYTHON}" -c "import json; print(json.load(open('${SEARCH_DIR}/results/summary.json', encoding='utf-8')).get('status'))")"
else
  SEARCH_STATUS=""
fi

if [[ "${SEARCH_STATUS}" == "completed" ]]; then
  echo "[pipeline] search already completed; skipping search"
elif [[ -d "${SEARCH_DIR}" ]]; then
  if [[ -f "${SEARCH_DIR}/checkpoints/controller_latest.pt" && -f "${SEARCH_DIR}/checkpoints/search_state.json" && -f "${SEARCH_DIR}/results/candidates.jsonl" ]]; then
    echo "[pipeline] resuming RL search from ${SEARCH_DIR}"
    "${PYTHON}" run_rl_search.py --config "${CONFIG}" --device cuda --resume
  else
    echo "[pipeline] found non-resumable existing search dir: ${SEARCH_DIR}" >&2
    exit 20
  fi
else
  echo "[pipeline] starting RL search"
  "${PYTHON}" run_rl_search.py --config "${CONFIG}" --device cuda
fi

echo "[pipeline] validating search artifacts and strict LUT stats"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
run = Path("results/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_v1")
required = [
    run / "results" / "summary.json",
    run / "results" / "best_candidate.json",
    run / "results" / "pareto_selection.json",
    run / "results" / "lut_stats.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing search artifacts: {missing}")
summary = json.loads(required[0].read_text(encoding="utf-8"))
if summary.get("status") != "completed":
    raise SystemExit(f"search did not complete: {summary.get('status')}")
lut = json.loads(required[3].read_text(encoding="utf-8"))
if lut.get("true_misses") != 0:
    raise SystemExit(f"true_misses != 0: {lut.get('true_misses')}")
if lut.get("deferred_hits") != 0:
    raise SystemExit(f"deferred_hits != 0: {lut.get('deferred_hits')}")
if lut.get("strict_formal_lut") is not True:
    raise SystemExit(f"strict_formal_lut is not true: {lut.get('strict_formal_lut')}")
print("SEARCH_ARTIFACTS_OK")
PY

if [[ -f "${FINAL_CKPT}" ]]; then
  echo "[pipeline] retrain checkpoint already exists; skipping retrain"
elif [[ -d "${RETRAIN_DIR}" ]]; then
  echo "[pipeline] found existing retrain dir without final checkpoint: ${RETRAIN_DIR}" >&2
  exit 30
else
  echo "[pipeline] retraining best candidate for 150 epochs"
  "${PYTHON}" run_retrain.py \
    --run-dir "${SEARCH_DIR}" \
    --epochs 150 \
    --device cuda \
    --run-name "${RETRAIN_RUN}"
fi

if [[ ! -f "${FINAL_CKPT}" ]]; then
  echo "[pipeline] missing final checkpoint: ${FINAL_CKPT}" >&2
  exit 31
fi

echo "[pipeline] evaluating final checkpoint"
"${PYTHON}" run_eval_checkpoint.py \
  --checkpoint "${FINAL_CKPT}" \
  --config "${SEARCH_DIR}/config.yaml" \
  --output "${FINAL_EVAL}" \
  --device cuda

echo "[pipeline] exporting ONNX, INT8 weights, and HLS stub"
"${PYTHON}" run_export.py \
  --checkpoint "${FINAL_CKPT}" \
  --config "${SEARCH_DIR}/config.yaml" \
  --output-dir "${EXPORT_DIR}" \
  --device cuda \
  --quantize-int8 \
  --prepare-hls \
  --board alinx_av7k325 \
  --clock-mhz 200

echo "[pipeline] validating export artifacts"
test -f "${EXPORT_DIR}/model.onnx"
test -f "${EXPORT_DIR}/model.json"
test -f "${RETRAIN_DIR}/checkpoints/quantized_weights_int8.pt"
test -f "${RETRAIN_DIR}/checkpoints/quantized_weights_int8.json"
test -d "${EXPORT_DIR}/hls_project"

echo "[pipeline] writing final report"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path

search = Path("results/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_v1")
retrain = Path("results/nas_board_lut_strict_current84_arch84_full_best_retrain150_cuda_v1")
export = retrain / "export"
best = json.loads((search / "results" / "best_candidate.json").read_text(encoding="utf-8"))
summary = json.loads((search / "results" / "summary.json").read_text(encoding="utf-8"))
lut = json.loads((search / "results" / "lut_stats.json").read_text(encoding="utf-8"))
final_eval = json.loads((retrain / "results" / "final_eval_summary.json").read_text(encoding="utf-8"))
retrain_summary_path = retrain / "results" / "retrain_summary.json"
retrain_summary = json.loads(retrain_summary_path.read_text(encoding="utf-8")) if retrain_summary_path.exists() else {}

candidate = best["candidate"]
metrics = candidate.get("metrics", {})
final_metrics = final_eval.get("metrics", {})
report = {
    "status": "completed",
    "search_run_dir": str(search),
    "retrain_run_dir": str(retrain),
    "best_arch_id": candidate.get("arch_id"),
    "best_network_structure": candidate.get("encoding"),
    "search_stage_metrics": {
        "macro_f1": metrics.get("macro_f1"),
        "top1": metrics.get("top1"),
        "top5": metrics.get("top5"),
        "latency_ms": metrics.get("latency_ms"),
        "lut": metrics.get("lut"),
        "dsp": metrics.get("dsp"),
        "bram": metrics.get("bram"),
        "power_w": metrics.get("power_w"),
    },
    "final_retrain_validation_metrics": {
        "macro_f1": final_metrics.get("macro_f1"),
        "top1": final_metrics.get("top1"),
        "top5": final_metrics.get("top5"),
        "loss": final_metrics.get("loss"),
        "num_samples": final_metrics.get("num_samples"),
    },
    "retrain_checkpoint_metrics": retrain_summary.get("metrics"),
    "estimated_hardware_cost": {
        "latency_ms": metrics.get("latency_ms"),
        "lut": metrics.get("lut"),
        "dsp": metrics.get("dsp"),
        "bram": metrics.get("bram"),
        "power_w": metrics.get("power_w"),
    },
    "strict_lut_full_hit": (
        lut.get("true_misses") == 0
        and lut.get("deferred_hits") == 0
        and lut.get("strict_formal_lut") is True
    ),
    "lut_stats": {
        "hits": lut.get("hits"),
        "misses": lut.get("misses"),
        "true_misses": lut.get("true_misses"),
        "deferred_hits": lut.get("deferred_hits"),
        "strict_formal_lut": lut.get("strict_formal_lut"),
        "hit_rate": lut.get("hit_rate"),
    },
    "artifacts": {
        "final_best_model_pt": str(retrain / "checkpoints" / "final_best_model.pt"),
        "final_best_model_pt_exists": (retrain / "checkpoints" / "final_best_model.pt").exists(),
        "onnx": str(export / "model.onnx"),
        "onnx_exists": (export / "model.onnx").exists(),
        "int8_weight_package": str(retrain / "checkpoints" / "quantized_weights_int8.pt"),
        "int8_weight_package_exists": (retrain / "checkpoints" / "quantized_weights_int8.pt").exists(),
        "hls_project_dir": str(export / "hls_project"),
        "hls_project_dir_exists": (export / "hls_project").is_dir(),
    },
    "search_summary": {
        "total_evaluated": summary.get("total_evaluated"),
        "feasible": summary.get("feasible"),
        "infeasible": summary.get("infeasible"),
        "selection_metric": summary.get("extra", {}).get("selection_metric"),
    },
    "hardware_latency_note": (
        "NAS latency is a strict LUT aggregate estimate. Do not claim real "
        "candidate end-to-end board latency until a complete-network harness "
        "and COM5 measurement are run."
    ),
    "recommended_next_step": "Run full-network hardware harness plus COM5 real end-to-end validation.",
}
target = retrain / "results" / "final_report.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

echo "[pipeline] completed_at=$(date --iso-8601=seconds)"
