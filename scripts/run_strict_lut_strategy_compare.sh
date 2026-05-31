#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/hwnas/hwnas}"
PYTHON="${PYTHON:-${REPO_DIR}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
RUN_EXPORTS="${RUN_EXPORTS:-1}"

RL_RUN="nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_v1"
RANDOM_CONFIG="configs/search/nas_board_lut_strict_current84_arch84_nksid_full_random300_eval10_cuda_av7k325.yaml"
RANDOM_RUN="nas_board_lut_strict_current84_arch84_nksid_full_random300_eval10_cuda_v1"
RANDOM_RETRAIN_RUN="nas_board_lut_strict_current84_arch84_random300_best_retrain150_cuda_v1"
PROXYLESS_CONFIG="configs/search/nas_board_lut_strict_current84_arch84_nksid_full_proxyless300_eval10_cuda_av7k325.yaml"
PROXYLESS_RUN="nas_board_lut_strict_current84_arch84_nksid_full_proxyless300_eval10_cuda_official_strict_v2"
PROXYLESS_RETRAIN_RUN="nas_board_lut_strict_current84_arch84_proxyless300_official_strict_best_retrain150_cuda_v1"
COMPARE_REPORT="results/nas_board_lut_strict_current84_arch84_strategy_compare_rl_random_proxyless_v1.json"

cd "${REPO_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ensure_proxyless_submodule() {
  if [[ ! -f reference/ProxylessNAS/search/nas_manager.py ]]; then
    echo "[compare] initializing official ProxylessNAS submodule"
    git submodule update --init --recursive reference/ProxylessNAS
  fi
  if [[ ! -f reference/ProxylessNAS/search/nas_manager.py ]]; then
    echo "[compare] official ProxylessNAS clone is missing under reference/ProxylessNAS" >&2
    exit 20
  fi
  if ! grep -q "1x1_Conv" reference/ProxylessNAS/search/modules/mix_op.py; then
    echo "[compare] applying HW-NAS strict-current84 ProxylessNAS compatibility patch"
    git -C reference/ProxylessNAS apply ../../patches/proxylessnas_hwnas_strict_current84.patch
  fi
}

echo "[compare] started_at=$(date --iso-8601=seconds)"
echo "[compare] repo=${REPO_DIR}"
"${PYTHON}" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

echo "[compare] validating strict LUT pytest"
"${PYTHON}" -m pytest tests/test_nas_board_lut_strict_current84_arch84.py

validate_config() {
  local config="$1"
  local method="$2"
  local run_name="$3"
  "${PYTHON}" - "${config}" "${method}" "${run_name}" <<'PY'
import sys
from hwnas_fpga.runtime import load_config

config_path, method, run_name = sys.argv[1:4]
cfg = load_config(config_path)
assert cfg["project"]["run_name"] == run_name
assert cfg["search"]["method"] == method
assert cfg["search"]["num_candidates"] == 300
assert cfg["search"]["episodes"] == 300
assert cfg["search"]["eval_epochs"] == 10
assert cfg["search"]["eval_early_stopping_patience"] == 3
assert cfg["search"]["pareto"]["topk"] == 8
assert cfg["training"]["epochs"] == 150
assert cfg["training"]["batch_size"] == 32
assert cfg["training"]["early_stopping_patience"] is None
assert cfg["search"]["selection_metric"] == "macro_f1"
assert cfg["dataset"]["num_workers"] == 8
assert cfg["hardware"]["strict_formal_lut"] is True
assert cfg["hardware"]["lut_allow_shape_only_match"] is False
assert cfg["hardware"]["lut_enable_interpolation"] is False
assert cfg["hardware"]["use_dummy_lut"] is False
print(f"CONFIG_OK {config_path}")
PY
}

search_status() {
  local run_dir="$1"
  if [[ -f "${run_dir}/results/summary.json" ]]; then
    "${PYTHON}" - "${run_dir}/results/summary.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
PY
  fi
}

run_search() {
  local config="$1"
  local run_name="$2"
  local run_dir="results/${run_name}"
  local status
  status="$(search_status "${run_dir}" || true)"
  if [[ "${status}" == "completed" ]]; then
    echo "[compare] search already completed: ${run_name}"
    return
  fi
  if [[ -d "${run_dir}" ]]; then
    echo "[compare] found incomplete non-resumable search dir: ${run_dir}" >&2
    echo "[compare] move or archive that directory before restarting this baseline." >&2
    exit 21
  fi
  echo "[compare] starting search: ${run_name}"
  "${PYTHON}" run_search.py --config "${config}" --device "${DEVICE}"
}

run_official_proxyless_search() {
  local config="$1"
  local run_name="$2"
  local run_dir="results/${run_name}"
  local status
  status="$(search_status "${run_dir}" || true)"
  if [[ "${status}" == "completed" ]]; then
    echo "[compare] official proxyless search already completed: ${run_name}"
    return
  fi
  if [[ -d "${run_dir}" ]]; then
    echo "[compare] found incomplete non-resumable official proxyless dir: ${run_dir}" >&2
    echo "[compare] move or archive that directory before restarting this baseline." >&2
    exit 22
  fi
  ensure_proxyless_submodule
  echo "[compare] starting official ProxylessNAS clone search: ${run_name}"
  "${PYTHON}" run_official_proxyless_nksid.py --config "${config}" --device "${DEVICE}"
}

validate_search() {
  local run_name="$1"
  local run_dir="results/${run_name}"
  "${PYTHON}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
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
print(f"SEARCH_ARTIFACTS_OK {run}")
PY
}

run_retrain_eval_export() {
  local search_run="$1"
  local retrain_run="$2"
  local search_dir="results/${search_run}"
  local retrain_dir="results/${retrain_run}"
  local final_ckpt="${retrain_dir}/checkpoints/final_best_model.pt"
  local final_eval="${retrain_dir}/results/final_eval_summary.json"
  local export_dir="${retrain_dir}/export"

  if [[ -f "${final_ckpt}" ]]; then
    echo "[compare] retrain checkpoint already exists: ${retrain_run}"
  elif [[ -d "${retrain_dir}" ]]; then
    echo "[compare] found existing retrain dir without final checkpoint: ${retrain_dir}" >&2
    exit 31
  else
    echo "[compare] retraining best candidate: ${retrain_run}"
    "${PYTHON}" run_retrain.py \
      --run-dir "${search_dir}" \
      --epochs 150 \
      --device "${DEVICE}" \
      --run-name "${retrain_run}"
  fi

  echo "[compare] evaluating final checkpoint: ${retrain_run}"
  "${PYTHON}" run_eval_checkpoint.py \
    --checkpoint "${final_ckpt}" \
    --config "${search_dir}/config.yaml" \
    --output "${final_eval}" \
    --device "${DEVICE}"

  if [[ "${RUN_EXPORTS}" == "1" ]]; then
    echo "[compare] exporting model: ${retrain_run}"
    "${PYTHON}" run_export.py \
      --checkpoint "${final_ckpt}" \
      --config "${search_dir}/config.yaml" \
      --output-dir "${export_dir}" \
      --device "${DEVICE}" \
      --quantize-int8 \
      --prepare-hls \
      --board alinx_av7k325 \
      --clock-mhz 200
    test -f "${export_dir}/model.onnx"
    test -f "${export_dir}/model.json"
    test -f "${retrain_dir}/checkpoints/quantized_weights_int8.pt"
    test -f "${retrain_dir}/checkpoints/quantized_weights_int8.json"
    test -d "${export_dir}/hls_project"
  fi
}

write_compare_report() {
  "${PYTHON}" - "${COMPARE_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
runs = {
    "rl": {
        "search_run": "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_v1",
        "retrain_run": "nas_board_lut_strict_current84_arch84_full_best_retrain150_cuda_v1",
    },
    "random": {
        "search_run": "nas_board_lut_strict_current84_arch84_nksid_full_random300_eval10_cuda_v1",
        "retrain_run": "nas_board_lut_strict_current84_arch84_random300_best_retrain150_cuda_v1",
    },
    "proxyless": {
        "search_run": "nas_board_lut_strict_current84_arch84_nksid_full_proxyless300_eval10_cuda_official_strict_v2",
        "retrain_run": "nas_board_lut_strict_current84_arch84_proxyless300_official_strict_best_retrain150_cuda_v1",
    },
}

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

report = {
    "status": "completed",
    "comparison_scope": "strict current84 arch84 NKSID fold-0, macro_f1 selection, batch_size 32, retrain 150 epochs",
    "latency_note": "NAS latency is a strict LUT aggregate estimate, not COM5 board end-to-end latency.",
    "methods": {},
}
for method, item in runs.items():
    search = Path("results") / item["search_run"]
    retrain = Path("results") / item["retrain_run"]
    summary = load_json(search / "results" / "summary.json")
    best = load_json(search / "results" / "best_candidate.json")
    pareto = load_json(search / "results" / "pareto_selection.json")
    lut = load_json(search / "results" / "lut_stats.json")
    final_eval = load_json(retrain / "results" / "final_eval_summary.json")
    retrain_summary = load_json(retrain / "results" / "retrain_summary.json")
    candidate = (best or {}).get("candidate") or {}
    metrics = candidate.get("metrics") or {}
    final_metrics = (final_eval or {}).get("metrics") or {}
    report["methods"][method] = {
        "search_run_dir": str(search),
        "retrain_run_dir": str(retrain),
        "search_completed": (summary or {}).get("status") == "completed",
        "best_arch_id": candidate.get("arch_id"),
        "search_metrics": {
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
        "retrain_checkpoint_metrics": (retrain_summary or {}).get("metrics"),
        "search_summary": {
            "total_evaluated": (summary or {}).get("total_evaluated"),
            "feasible": (summary or {}).get("feasible"),
            "infeasible": (summary or {}).get("infeasible"),
            "selection_metric": ((summary or {}).get("extra") or {}).get("selection_metric"),
        },
        "pareto_front_size": (pareto or {}).get("pareto_front_size"),
        "strict_lut_full_hit": (
            (lut or {}).get("true_misses") == 0
            and (lut or {}).get("deferred_hits") == 0
            and (lut or {}).get("strict_formal_lut") is True
        ),
        "lut_stats": {
            "hits": (lut or {}).get("hits"),
            "misses": (lut or {}).get("misses"),
            "true_misses": (lut or {}).get("true_misses"),
            "deferred_hits": (lut or {}).get("deferred_hits"),
            "hit_rate": (lut or {}).get("hit_rate"),
        },
        "artifacts": {
            "final_best_model_pt_exists": (retrain / "checkpoints" / "final_best_model.pt").exists(),
            "onnx_exists": (retrain / "export" / "model.onnx").exists(),
            "int8_weight_package_exists": (retrain / "checkpoints" / "quantized_weights_int8.pt").exists(),
        },
        "best_network_structure": candidate.get("encoding"),
    }

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
}

validate_config "${RANDOM_CONFIG}" "random" "${RANDOM_RUN}"
validate_config "${PROXYLESS_CONFIG}" "proxyless" "${PROXYLESS_RUN}"
ensure_proxyless_submodule

run_search "${RANDOM_CONFIG}" "${RANDOM_RUN}"
validate_search "${RANDOM_RUN}"
run_retrain_eval_export "${RANDOM_RUN}" "${RANDOM_RETRAIN_RUN}"

run_official_proxyless_search "${PROXYLESS_CONFIG}" "${PROXYLESS_RUN}"
validate_search "${PROXYLESS_RUN}"
run_retrain_eval_export "${PROXYLESS_RUN}" "${PROXYLESS_RETRAIN_RUN}"

write_compare_report
echo "[compare] completed_at=$(date --iso-8601=seconds)"
