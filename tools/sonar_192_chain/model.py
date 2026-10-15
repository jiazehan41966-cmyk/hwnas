"""Load the frozen 192 v2 model and explicitly selected EMA inference weights."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

import torch

from .contract import (FROZEN, ROOT, MODEL_DEPENDENCIES, EXECUTION_MANIFEST_SHA256,
                       digest, model_contract, sha256_file, verify_frozen_sources,
                       require_registered_final_ema)

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@lru_cache(maxsize=1)
def _adapter():
    verify_frozen_sources()
    spec = importlib.util.spec_from_file_location("sonar_192_chain_frozen_model_v2", FROZEN / "package/model_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(architecture="9ee366", seed=42, *, source_root=None):
    expected = model_contract(architecture)
    model = _adapter().build_model(expected["architecture"], seed, source_root=source_root)
    if model.architecture_contract()["full_contract_sha256"] != expected["full_contract_sha256"]:
        raise ValueError("Constructed model differs from frozen complete contract")
    return model


def state_hash(state):
    # The established PTQ hasher is the shared model-state byte contract.
    from tools.modnas_comparison.quantization import model_state_hash
    class State:
        def state_dict(self):
            return state
    return model_state_hash(State())


def inspect_checkpoint(path, *, architecture, seed, purpose="diagnostic",
                       execution_manifest=None, training_result=None):
    """Check actual tensor bytes and EMA metadata; never load an unrestricted pickle.

    Formal admission additionally needs source-bound execution/result artifacts.
    It still does not certify quantization accuracy or hardware deployment.
    """
    if purpose not in ("diagnostic", "formal"):
        raise ValueError("Unknown checkpoint purpose")
    spec = model_contract(architecture)
    path = Path(path).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    binding = state.get("binding", {})
    epoch = state.get("epoch")
    required = dict(architecture_id=spec["structure_id"], seed=seed,
                    head="quality_direct6_CIoU", recipe="C01_plus_EMA",
                    validation_weights="ema", test_evaluated=False)
    if any(binding.get(k) != v for k, v in required.items()):
        raise ValueError("Checkpoint training/head/EMA/architecture binding mismatch")
    if type(epoch) is not int or not 1 <= epoch <= 80 or state.get("test_evaluated") is not False:
        raise ValueError("Checkpoint epoch or sealed-test metadata invalid")
    if binding.get("mode") not in ("smoke", "formal"):
        raise ValueError("Control/unknown checkpoint cannot become a 192 v2 target")
    if state.get("weights") == "ema" and state.get("inference_only") is True:
        tensors = state.get("model_state_dict")
        updates = state.get("ema_updates")
        origin = "ema_final.pt/model_state_dict"
    elif state.get("validation_weights") == "ema" and isinstance(state.get("ema_state_dict"), dict):
        ema = state["ema_state_dict"]
        if ema.get("decay") != .9999 or ema.get("tau") != 2000:
            raise ValueError("EMA recipe changed")
        tensors = ema.get("model_state_dict")
        updates = ema.get("updates")
        origin = "checkpoint/ema_state_dict/model_state_dict"
    else:
        raise ValueError("Explicit EMA payload required; raw model_state_dict is not a fallback")
    if not isinstance(tensors, dict) or not tensors or any(not isinstance(v, torch.Tensor) for v in tensors.values()):
        raise ValueError("Tensor-only EMA state required")
    if any(not torch.isfinite(v).all().item() for v in tensors.values()):
        raise ValueError("Checkpoint contains nonfinite tensors")
    if type(updates) is not int or updates != epoch * 83:
        raise ValueError("EMA updates disagree with frozen 330-image batch4 complete epochs")
    formal = False
    evidence = {}
    if purpose == "formal":
        if epoch != 80 or binding["mode"] != "formal":
            raise ValueError("Formal candidate selection requires final epoch80 EMA")
        if execution_manifest is None or training_result is None:
            raise ValueError("Formal checkpoint needs execution manifest and completed training result")
        manifest_path, result_path = Path(execution_manifest), Path(training_result)
        if sha256_file(manifest_path) != binding.get("execution_manifest_sha256"):
            raise ValueError("Checkpoint execution manifest hash mismatch")
        # The first bridge admits only the already frozen Stage B execution.
        # A different worker/recipe/data release needs its own registered version,
        # not merely a freshly self-signed manifest with the same head label.
        if sha256_file(manifest_path) != EXECUTION_MANIFEST_SHA256:
            raise ValueError("Unregistered training execution; worker/recipe/data must be independently frozen")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        # Stage B's result has no headline status; its actual schema uses epochs,
        # formal_eligible, history and file hashes. Do not invent a COMPLETE key.
        if result.get("formal_eligible") is not True or result.get("binding") != binding:
            raise ValueError("Completed training result does not match selected EMA run")
        if result.get("epochs") != 80 or len(result.get("history", [])) != 80:
            raise ValueError("Formal result does not certify 80 complete epochs")
        if [row.get("epoch") for row in result["history"]] != list(range(1, 81)):
            raise ValueError("Formal epoch history is incomplete or duplicated")
        if result.get("validation_weights") != "ema":
            raise ValueError("Formal metrics were not evaluated with EMA")
        if result.get("hashes", {}).get(path.name) != sha256_file(path):
            raise ValueError("Formal training result does not bind the selected checkpoint bytes")
        cfg = manifest.get("config", {})
        if spec["structure_id"] not in cfg.get("architecture_ids", []) or seed not in cfg.get("seeds", []):
            raise ValueError("Formal candidate/seed not registered in this training execution")
        if (cfg.get("space_config", {}).get("sha256") != spec["source_sha256"]["sonar_backbone_neck192_v2.yaml"]
                or cfg.get("space_list", {}).get("sha256") != spec["source_sha256"]["space192_v2_list.csv"]
                or cfg.get("training", {}).get("validation_weights") != "ema"):
            raise ValueError("Training execution manifest belongs to another model or recipe")
        sources = manifest.get("source_sha256", {})
        if any(sources.get(name) != spec["source_sha256"][name] for name in MODEL_DEPENDENCIES):
            raise ValueError("Training source closure differs from the selected frozen model")
        model_path = "results/space192_v2_20261013/stage_B/package/model_v2.py"
        if sources.get(model_path) != spec["source_sha256"]["package/model_v2.py"]:
            raise ValueError("Training source does not bind the frozen 192 v2 model")
        if result.get("test_evaluated") is not False:
            raise ValueError("Formal result must keep test sealed")
        if not manifest or not result:
            raise ValueError("Empty training evidence")
        evidence = dict(execution_manifest_sha256=sha256_file(manifest_path),
                        training_result_sha256=sha256_file(result_path))
        formal = True
    record = dict(path=str(path), sha256=sha256_file(path), model_state_sha256=state_hash(tensors),
                  full_contract_sha256=spec["full_contract_sha256"],
                  architecture_sha256=spec["architecture_sha256"], seed=seed, epoch=epoch,
                  mode=binding["mode"], weights="ema", validation_weights="ema", ema_updates=updates,
                  payload_selection=origin, selection_policy="epoch80_ema",
                  binding=deepcopy_binding(binding), formal_eligible=formal, evidence=evidence,
                  stable_mean_is_checkpoint_metric=False, test_evaluated=False)
    if formal:
        require_registered_final_ema(record, spec["full_contract_sha256"])
    return record, tensors


def deepcopy_binding(binding):
    return json.loads(json.dumps(binding, allow_nan=False))


def load_checkpoint(path, *, architecture, seed, purpose="diagnostic", source_root=None,
                    execution_manifest=None, training_result=None):
    record, tensors = inspect_checkpoint(path, architecture=architecture, seed=seed, purpose=purpose,
                                         execution_manifest=execution_manifest, training_result=training_result)
    model = build_model(architecture, seed, source_root=source_root).eval()
    model.load_state_dict(tensors, strict=True)
    if state_hash(model.state_dict()) != record["model_state_sha256"]:
        raise ValueError("EMA tensor hash changed during model loading")
    model.chain_checkpoint_binding = record
    return model, record
