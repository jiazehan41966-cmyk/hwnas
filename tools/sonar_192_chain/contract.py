"""One immutable identity across selection, quantization, cost and deployment.

The frozen 192 v2 inventory remains authoritative. Structure IDs alone omit the
new head. A transport identity additionally binds EMA bytes, precision,
calibration and the concrete mapping. No evidence eligibility is inferred from
a headline PASS or from the presence of a hash.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results/space192_v2_20261013/stage_B"
SPACE_VERSION = "sonar_backbone_neck192_v2"
SCHEMA = "sonar_192_chain_identity_v1"
PROFILES = ("W8A8", "HEAD_W8A16", "HEAD_W16A16")
FROZEN_HASHES = {
    "sonar_backbone_neck192_v2.yaml": "e301dba810e7bb0bbb8d98fe2f8d527962c2697e9efae510eced8a5b51997c0e",
    "space192_v2_list.csv": "60a8edafb21ec7b21f4076073ef049bc912b8e044fac036a9e1f5c94a0f528c0",
    "package/model_v2.py": "c535593898e2e19a980d551e227a6e2d2f0d9bdf44defaf2ea6ab40604235fe4",
}
EXECUTION_MANIFEST = FROZEN / "collected_terminal_20261013T142134Z_9cbac346/snapshot/remote/execution_manifest.json"
EXECUTION_MANIFEST_SHA256 = "4993beb2076511f8140b7e45397914142c32234064170c7ca3b75c186cb62dc5"
MODEL_DEPENDENCIES = (
    "tools/sonar_joint192_validation/model.py", "tools/sonar_head_factorial/model.py",
    "tools/sonar_head_c01b/model.py", "tools/detector_elastic_pilot/model.py",
    "tools/sonar_yolov9_baseline/author_adapter.py", "src/hwnas_fpga/models/builder.py",
    "src/hwnas_fpga/search_space/__init__.py", "src/hwnas_fpga/search_space/space.py",
    "reference/yolov9/models/yolo.py",
    "reference/yolov9/models/common.py", "reference/yolov9/utils/loss_tal.py",
    "reference/yolov9/utils/tal/anchor_generator.py",
)
CHECKPOINT_POLICY = {
    "version": "epoch80_ema_v1", "weights": "ema", "epoch": 80,
    "rule": "fixed final epoch 80 EMA; no post-hoc best-epoch selection",
    "checkpoint_metric": "recomputed_full_valid_for_this_checkpoint",
    "stable_metric": "epochs76_80_mean_is_separate_not_checkpoint_performance",
    "test_policy": "sealed",
}
# A completed training artifact is a trust anchor, not a caller-provided flag.
# No registered epoch80 EMA was present in the audited Stage B collection.
# Populate only in a separately reviewed source revision after checking the
# actual training run and all three artifacts. CLI input cannot extend this.
REGISTERED_FINAL_EMA_ARTIFACTS = ()


def require_registered_final_ema(checkpoint, full_contract_sha256):
    evidence = checkpoint.get("evidence", {})
    key = dict(full_contract_sha256=full_contract_sha256,
               seed=checkpoint.get("seed"), epoch=checkpoint.get("epoch"),
               checkpoint_sha256=checkpoint.get("sha256"),
               model_state_sha256=checkpoint.get("model_state_sha256"),
               execution_manifest_sha256=evidence.get("execution_manifest_sha256"),
               training_result_sha256=evidence.get("training_result_sha256"))
    if not any(all(row.get(k) == v for k, v in key.items())
               for row in REGISTERED_FINAL_EMA_ARTIFACTS):
        raise ValueError("Unregistered final EMA artifact: independently reviewed checkpoint/result/manifest hashes required")
    return deepcopy(key)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def require_hash(value, label="sha256"):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def verify_frozen_sources():
    for name, expected in FROZEN_HASHES.items():
        if sha256_file(FROZEN / name) != expected:
            raise ValueError(f"Frozen 192 v2 source changed: {name}")
    if sha256_file(EXECUTION_MANIFEST) != EXECUTION_MANIFEST_SHA256:
        raise ValueError("Frozen execution source manifest changed")
    sources = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))["source_sha256"]
    bound = dict(FROZEN_HASHES)
    for name in MODEL_DEPENDENCIES:
        if name not in sources or sha256_file(ROOT / name) != sources[name]:
            raise ValueError(f"Frozen model dependency changed: {name}")
        bound[name] = sources[name]
    return bound


def model_contract(architecture="9ee366"):
    source_hashes = verify_frozen_sources()
    with (FROZEN / "space192_v2_list.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 192 or len({r["architecture_hash"] for r in rows}) != 192:
        raise ValueError("Frozen 192 inventory incomplete or duplicated")
    if isinstance(architecture, dict):
        selected = [r for r in rows if json.loads(r["architecture_json"]) == architecture]
    else:
        selected = [r for r in rows if architecture in
                    (r["structure_id"], r["architecture_hash"], r["architecture_id"])]
    if len(selected) != 1:
        raise ValueError("An exact registered 192 v2 architecture is required")
    row = selected[0]
    cfg = yaml.safe_load((FROZEN / "sonar_backbone_neck192_v2.yaml").read_text(encoding="utf-8"))
    arch = json.loads(row["architecture_json"])
    full = dict(search_space_version=SPACE_VERSION, architecture=arch,
                head=cfg["fixed_head"], input_chw=[1, 640, 640],
                trunk_initialization_policy="JOINT192_PILOT_SAME_SHAPE_PAIRED_DW_SEMANTIC_V1",
                shared_quality_head_initialization_policy="FACTORIAL_NATIVE_TRUNK_PAIRED_CPU_SEED_V1",
                direct_prediction_initialization_policy="C01B_DIRECT_DISTANCE_FINAL_PREDICTION_CPU_SEED_V1")
    if digest(full) != row["full_contract_sha256"]:
        raise ValueError("192 v2 full head/model contract differs from frozen inventory")
    return dict(space_version=SPACE_VERSION, architecture=arch,
                architecture_id=row["architecture_id"], structure_id=row["structure_id"],
                architecture_sha256=row["architecture_hash"],
                full_contract_sha256=row["full_contract_sha256"],
                head_contract=deepcopy(cfg["fixed_head"]),
                head_contract_sha256=digest(cfg["fixed_head"]), input_chw=[1, 640, 640],
                source_sha256=source_hashes)


def validate_mapping(mapping, *, formal=False):
    if not isinstance(mapping, dict):
        raise ValueError("Concrete mapping must be a dictionary, not a policy name")
    required = ("implementation_id", "PE", "SIMD", "buffer_profile", "numeric_contract_sha256",
                "interface_contract_sha256", "status")
    if any(k not in mapping for k in required):
        raise ValueError("Concrete mapping missing implementation/parallelism/buffer/numeric/interface identity")
    for k in ("PE", "SIMD"):
        if type(mapping[k]) is not int or mapping[k] <= 0:
            raise ValueError(f"Positive integer {k} required")
    if not isinstance(mapping["implementation_id"], str) or not mapping["implementation_id"].strip():
        raise ValueError("Named implementation required")
    if not mapping["buffer_profile"]:
        raise ValueError("Explicit buffer/tiling profile required")
    for k in ("numeric_contract_sha256", "interface_contract_sha256"):
        require_hash(mapping[k], k)
    if mapping["status"] not in ("PROPOSED", "FROZEN_CONCRETE_MAPPING"):
        raise ValueError("Mapping status is not registered")
    if formal and mapping["status"] != "FROZEN_CONCRETE_MAPPING":
        raise ValueError("Formal identity requires a frozen concrete mapping")
    return deepcopy(mapping)


def make_identity(architecture="9ee366", *, checkpoint, profile, mapping, calibration,
                  purpose="diagnostic"):
    if purpose not in ("diagnostic", "formal") or profile not in PROFILES:
        raise ValueError("Unregistered purpose or precision profile")
    identity = dict(schema_version=SCHEMA, **model_contract(architecture),
                    checkpoint=deepcopy(checkpoint), profile=profile,
                    mapping=deepcopy(mapping), calibration=deepcopy(calibration),
                    checkpoint_selection=deepcopy(CHECKPOINT_POLICY), purpose=purpose)
    identity["identity_sha256"] = digest(identity)
    return validate_identity(identity)


def validate_identity(identity):
    if not isinstance(identity, dict) or identity.get("schema_version") != SCHEMA:
        raise ValueError("Versioned 192 v2 candidate identity required")
    actual = deepcopy(identity)
    checksum = actual.pop("identity_sha256", None)
    if checksum != digest(actual):
        raise ValueError("Candidate identity hash mismatch")
    expected = model_contract(actual.get("architecture"))
    if any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("Candidate model/head/source identity mismatch")
    if actual.get("purpose") not in ("diagnostic", "formal") or actual.get("profile") not in PROFILES:
        raise ValueError("Unregistered purpose or precision profile")
    if actual.get("checkpoint_selection") != CHECKPOINT_POLICY:
        raise ValueError("Final EMA selection policy drift")
    formal = actual["purpose"] == "formal"
    validate_mapping(actual.get("mapping"), formal=formal)
    cp = actual.get("checkpoint", {})
    for k in ("sha256", "model_state_sha256"):
        require_hash(cp.get(k), "checkpoint." + k)
    if cp.get("weights") != "ema" or cp.get("validation_weights") != "ema":
        raise ValueError("192 v2 must bind EMA inference weights, never raw training weights")
    if cp.get("seed") not in (42, 43, 44) or type(cp.get("epoch")) is not int or not 1 <= cp["epoch"] <= 80:
        raise ValueError("Checkpoint seed/epoch is outside the frozen training contract")
    if cp.get("full_contract_sha256") != expected["full_contract_sha256"]:
        raise ValueError("Checkpoint is bound to a different model/head contract")
    if cp.get("selection_policy") != "epoch80_ema":
        raise ValueError("Explicit epoch80 EMA target policy required")
    if formal and (cp["epoch"] != 80 or cp.get("mode") != "formal" or cp.get("formal_eligible") is not True):
        raise ValueError("Formal candidate requires verified complete epoch80 EMA evidence")
    if formal:
        require_registered_final_ema(cp, expected["full_contract_sha256"])
    cal = actual.get("calibration", {})
    require_hash(cal.get("sha256"), "calibration.sha256")
    if cal.get("split") != "train" or cal.get("test_accessed") is not False:
        raise ValueError("Calibration must bind train-only data with sealed test")
    if type(cal.get("number_of_calibration_images")) is not int or cal["number_of_calibration_images"] < 1:
        raise ValueError("Calibration count required")
    if formal and (cal["number_of_calibration_images"] != 64 or cal.get("source_verified") is not True):
        raise ValueError("Formal calibration requires source-verified frozen train64")
    return deepcopy(identity)


def assert_same_identity(expected, actual):
    validate_identity(expected)
    validate_identity(actual)
    if expected["identity_sha256"] != actual["identity_sha256"]:
        raise ValueError("Cross-stage candidate identity mismatch")


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
