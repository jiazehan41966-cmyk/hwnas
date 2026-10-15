"""Isolated, audited continuation of the existing seven-epoch AAA run.

The original campaign is read-only. This module never relaxes train_one's
source/config/checkpoint bindings; it explicitly clones and rebinds verified
state before the normal campaign controller can resume it.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from .common import ROOT, read_json, sha256_file, utc_now, write_json

PARENT = "/home/jiajia/experiments/sonar_fcos_pilot_20261008"
RECOVERY = PARENT + "_recovery_v1"
PYTHON = "/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python"
OUTPUT = "results/sonar_fcos_pilot_20261008"
CONFIG = "configs/sonar_fcos_pilot_recovery_v1.json"
EVIDENCE = OUTPUT + "/recovery"
LOCAL = ROOT / OUTPUT / "recovery_transport"
ORIGINAL_ARCHIVE = ROOT / OUTPUT / "recovery_original_snapshot/source_89e2001eff4e8aa2.tar.gz"
ORIGINAL_ARCHIVE_SHA = "644d6dfe2f1505bead8b948e90c357ac9973c1d0c735a36a88fe84df8aad6275"
CHANGED_SOURCE = {"tools/sonar_fcos_pilot/train.py"}
NEW_SOURCE = {"tools/sonar_fcos_pilot/numerics.py", "tools/sonar_fcos_pilot/recovery.py"}
PATH_CHANGES = {"output_dir", "checkpoint_root"}
REVIEW_KEYS = ["source_freeze_sha256", "environment_lock_sha256", "preflight_sha256",
               "initialization_audit_sha256", "numerics_witness_sha256", "migration_sha256", "check_sha256"]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_source_delta(original, current):
    """Only the overflow repair and its isolated transport may change Python."""
    for relative, digest in original.items():
        if relative not in current:
            raise RuntimeError("A frozen original source was removed: " + relative)
        if current[relative] != digest and relative not in CHANGED_SOURCE and not relative.endswith(".md"):
            raise RuntimeError("Unrelated frozen source changed: " + relative)
    for relative in set(current) - set(original):
        if relative not in NEW_SOURCE and not relative.endswith(".md"):
            raise RuntimeError("Unexpected new executable source: " + relative)
    if not NEW_SOURCE.issubset(current):
        raise RuntimeError("The audited numerical repair and recovery module are required")


def verify_config_delta(original, new):
    differences = {key for key in set(original) | set(new) if original.get(key) != new.get(key)}
    if differences != PATH_CHANGES | {"recovery_parent"}:
        raise RuntimeError("Only isolated output/checkpoint paths may change: " + repr(sorted(differences)))
    provenance = new["recovery_parent"]
    if set(provenance) != {"root", "source_freeze_sha256", "config_sha256", "last_checkpoint_sha256", "completed_epochs", "resumed_run_id", "new_candidate_run"}:
        raise RuntimeError("Unexpected recovery provenance fields")
    if provenance["root"] != PARENT or provenance["completed_epochs"] != 7 or provenance["resumed_run_id"] != "AAA_seed42" or provenance["new_candidate_run"] is not False:
        raise RuntimeError("Invalid existing-run recovery provenance")
    if new["max_formal_runs"] != 14 or new["epochs"] != 30:
        raise RuntimeError("The existing approved training budget changed")
    return sorted(differences)


def state_digest(value):
    """Content fingerprint of tensor/array/nested training state, without pickle."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor"); digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"array"); digest.update(str(item.dtype).encode())
            digest.update(repr(item.shape).encode()); digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr): visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            digest.update(str(len(item)).encode())
            for child in item: visit(child)
        elif isinstance(item, (str, bytes, int, float, bool, type(None), np.generic)):
            digest.update(type(item).__name__.encode()); digest.update(repr(item).encode()); digest.update(b"\0")
        else:
            raise TypeError("Unsupported training-state value: " + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def _archive_marker():
    if sha256_file(ORIGINAL_ARCHIVE) != ORIGINAL_ARCHIVE_SHA:
        raise RuntimeError("Preserved original source archive changed")
    with tarfile.open(ORIGINAL_ARCHIVE) as archive:
        marker = json.loads(archive.extractfile("source_freeze.json").read())
        for relative, digest in marker["source_sha256"].items():
            if hashlib.sha256(archive.extractfile(relative).read()).hexdigest() != digest:
                raise RuntimeError("Original archive source content changed: " + relative)
    return marker


def stage():
    """Freeze repaired code in a new directory, never restage the original."""
    from .remote import _copy, _execute
    original = _archive_marker()
    paths = sorted(path for path in (ROOT / "tools/sonar_fcos_pilot").iterdir()
                   if path.is_file() and path.suffix in {".py", ".md"})
    paths.append(ROOT / "configs/sonar_fcos_pilot_v1.json")
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("Source symlinks are not accepted")
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    verify_source_delta(original["source_sha256"], hashes)
    marker = {"schema_version": 1, "timestamp_utc": utc_now(), "parent_root": PARENT,
              "recovery_root": RECOVERY, "original_archive_sha256": ORIGINAL_ARCHIVE_SHA,
              "parent_source_sha256": original["source_sha256"],
              "parent_source_set_sha256": original["source_set_sha256"],
              "source_sha256": hashes, "source_set_sha256": canonical_sha(hashes),
              "scientific_parameters_changed": False, "data_copied": False}
    # Stable when the reviewer replays the exact stage command.
    local_marker = LOCAL / "source_freeze.json"
    if local_marker.exists():
        previous = read_json(local_marker)
        if previous["source_sha256"] == hashes:
            marker = previous
    write_json(local_marker, marker)
    archive_path = LOCAL / ("recovery_source_" + marker["source_set_sha256"][:16] + ".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in paths: archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        archive.add(local_marker, arcname="source_freeze.json", recursive=False)
    code = "from pathlib import Path\nr=Path(" + repr(RECOVERY) + ")\nassert r.parent==Path('/home/jiajia/experiments')\n(r/'staging').mkdir(parents=True,exist_ok=True)\n"
    _execute(code)
    remote_archive = RECOVERY + "/staging/" + archive_path.name
    _copy(archive_path, remote_archive)
    code = "from pathlib import Path\nimport json,hashlib,tarfile\nr=Path(" + repr(RECOVERY) + ")\np=Path(" + repr(PARENT) + ")\na=Path(" + repr(remote_archive) + ")\nexpected=" + repr(sha256_file(archive_path)) + "\n" + r'''
assert hashlib.sha256(a.read_bytes()).hexdigest()==expected
with tarfile.open(a) as stream:
 payload={}
 for member in stream.getmembers():
  relative=Path(member.name); target=(r/relative).resolve()
  if not member.isfile() or relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(r): raise RuntimeError('Unsafe source archive member')
  if member.name in payload: raise RuntimeError('Duplicate archive member')
  payload[member.name]=stream.extractfile(member).read()
marker=json.loads(payload['source_freeze.json'])
if set(payload)!=set(marker['source_sha256'])|{'source_freeze.json'}: raise RuntimeError('Source archive member mismatch')
parent=json.loads((p/'source_freeze.json').read_text())
if parent['source_sha256']!=marker['parent_source_sha256']: raise RuntimeError('Original remote snapshot differs from archived source')
for relative,digest in parent['source_sha256'].items():
 if hashlib.sha256((p/relative).read_bytes()).hexdigest()!=digest: raise RuntimeError('Original remote source changed: '+relative)
for relative,digest in marker['source_sha256'].items():
 if hashlib.sha256(payload[relative]).hexdigest()!=digest: raise RuntimeError('Recovery archive hash mismatch')
existing=r/'source_freeze.json'
if existing.exists():
 old=json.loads(existing.read_text())
 if old!=marker: raise RuntimeError('Isolated recovery source already frozen differently; use another recovery version')
 for relative,digest in marker['source_sha256'].items():
  if hashlib.sha256((r/relative).read_bytes()).hexdigest()!=digest: raise RuntimeError('Existing frozen recovery source changed')
else:
 for relative,data in payload.items():
  target=r/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
vendor=r/'vendor'
if not vendor.exists(): vendor.symlink_to(p/'vendor',target_is_directory=True)
if vendor.resolve()!=(p/'vendor').resolve(): raise RuntimeError('Unexpected private dependency overlay')
print(json.dumps({'status':'RECOVERY_SOURCE_STAGE_PASS','recovery_root':str(r),'parent_preserved':True,'source_freeze_sha256':hashlib.sha256((r/'source_freeze.json').read_bytes()).hexdigest(),'source_set_sha256':marker['source_set_sha256'],'private_overlay_reused_read_only':True,'formal_training_started':False}))
'''
    value = json.loads(_execute(code, 120))
    write_json(LOCAL / "stage.json", value)
    return value


def _remote_call(mode, timeout=600):
    from .remote import _execute
    code = "from pathlib import Path\nimport os,sys,json\nr=Path(" + repr(RECOVERY) + ")\nos.chdir(r)\nos.environ['CUDA_VISIBLE_DEVICES']='1'\nos.environ['OMP_NUM_THREADS']='4'\nos.environ['MKL_NUM_THREADS']='4'\nos.environ['OPENBLAS_NUM_THREADS']='4'\nsys.path.insert(0,str(r))\nsys.path.insert(0,str(r/'vendor'))\nfrom tools.sonar_fcos_pilot.recovery import remote_" + mode + "\nprint(json.dumps(remote_" + mode + "(),ensure_ascii=False))\n"
    value = json.loads(_execute(code, timeout))
    write_json(LOCAL / (mode + ".json"), value)
    return value


def _layout():
    root, parent = Path(RECOVERY), Path(PARENT)
    if ROOT.resolve() != root.resolve():
        raise RuntimeError("This operation must run inside the isolated remote recovery root")
    result = root / EVIDENCE
    result.mkdir(parents=True, exist_ok=True)
    return root, parent, result


def _verify_sources(root, parent):
    marker = read_json(root / "source_freeze.json")
    parent_marker = read_json(parent / "source_freeze.json")
    if parent_marker["source_sha256"] != marker["parent_source_sha256"]:
        raise RuntimeError("Parent source marker changed")
    for directory, sources in [(parent, marker["parent_source_sha256"]), (root, marker["source_sha256"])]:
        for relative, digest in sources.items():
            if sha256_file(directory / relative) != digest:
                raise RuntimeError("Frozen source changed: " + str(directory / relative))
    verify_source_delta(marker["parent_source_sha256"], marker["source_sha256"])
    if canonical_sha(marker["source_sha256"]) != marker["source_set_sha256"]:
        raise RuntimeError("Recovery source set hash changed")
    return marker


def _assert_no_controller(root):
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid(): continue
        try:
            if (process / "cwd").resolve() == root.resolve():
                command = (process / "cmdline").read_bytes()
                if b"tools.sonar_fcos_pilot.run" in command:
                    raise RuntimeError("An existing campaign controller is active")
        except (PermissionError, FileNotFoundError, ProcessLookupError): pass


def remote_migrate():
    """Clone committed state, preserving the actual optimizer/RNG trajectory."""
    import torch
    from .common import environment, load_config, require_manifest, source_binding
    root, parent, evidence = _layout()
    marker = _verify_sources(root, parent)
    _assert_no_controller(parent); _assert_no_controller(root)
    parent_config_path = parent / "configs/sonar_fcos_pilot_remote_v1.json"
    old_config_raw = read_json(parent_config_path)
    old_config = load_config(parent_config_path)
    manifest_path, manifest = require_manifest(old_config)
    parent_runs = list((Path(old_config["output_dir"]) / "runs").glob("*/run_manifest.json"))
    if [path.parent.name for path in parent_runs] != ["AAA_seed42"]:
        raise RuntimeError("Expected exactly the original AAA_seed42 run within the existing budget")
    source_run = parent_runs[0].parent
    source_receipt = read_json(source_run / "run_manifest.json")
    if read_json(source_run / "status.json")["status"] != "ERROR":
        raise RuntimeError("Only the explicitly interrupted original run may be migrated")
    source_cp = Path(old_config["checkpoint_root"]) / "AAA_seed42"
    last = torch.load(source_cp / "last.pt", map_location="cpu", weights_only=False)
    old_binding = source_receipt["binding"]
    if last["binding"] != old_binding or last["epoch"] != 6 or len(last["history"]) != 7:
        raise RuntimeError("The committed original last.pt is not the matching seven-epoch run")
    if last["history"] != read_json(source_run / "history.json"):
        raise RuntimeError("Original committed history differs from last.pt")
    expected_old_binding = {"config_sha256": sha256_file(parent_config_path),
                            "sources": {str(Path(relative)): digest for relative, digest in marker["parent_source_sha256"].items() if relative.endswith(".py")},
                            "data_manifest_sha256": sha256_file(manifest_path), "combo": "AAA", "seed": 42,
                            "environment": environment()}
    if old_binding != expected_old_binding:
        raise RuntimeError("Original config/source/data/environment receipt does not match live evidence")
    for artifact in last["best_artifacts"].values():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("Original committed best artifact changed")
    for sample in manifest["samples"]:
        if sample["split"] in {"train", "val", "test"}:
            for field, digest_field in [("image_path", "sha256"), ("annotation_path", "annotation_sha256")]:
                if sha256_file(sample[field]) != sample[digest_field]: raise RuntimeError("Original raw input changed")
    official = Path(old_config["weight_cache"]) / "checkpoints/fcos_resnet50_fpn_coco-99b0c9b7.pth"
    from .remote import WEIGHT_SHA256
    if sha256_file(official) != WEIGHT_SHA256: raise RuntimeError("Official initialization checkpoint changed")
    new_config_raw = copy.deepcopy(old_config_raw)
    new_config_raw["output_dir"] = str(root / OUTPUT)
    new_config_raw["checkpoint_root"] = str(root / "checkpoints")
    new_config_raw["recovery_parent"] = {"root": str(parent),
        "source_freeze_sha256": sha256_file(parent / "source_freeze.json"),
        "config_sha256": sha256_file(parent_config_path),
        "last_checkpoint_sha256": sha256_file(source_cp / "last.pt"),
        "completed_epochs": 7, "resumed_run_id": "AAA_seed42", "new_candidate_run": False}
    verify_config_delta(old_config_raw, new_config_raw)
    config_path = root / CONFIG
    report_path = evidence / "migration.json"
    if report_path.exists():
        report = read_json(report_path)
        if read_json(config_path) != new_config_raw: raise RuntimeError("Existing recovery config changed")
        _verify_migration_files(report)
        return report
    if config_path.exists() or (root / "checkpoints").exists():
        raise RuntimeError("Uncommitted recovery migration artifacts exist; inspect them before retrying")
    write_json(config_path, new_config_raw)
    new_config = load_config(config_path)
    binding = {**source_binding(new_config), "data_manifest_sha256": sha256_file(manifest_path),
               "combo": "AAA", "seed": 42, "environment": environment()}
    new_run = root / OUTPUT / "runs/AAA_seed42"
    new_cp = root / "checkpoints/AAA_seed42"
    new_run.mkdir(parents=True, exist_ok=True); new_cp.mkdir(parents=True, exist_ok=True)
    transports = []
    def record(source, target, kind):
        transports.append({"kind": kind, "parent_path": str(source), "parent_sha256": sha256_file(source),
                           "recovery_path": str(target), "recovery_sha256": sha256_file(target)})
    # Clone all committed best snapshots and validation artifacts, not just last.pt.
    for source in sorted(source_cp.glob("best_epoch_*.pt")):
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if saved["binding"] != old_binding or saved["epoch"] > last["epoch"]:
            raise RuntimeError("An original best checkpoint has invalid binding/epoch")
        saved["binding"] = binding
        target = new_cp / source.name
        torch.save(saved, target); record(source, target, "best_checkpoint_rebound")
    for source in sorted((source_run / "validation").glob("*.json")):
        target = new_run / "validation" / source.name
        target.parent.mkdir(exist_ok=True); shutil.copyfile(source, target); record(source, target, "validation_copy")
    best_artifacts = copy.deepcopy(last["best_artifacts"])
    for artifact in best_artifacts.values():
        old_path = Path(artifact["path"])
        new_path = new_cp / old_path.name if old_path.parent == source_cp else new_run / "validation" / old_path.name
        if not new_path.is_file(): raise RuntimeError("Committed artifact was not copied")
        artifact.update(path=str(new_path), sha256=sha256_file(new_path))
    retained_keys = ["model", "optimizer", "scaler", "rng", "history", "best_map", "epoch"]
    retained_digest = state_digest({key: last[key] for key in retained_keys})
    last["binding"] = binding; last["best_artifacts"] = best_artifacts
    torch.save(last, new_cp / "last.pt"); record(source_cp / "last.pt", new_cp / "last.pt", "resume_checkpoint_rebound")
    cloned = torch.load(new_cp / "last.pt", map_location="cpu", weights_only=False)
    if state_digest({key: cloned[key] for key in retained_keys}) != retained_digest:
        raise RuntimeError("Model/optimizer/scaler/RNG/history changed during recovery cloning")
    receipt = copy.deepcopy(source_receipt); receipt["binding"] = binding
    receipt["recovery"] = {"status": "RESUMED_EXISTING_RUN", "parent_run_id": "AAA_seed42",
                           "committed_epochs": 7, "resume_at_epoch": 8, "new_candidate_run": False,
                           "parent_receipt_sha256": sha256_file(source_run / "run_manifest.json")}
    write_json(new_run / "run_manifest.json", receipt); record(source_run / "run_manifest.json", new_run / "run_manifest.json", "run_receipt_rebound")
    shutil.copyfile(source_run / "history.json", new_run / "history.json"); record(source_run / "history.json", new_run / "history.json", "history_copy")
    write_json(new_run / "status.json", {"status": "RECOVERY_READY", "completed_epochs": 7, "resume_at_epoch": 8,
                                        "formal_training_started": False, "updated_at": utc_now()})
    report = {"status": "MIGRATION_PASS", "created_at": utc_now(), "parent_root": str(parent), "recovery_root": str(root),
              "source_freeze_sha256": sha256_file(root / "source_freeze.json"), "parent_source_freeze_sha256": sha256_file(parent / "source_freeze.json"),
              "parent_config_path": str(parent_config_path), "parent_config_sha256": sha256_file(parent_config_path),
              "recovery_config_path": str(config_path), "recovery_config_sha256": sha256_file(config_path),
              "manifest_path": str(manifest_path), "data_manifest_sha256": sha256_file(manifest_path),
              "acknowledgment_path": old_config["approved_protocol"]["acknowledgment_path"],
              "acknowledgment_sha256": old_config["approved_protocol"]["acknowledgment_sha256"],
              "official_weight_sha256": WEIGHT_SHA256, "config_changed_fields": sorted(PATH_CHANGES | {"recovery_parent"}),
              "retained_state_sha256": retained_digest, "retained_state_keys": retained_keys,
              "binding": binding, "parent_binding": old_binding, "artifacts": transports,
              "committed_epochs": 7, "resume_at_epoch": 8, "parent_run_id": "AAA_seed42",
              "new_candidate_run": False, "max_formal_runs": 14, "formal_training_started": False}
    write_json(report_path, report)
    return report


def _verify_migration_files(report):
    for field in ["parent_config", "recovery_config", "data_manifest", "acknowledgment"]:
        path_field = "manifest_path" if field == "data_manifest" else field + "_path"
        if sha256_file(report[path_field]) != report[field + "_sha256"]:
            raise RuntimeError("Migration evidence changed: " + field)
    for artifact in report["artifacts"]:
        for prefix in ["parent", "recovery"]:
            if sha256_file(artifact[prefix + "_path"]) != artifact[prefix + "_sha256"]:
                raise RuntimeError("Migration artifact changed: " + artifact[prefix + "_path"])


def _gpu_idle():
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    row = next([int(v.strip()) for v in line.split(",")] for line in lines.splitlines() if int(line.split(",")[0]) == 1)
    if row[1] > 1024 or row[2] - row[1] < 4096 or row[3] > 10:
        raise RuntimeError("Physical GPU 1 is occupied; no other process is modified")
    return {"physical_gpu": 1, "used_mib": row[1], "total_mib": row[2], "utilization_percent": row[3]}


def remote_preflight():
    """Actual CUDA kernels and the overflow retry witness; no formal epoch."""
    from .common import environment
    root, parent, evidence = _layout()
    _verify_sources(root, parent)
    migration = read_json(evidence / "migration.json"); _verify_migration_files(migration)
    _assert_no_controller(root); before = _gpu_idle()
    import numpy as np
    import pycocotools
    from pycocotools import mask
    sample = np.zeros((5, 5), dtype=np.uint8, order="F"); sample[1:4, 1:5] = 1
    area = float(mask.area(mask.encode(sample)))
    if area != 12: raise RuntimeError("Private pycocotools compiled-kernel witness failed")
    lock = {"status": "PASS", **environment(), "physical_gpu": 1,
            "private_overlay_path": str(Path(pycocotools.__file__).resolve()),
            "pycocotools_kernel_area": area, "shared_environment_modified": False,
            "source_freeze_sha256": sha256_file(root / "source_freeze.json")}
    if not Path(lock["private_overlay_path"]).is_relative_to(parent / "vendor"):
        raise RuntimeError("An unexpected metric dependency overlay was used")
    write_json(evidence / "environment_lock.json", lock)
    config = read_json(root / CONFIG)
    env = os.environ.copy(); env.update(PYTHONPATH=str(root / "vendor") + os.pathsep + str(root), CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1")
    model_output = evidence / "model_preflight"
    commands = [[PYTHON, "-m", "tools.sonar_fcos_pilot.model", "--output-dir", str(model_output), "--cache-dir", config["weight_cache"]],
                [PYTHON, "-m", "tools.sonar_fcos_pilot.numerics", "--output-dir", str(evidence / "numerics_witness"), "--device", "cuda:0"]]
    for index, command in enumerate(commands):
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        (evidence / ("preflight_" + str(index) + ".log")).write_text(completed.stdout + "\n" + completed.stderr)
        if completed.returncode: raise RuntimeError("Recovery CUDA witness failed; inspect preflight log " + str(index))
    model = read_json(model_output / "preflight.json"); initial = read_json(model_output / "initialization_audit.json")
    numeric_path = evidence / "numerics_witness/amp_recovery_witness.json"
    numeric = read_json(numeric_path)
    if model["status"] != "PASS" or initial["status"] != "PASS" or numeric["status"] != "PASS":
        raise RuntimeError("Recovery GPU/model/numerics witness rejected")
    if model["kernel_witness"]["device"] != "cuda" or not model["kernel_witness"]["finite"]:
        raise RuntimeError("Actual GPU dispatch was not demonstrated")
    binding = {"status": "PREFLIGHT_PASS", "physical_gpu": 1, "gpu_before_witness": before, "commands": commands,
               "source_freeze_sha256": sha256_file(root / "source_freeze.json"),
               "environment_lock_sha256": sha256_file(evidence / "environment_lock.json"),
               "preflight_sha256": sha256_file(model_output / "preflight.json"),
               "initialization_audit_sha256": sha256_file(model_output / "initialization_audit.json"),
               "numerics_witness_sha256": sha256_file(numeric_path),
               "migration_sha256": sha256_file(evidence / "migration.json"), "formal_training_started": False}
    write_json(evidence / "preflight_binding.json", binding)
    return binding


def remote_check():
    import torch
    from .common import load_config, require_manifest, source_binding
    root, parent, evidence = _layout()
    _verify_sources(root, parent)
    migration = read_json(evidence / "migration.json"); _verify_migration_files(migration)
    _assert_no_controller(root); _assert_no_controller(parent)
    config = load_config(root / CONFIG); path, manifest = require_manifest(config)
    verify_config_delta(read_json(migration["parent_config_path"]), read_json(root / CONFIG))
    saved = torch.load(root / "checkpoints/AAA_seed42/last.pt", map_location="cpu", weights_only=False)
    if saved["binding"] != migration["binding"] or saved["epoch"] != 6 or len(saved["history"]) != 7:
        raise RuntimeError("Recovery checkpoint is not the matching seven-epoch state")
    if source_binding(config) != {key: saved["binding"][key] for key in ["config_sha256", "sources"]}:
        raise RuntimeError("Resumed training would reject its source/config binding")
    if state_digest({key: saved[key] for key in migration["retained_state_keys"]}) != migration["retained_state_sha256"]:
        raise RuntimeError("Retained model/optimizer/scaler/RNG/history changed")
    for sample in manifest["samples"]:
        if sample["split"] in {"train", "val", "test"}:
            for field, digest_field in [("image_path", "sha256"), ("annotation_path", "annotation_sha256")]:
                if sha256_file(sample[field]) != sample[digest_field]: raise RuntimeError("Approved raw input changed")
    witness = read_json(evidence / "preflight_binding.json")
    if witness.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError("A passed recovery GPU/numerics witness is required")
    actual = {"source_freeze_sha256": sha256_file(root / "source_freeze.json"),
              "environment_lock_sha256": sha256_file(evidence / "environment_lock.json"),
              "preflight_sha256": sha256_file(evidence / "model_preflight/preflight.json"),
              "initialization_audit_sha256": sha256_file(evidence / "model_preflight/initialization_audit.json"),
              "numerics_witness_sha256": sha256_file(evidence / "numerics_witness/amp_recovery_witness.json"),
              "migration_sha256": sha256_file(evidence / "migration.json")}
    if any(witness.get(key) != digest for key, digest in actual.items()):
        raise RuntimeError("Current preflight belongs to different recovery evidence")
    report = {"status": "RECOVERY_CHECK_PASS", "updated_at": utc_now(), **actual,
              "data_manifest_sha256": sha256_file(path), "recovery_config_sha256": sha256_file(root / CONFIG),
              "resumed_candidate": "AAA_seed42", "completed_epochs": 7, "resume_at_epoch": 8,
              "scientific_parameters_changed": False, "max_formal_runs": 14,
              "original_preserved": True, "formal_training_started": False}
    write_json(evidence / "check.json", report)
    report["check_sha256"] = sha256_file(evidence / "check.json")
    return report


def remote_launch():
    """Root invokes this only after fresh independent command replay passes."""
    root, parent, evidence = _layout()
    if (evidence / "formal_launch.json").exists() or (evidence / "formal_launch.claim").exists():
        raise RuntimeError("An isolated launch already exists; automatic duplicate launch refused")
    # Do not regenerate check bytes after independent review.
    check = read_json(evidence / "check.json"); check["check_sha256"] = sha256_file(evidence / "check.json")
    if check.get("status") != "RECOVERY_CHECK_PASS":
        raise RuntimeError("A passed recovery check is required before launch")
    review = read_json(evidence / "independent_doc_review.json")
    if review.get("status") != "PASS" or any(review.get(key) != check.get(key) for key in REVIEW_KEYS):
        raise RuntimeError("Fresh independent documented-command replay for this exact recovery is required")
    if sha256_file(root / "source_freeze.json") != check["source_freeze_sha256"]:
        raise RuntimeError("Reviewed recovery source freeze changed")
    _verify_sources(root, parent); _verify_migration_files(read_json(evidence / "migration.json"))
    for relative, key in [("environment_lock.json", "environment_lock_sha256"),
                          ("model_preflight/preflight.json", "preflight_sha256"),
                          ("model_preflight/initialization_audit.json", "initialization_audit_sha256"),
                          ("numerics_witness/amp_recovery_witness.json", "numerics_witness_sha256"),
                          ("migration.json", "migration_sha256")]:
        if sha256_file(evidence / relative) != check[key]:
            raise RuntimeError("Reviewed witness changed: " + relative)
    _assert_no_controller(root); _assert_no_controller(parent)
    _gpu_idle()
    from .common import load_config, require_manifest
    config = load_config(root / CONFIG)
    _, manifest = require_manifest(config)
    for sample in manifest["samples"]:
        if sample["split"] in {"train", "val", "test"}:
            for field, digest_field in [("image_path", "sha256"), ("annotation_path", "annotation_sha256")]:
                if sha256_file(sample[field]) != sample[digest_field]: raise RuntimeError("Approved raw input changed after review")
    if (Path(config["output_dir"]) / "campaign.lock").exists(): raise RuntimeError("Existing recovery campaign lock")
    env = os.environ.copy(); env.update(PYTHONPATH=str(root / "vendor") + os.pathsep + str(root), CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env[key] = str(config["torch_threads"])
    command = [PYTHON, "-m", "tools.sonar_fcos_pilot.run", "campaign", "--config", str(root / CONFIG)]
    claim = os.open(evidence / "formal_launch.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(claim, json.dumps(check).encode()); os.close(claim)
    with (evidence / "formal_stdout.log").open("ab") as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"status": "LAUNCHED", "pid": child.pid, "physical_gpu": 1, "command": command,
               "resume_at_epoch": 8, "resumed_existing_run": "AAA_seed42", "new_candidate_run": False,
               "max_formal_runs": 14, "epochs_per_run": 30, "independent_doc_review_sha256": sha256_file(evidence / "independent_doc_review.json"),
               **{key: check[key] for key in REVIEW_KEYS}, "formal_training_started": True, "launched_at": utc_now()}
    write_json(evidence / "formal_launch.json", receipt)
    return receipt


def remote_status():
    root, parent, evidence = _layout()
    payload = {"root": str(root), "parent_root": str(parent), "formal_training_started": False}
    for relative in ["migration.json", "preflight_binding.json", "check.json", "formal_launch.json"]:
        if (evidence / relative).is_file(): payload[relative] = read_json(evidence / relative)
    launch = payload.get("formal_launch.json")
    if launch:
        payload["formal_training_started"] = True
        payload["controller_alive"] = (Path("/proc") / str(launch["pid"])).exists()
    payload["runs"] = {path.parent.name: read_json(path) for path in sorted((root / OUTPUT / "runs").glob("*/status.json"))}
    for name in ["campaign_status.json", "baseline_gate.json", "selection.json", "comparison.json"]:
        if (root / OUTPUT / name).exists(): payload[name] = read_json(root / OUTPUT / name)
    log = evidence / "formal_stdout.log"
    if log.is_file():
        with log.open("rb") as stream: stream.seek(max(0, log.stat().st_size - 6000)); payload["stdout_tail"] = stream.read().decode(errors="replace")
    return payload


def fetch():
    from .remote import _execute
    code = "from pathlib import Path\nimport json,hashlib,base64\nr=Path(" + repr(RECOVERY) + ")\n" + r'''
names={'migration.json','preflight_binding.json','environment_lock.json','preflight.json','initialization_audit.json','amp_recovery_witness.json','check.json','independent_doc_review.json','formal_launch.json','campaign_status.json','baseline_gate.json','selection.json','comparison.json','status.json','history.json','summary.json','run_manifest.json'}
values={}
for path in (r/'results/sonar_fcos_pilot_20261008').rglob('*'):
 if path.is_file() and path.name in names and path.stat().st_size<16*1024*1024:
  raw=path.read_bytes();values[path.relative_to(r).as_posix()]={'raw_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(values))
'''
    records = json.loads(_execute(code, 180))
    destination = LOCAL / "snapshots" / utc_now().replace(":", "").replace("+", "_")
    for relative, record in records.items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()): raise RuntimeError("Unsafe evidence transport path")
        raw = base64.b64decode(record["raw_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]: raise RuntimeError("Evidence transport checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    receipt = {"status": "FETCH_PASS", "destination": str(destination), "files": len(records), "checkpoint_files_copied": False}
    write_json(LOCAL / "fetch.json", receipt)
    return receipt


def review_upload(path):
    from .remote import _copy, _execute
    review = read_json(path)
    if review.get("status") != "PASS" or any(not review.get(key) for key in REVIEW_KEYS):
        raise RuntimeError("Only a complete independent PASS report can be uploaded")
    _copy(path, RECOVERY + "/" + EVIDENCE + "/independent_doc_review.json")
    code = "from pathlib import Path\nimport json,hashlib\np=Path(" + repr(RECOVERY + "/" + EVIDENCE + "/independent_doc_review.json") + ")\nprint(json.dumps({'status':'REVIEW_UPLOADED','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))\n"
    receipt = json.loads(_execute(code))
    if receipt["sha256"] != sha256_file(path): raise RuntimeError("Independent review transport hash mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage", "migrate", "preflight", "check", "launch", "status", "fetch", "review-upload"])
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args()
    if args.mode == "stage": value = stage()
    elif args.mode == "fetch": value = fetch()
    elif args.mode == "review-upload":
        if not args.review_file: parser.error("--review-file is required")
        value = review_upload(args.review_file)
    else: value = _remote_call(args.mode, timeout=900 if args.mode in {"migrate", "preflight", "check"} else 180)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
