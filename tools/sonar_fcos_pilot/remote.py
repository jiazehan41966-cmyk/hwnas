"""Isolated existing-server preparation and strictly gated formal execution.

Credentials are read by the existing Credential Manager transport. Only this
pilot's new source, configuration and verified official FCOS weights are staged.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from .common import DEFAULT_CONFIG, ROOT, load_config, require_manifest, resolve_path, sha256_file, utc_now, write_json
from .model import DEFAULT_CACHE


REMOTE = "/home/jiajia/experiments/sonar_fcos_pilot_20261008"
PYTHON = "/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python"
GPU = 1
REMOTE_CONFIG = "configs/sonar_fcos_pilot_remote_v1.json"
RESULT = "results/sonar_fcos_pilot_20261008/remote"
LOCAL = ROOT / "results" / "sonar_fcos_pilot_20261008" / "remote_transport"
WEIGHT_FILENAME = "fcos_resnet50_fpn_coco-99b0c9b7.pth"
WEIGHT_SHA256 = "99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7"


def _transport():
    from tools.detector_v0_2.remote import ENV, HOST, SCP, execute
    return execute, SCP, ENV, HOST


def _execute(code, timeout=60):
    return _transport()[0](code, timeout)


def _json(code, timeout=60):
    return json.loads(_execute(code, timeout))


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _copy(local_path, remote_path, timeout=300):
    _, scp, env, host = _transport()
    subprocess.run(scp + [str(local_path), host + ":" + remote_path], env=env, check=True, capture_output=True, timeout=timeout)


def _base_code():
    return "from pathlib import Path\nimport json, hashlib, os, subprocess, sys\nroot=Path(" + repr(REMOTE) + ").resolve()\nassert root.parent == Path('/home/jiajia/experiments').resolve()\ndef file_sha(path):\n digest=hashlib.sha256()\n with Path(path).open('rb') as stream:\n  for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)\n return digest.hexdigest()\n"


def probe():
    code = _base_code() + """
import importlib.metadata, platform
versions={}
for name in ['torch','torchvision','numpy','Pillow','pycocotools']:
 try: versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError: versions[name]=None
gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
payload={'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'python':platform.python_version(),'executable':sys.executable,'base_packages':versions,'remote_root':str(root),'root_exists':root.exists(),'physical_gpu_selected':1,'gpu_status':gpu_lines.strip().splitlines(),'formal_training_started':False}
print(json.dumps(payload))
"""
    value = _json(code)
    write_json(LOCAL / "probe.json", value)
    return value


def _environment_spec():
    return {
        "schema_version": 1,
        "provider": "existing_user_owned_server",
        "python_executable": PYTHON,
        "remote_root": REMOTE,
        "base_packages": {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "numpy": "2.4.4", "Pillow": "11.3.0"},
        "overlay": {"directory": "vendor", "packages": ["pycocotools==2.0.11"], "no_deps": True, "shared_environment_modified": False},
        "physical_gpu": GPU,
        "threads": 4,
        "smoke": {"image_size": 640, "batch_size": 1, "amp": "float16", "combinations": ["AAA", "BBB"], "formal_training": False},
        "official_weight": {"filename": WEIGHT_FILENAME, "sha256": WEIGHT_SHA256, "source_url": "https://download.pytorch.org/models/" + WEIGHT_FILENAME},
    }


def stage(weight_path=None):
    """Stage a source snapshot; refused while pilot processes/formal runs exist."""
    paths = sorted(path for path in (ROOT / "tools" / "sonar_fcos_pilot").iterdir() if path.is_file() and path.suffix in {".py", ".md"})
    paths.append(ROOT / "configs" / "sonar_fcos_pilot_v1.json")
    if any(path.is_symlink() for path in paths):
        raise ValueError("Symlinked source files are not accepted")
    spec = _environment_spec()
    spec_sha = _canonical_sha(spec)
    write_json(LOCAL / "environment_spec.json", spec)
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    manifest = {"schema_version": 1, "timestamp_utc": utc_now(), "source_sha256": hashes, "environment_spec_sha256": spec_sha, "legacy_networks_copied": False, "data_copied": False, "legacy_weights_copied": False, "formal_training_started": False}
    manifest["source_set_sha256"] = _canonical_sha(hashes)
    write_json(LOCAL / "source_freeze.json", manifest)
    archive = LOCAL / ("source_" + manifest["source_set_sha256"][:16] + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        for path in paths:
            stream.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        stream.add(LOCAL / "source_freeze.json", arcname="source_freeze.json", recursive=False)
        stream.add(LOCAL / "environment_spec.json", arcname="environment_spec.json", recursive=False)
    archive_sha = sha256_file(archive)
    code = _base_code() + "root.mkdir(parents=True,exist_ok=True)\n(root/'staging').mkdir(exist_ok=True)\nprint(json.dumps({'status':'DIRECTORY_READY'}))\n"
    _json(code)
    remote_archive = REMOTE + "/staging/" + archive.name
    _copy(archive, remote_archive)
    code = _base_code() + "archive=root/'staging'/" + repr(archive.name) + "\nexpected_archive_sha=" + repr(archive_sha) + "\n" + """
import tarfile
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_archive_sha: raise RuntimeError('Source archive transport hash mismatch')
for process in Path('/proc').iterdir():
 if not process.name.isdigit(): continue
 try:
  if (process/'cwd').resolve()==root and int(process.name)!=os.getpid():
   command=(process/'cmdline').read_bytes()
   if b'tools.sonar_fcos_pilot' in command or b'tools/sonar_fcos_pilot' in command: raise RuntimeError('An active pilot process prevents source restaging')
 except (PermissionError,FileNotFoundError,ProcessLookupError): pass
formal=root/'results'/'sonar_fcos_pilot_20261008'/'runs'
if formal.exists() and any(formal.rglob('status.json')): raise RuntimeError('Formal-run artifacts prevent source restaging')
if (root/'results'/'sonar_fcos_pilot_20261008'/'remote'/'formal_launch.json').exists(): raise RuntimeError('A formal launch receipt prevents source restaging')
payloads={}
with tarfile.open(archive,'r:gz') as stream:
 for member in stream.getmembers():
  name=Path(member.name)
  destination=(root/name).resolve()
  allowed=(member.name.startswith('tools/sonar_fcos_pilot/') and name.suffix in {'.py','.md'}) or member.name in {'configs/sonar_fcos_pilot_v1.json','source_freeze.json','environment_spec.json'}
  if not member.isfile() or name.is_absolute() or '..' in name.parts or not destination.is_relative_to(root) or not allowed: raise RuntimeError('Unsafe source archive member: '+member.name)
  if member.name in payloads: raise RuntimeError('Duplicate source member: '+member.name)
  payloads[member.name]=stream.extractfile(member).read()
marker=json.loads(payloads['source_freeze.json'])
spec=json.loads(payloads['environment_spec.json'])
canonical=lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if canonical(spec)!=marker['environment_spec_sha256']: raise RuntimeError('Spec hash mismatch')
if canonical(marker['source_sha256'])!=marker['source_set_sha256']: raise RuntimeError('Source-set hash mismatch')
expected=set(marker['source_sha256'])|{'source_freeze.json','environment_spec.json'}
if set(payloads)!=expected: raise RuntimeError('Archive member list differs from source freeze')
for relative,digest in marker['source_sha256'].items():
 if hashlib.sha256(payloads[relative]).hexdigest()!=digest: raise RuntimeError('Source content hash mismatch: '+relative)
for relative,data in payloads.items():
 destination=root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
 destination.write_bytes(data)
result=root/'results'/'sonar_fcos_pilot_20261008'/'remote';result.mkdir(parents=True,exist_ok=True)
report={'status':'STAGED','files':len(marker['source_sha256']),'source_set_sha256':marker['source_set_sha256'],'source_freeze_sha256':hashlib.sha256((root/'source_freeze.json').read_bytes()).hexdigest(),'environment_spec_sha256':marker['environment_spec_sha256'],'archive_sha256':expected_archive_sha,'data_copied':False,'legacy_networks_copied':False,'formal_training_started':False}
(result/'stage.json').write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps(report))
"""
    value = _json(code)
    candidates = [Path(weight_path)] if weight_path else [DEFAULT_CACHE / "checkpoints" / WEIGHT_FILENAME, Path("F:/sonar_fcos_pilot/weights/checkpoints") / WEIGHT_FILENAME, Path("F:/sonar_fcos_pilot/weights") / WEIGHT_FILENAME]
    weight = next((path for path in candidates if path.is_file()), None)
    if weight is None:
        raise FileNotFoundError("Verified official FCOS weights missing; supply --weight-path")
    if sha256_file(weight) != WEIGHT_SHA256:
        raise ValueError("Official FCOS checkpoint SHA256 mismatch")
    _execute(_base_code() + "(root/'weights'/'torch'/'checkpoints').mkdir(parents=True,exist_ok=True)\n")
    remote_weight = REMOTE + "/weights/torch/checkpoints/" + WEIGHT_FILENAME
    existing = _json(_base_code() + "p=root/'weights'/'torch'/'checkpoints'/" + repr(WEIGHT_FILENAME) + "\nprint(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}))\n")
    if existing["sha256"] != WEIGHT_SHA256:
        _copy(weight, remote_weight)
    value["official_weight"] = _json(_base_code() + "p=root/'weights'/'torch'/'checkpoints'/" + repr(WEIGHT_FILENAME) + "\ndigest=hashlib.sha256(p.read_bytes()).hexdigest()\nassert digest==" + repr(WEIGHT_SHA256) + "\nprint(json.dumps({'path':str(p),'sha256':digest,'bytes':p.stat().st_size}))\n")
    value["official_weight"]["warm_reused"] = existing["sha256"] == WEIGHT_SHA256
    write_json(LOCAL / "stage.json", value)
    return value


def environment():
    """Build only a private, one-package overlay; leave shared venv unchanged."""
    code = _base_code() + "python=" + repr(PYTHON) + "\n" + """
import importlib.metadata
spec=json.loads((root/'environment_spec.json').read_text())
canonical=lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
marker=json.loads((root/'source_freeze.json').read_text())
if canonical(spec)!=marker['environment_spec_sha256']: raise RuntimeError('Environment spec differs from source freeze')
for name,expected in spec['base_packages'].items():
 observed=importlib.metadata.version(name)
 if observed!=expected: raise RuntimeError('Unexpected base version for '+name+': '+observed)
vendor=root/'vendor';vendor.mkdir(exist_ok=True)
dist=list(vendor.glob('pycocotools-2.0.11.dist-info'))
if not dist:
 command=[python,'-m','pip','install','--index-url','https://pypi.org/simple','--target',str(vendor),'--no-deps','pycocotools==2.0.11']
 completed=subprocess.run(command,cwd=root,capture_output=True,text=True)
 log=root/'results'/'sonar_fcos_pilot_20261008'/'remote'/'overlay_install.log'
 log.write_text(completed.stdout+'\\n'+completed.stderr)
 if completed.returncode: raise RuntimeError('Private overlay installation failed; inspect overlay_install.log')
env=os.environ.copy();env.update(PYTHONPATH=str(vendor)+os.pathsep+str(root),CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
probe_code=r'''
import importlib.metadata,json,platform,sys
from pathlib import Path
import torch,torchvision,numpy as np,PIL,pycocotools
from pycocotools import mask
sample=np.zeros((5,5),dtype=np.uint8,order='F');sample[1:4,1:5]=1
encoded=mask.encode(sample)
area=float(mask.area(encoded)); assert area==12
print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'packages':{name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','Pillow','pycocotools']},'pycocotools_path':str(Path(pycocotools.__file__).resolve()),'pycocotools_kernel_area':area,'cuda_runtime':torch.version.cuda,'cuda_arch_list':torch.cuda.get_arch_list(),'selected_device_name':torch.cuda.get_device_name(0),'selected_device_capability':list(torch.cuda.get_device_capability(0))}))
'''
completed=subprocess.run([python,'-c',probe_code],cwd=root,env=env,capture_output=True,text=True,check=True)
lock=json.loads(completed.stdout)
if not Path(lock['pycocotools_path']).is_relative_to(vendor): raise RuntimeError('pycocotools did not load from the private overlay')
if lock['packages']['pycocotools']!='2.0.11': raise RuntimeError('Overlay package version mismatch')
lock.update({'status':'ENVIRONMENT_PROBE_PASS','environment_spec_sha256':canonical(spec),'source_freeze_sha256':hashlib.sha256((root/'source_freeze.json').read_bytes()).hexdigest(),'physical_gpu':1,'shared_environment_modified':False,'formal_training_started':False})
result=root/'results'/'sonar_fcos_pilot_20261008'/'remote';(result/'environment_lock.json').write_text(json.dumps(lock,indent=2)+'\\n')
print(json.dumps(lock))
"""
    value = _json(code, timeout=240)
    write_json(LOCAL / "environment_lock.json", value)
    return value


def smoke():
    """Run only synthetic model steps on explicitly selected physical GPU 1."""
    code = _base_code() + "python=" + repr(PYTHON) + "\n" + """
result=root/'results'/'sonar_fcos_pilot_20261008'/'remote'
marker=json.loads((root/'source_freeze.json').read_text())
for relative,digest in marker['source_sha256'].items():
 if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest: raise RuntimeError('Staged source changed: '+relative)
lock_path=result/'environment_lock.json';lock=json.loads(lock_path.read_text())
freeze_sha=hashlib.sha256((root/'source_freeze.json').read_bytes()).hexdigest()
if lock['source_freeze_sha256']!=freeze_sha: raise RuntimeError('Restaged source requires a fresh environment probe')
gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
selected=next([int(value.strip()) for value in line.split(',')] for line in gpu_lines.splitlines() if int(line.split(',')[0])==1)
if selected[1]>1024 or selected[2]-selected[1]<4096 or selected[3]>10: raise RuntimeError('GPU 1 is no longer free enough; no other process is modified')
env=os.environ.copy();env.update(PYTHONPATH=str(root/'vendor')+os.pathsep+str(root),CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
smoke_root=result/'model_preflight';smoke_root.mkdir(exist_ok=True)
command=[python,'-m','tools.sonar_fcos_pilot.model','--output-dir',str(smoke_root),'--cache-dir',str(root/'weights'/'torch')]
completed=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
(smoke_root/'stdout.log').write_text(completed.stdout+'\\n'+completed.stderr)
if completed.returncode: raise RuntimeError('Remote synthetic model smoke failed; inspect model_preflight/stdout.log')
model=json.loads((smoke_root/'preflight.json').read_text());initial=json.loads((smoke_root/'initialization_audit.json').read_text())
if model['status']!='PASS' or initial['status']!='PASS': raise RuntimeError('Model witness failed')
if model['kernel_witness']['device']!='cuda' or not model['kernel_witness']['finite']: raise RuntimeError('GPU dispatch witness invalid')
binding={'status':'PASS','physical_gpu':1,'source_freeze_sha256':freeze_sha,'source_set_sha256':marker['source_set_sha256'],'environment_spec_sha256':marker['environment_spec_sha256'],'environment_lock_sha256':hashlib.sha256(lock_path.read_bytes()).hexdigest(),'preflight_sha256':hashlib.sha256((smoke_root/'preflight.json').read_bytes()).hexdigest(),'initialization_audit_sha256':hashlib.sha256((smoke_root/'initialization_audit.json').read_bytes()).hexdigest(),'formal_training':False,'command':command,'gpu_before_smoke':{'used_mib':selected[1],'total_mib':selected[2],'utilization_percent':selected[3]}}
(smoke_root/'binding.json').write_text(json.dumps(binding,indent=2)+'\\n')
card={'status':'ENVIRONMENT_AND_MODEL_WITNESS_PASS','formal_training_status':'BLOCKED_DATA_AUDIT','physical_gpu':1,'shared_environment_modified':False,'data_mounted':False,'formal_training_started':False,'independent_document_replay':'PENDING','binding':binding}
(result/'ready_card.json').write_text(json.dumps(card,indent=2)+'\\n')
print(json.dumps(card))
"""
    value = _json(code, timeout=240)
    write_json(LOCAL / "ready_card.json", value)
    return value


def status():
    code = _base_code() + """
result=root/'results'/'sonar_fcos_pilot_20261008'/'remote';payload={}
for name in ['stage.json','environment_lock.json','ready_card.json']:
 path=result/name
 if path.is_file(): payload[name]=json.loads(path.read_text())
payload['gpu_status']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
formal=root/'results'/'sonar_fcos_pilot_20261008'/'runs'
payload['formal_training_started']=formal.exists() and any(formal.rglob('status.json'))
print(json.dumps(payload))
"""
    value = _json(code)
    write_json(LOCAL / "status.json", value)
    return value


def fetch():
    code = _base_code() + """
import base64
result=root/'results'/'sonar_fcos_pilot_20261008'/'remote'
paths=[root/'source_freeze.json',root/'environment_spec.json']
paths.extend(result/name for name in ['stage.json','environment_lock.json','ready_card.json','overlay_install.log','model_preflight/preflight.json','model_preflight/initialization_audit.json','model_preflight/binding.json','model_preflight/stdout.log'])
values={}
for path in paths:
 if not path.is_file(): continue
 if path.stat().st_size>10*1024*1024: raise RuntimeError('Unexpected evidence size')
 raw=path.read_bytes();values[path.relative_to(root).as_posix()]={'raw_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(values))
"""
    records = _json(code)
    stamp = utc_now().replace(":", "").replace("+", "_")
    destination = LOCAL / "snapshots" / stamp
    for relative, record in records.items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise ValueError("Unsafe fetched evidence path")
        raw = base64.b64decode(record["raw_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Fetched evidence hash mismatch: " + relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    receipt = {"status": "FETCHED", "timestamp_utc": utc_now(), "destination": str(destination), "files": len(records), "raw_sha256": {name: record["sha256"] for name, record in records.items()}}
    write_json(LOCAL / "fetch_receipt.json", receipt)
    return receipt


def _local_data_gate(config_path):
    """The gate runs before transport or archive creation, including on launch."""
    config = load_config(config_path)
    try:
        path, manifest = require_manifest(config)
    except (RuntimeError, FileNotFoundError, KeyError) as error:
        blocked = {"status": "BLOCKED_DATA", "reason": str(error), "formal_training_started": False, "ssh_invoked": False, "timestamp_utc": utc_now()}
        write_json(LOCAL / "data_gate.json", blocked)
        return config, None, None, blocked
    return config, path, manifest, None


def _data_archive(config, manifest_path, manifest):
    """Relocate a passed manifest, preserving raw content and original approval."""
    origin_sha = sha256_file(manifest_path)
    local = LOCAL / "data_packages" / origin_sha[:16]
    local.mkdir(parents=True, exist_ok=True)
    relocated = copy.deepcopy(manifest)
    relocated["origin_manifest_sha256"] = origin_sha
    relocated["dataset_root"] = REMOTE + "/data/UATD/raw"
    files = {}
    for sample in relocated["samples"]:
        key = PurePosixPath(sample["sample_key"])
        if key.is_absolute() or ".." in key.parts or len(key.parts) != 3 or key.parts[1] != "images":
            raise ValueError("Unexpected canonical raw sample key: " + str(key))
        excluded = sample.get("excluded") or sample["split"] == "excluded"
        if excluded and sample["split"] in {"train", "val", "test"}:
            raise ValueError("An excluded sample remains in an active loader split")
        prefix = "data/UATD/excluded_not_staged" if excluded else "data/UATD/raw"
        relative_image = prefix + "/" + key.as_posix()
        relative_annotation = prefix + "/" + key.parts[0] + "/annotations/" + Path(sample["annotation_path"]).name
        for field, digest_field, relative in [("image_path", "sha256", relative_image), ("annotation_path", "annotation_sha256", relative_annotation)]:
            source = Path(sample[field])
            if not excluded:
                if source.is_symlink() or not source.is_file():
                    raise ValueError("Missing or symlinked active raw input: " + str(source))
                observed = sha256_file(source)
                if observed != sample[digest_field]:
                    raise ValueError("Approved raw input content changed: " + str(source))
                if relative in files and files[relative][1] != observed:
                    raise ValueError("Two different raw files map to the same remote path")
                files[relative] = (source, observed)
            sample["origin_" + field] = sample[field]
            sample[field] = REMOTE + "/" + relative
    remote_manifest_rel = "results/sonar_fcos_pilot_20261008/data_audit/manifest.json"
    write_json(local / "manifest_remote.json", relocated)
    migrated_manifest_sha = sha256_file(local / "manifest_remote.json")
    remote_config = {key: value for key, value in config.items() if key not in {"config_path", "config_sha256"}}
    remote_config.update({"dataset_root": REMOTE + "/data/UATD/raw", "output_dir": REMOTE + "/results/sonar_fcos_pilot_20261008", "manifest_path": REMOTE + "/" + remote_manifest_rel, "checkpoint_root": REMOTE + "/checkpoints", "weight_cache": REMOTE + "/weights/torch", "device": "cuda", "group_map": None})
    if relocated.get("protocol_id") == "PUBLIC_SPLIT_EXPLORATORY_V1":
        approval = copy.deepcopy(config["approved_protocol"])
        acknowledgment = resolve_path(approval["acknowledgment_path"])
        if sha256_file(acknowledgment) != approval["acknowledgment_sha256"]:
            raise ValueError("Approved protocol acknowledgment changed")
        relative = "approvals/protocol_acknowledgment.json"
        files[relative] = (acknowledgment, approval["acknowledgment_sha256"])
        approval.update({"origin_manifest_sha256": origin_sha, "manifest_sha256": migrated_manifest_sha, "acknowledgment_path": REMOTE + "/" + relative})
        remote_config["approved_protocol"] = approval
    write_json(local / "config_remote.json", remote_config)
    files.update({remote_manifest_rel: (local / "manifest_remote.json", migrated_manifest_sha), REMOTE_CONFIG: (local / "config_remote.json", sha256_file(local / "config_remote.json")), "approvals/origin_manifest.json": (manifest_path, origin_sha), "approvals/origin_config.json": (Path(config["config_path"]), config["config_sha256"])})
    marker = {"status": "DATA_PAYLOAD_PASS", "origin_manifest_sha256": origin_sha, "remote_manifest_sha256": migrated_manifest_sha, "origin_config_sha256": config["config_sha256"], "remote_config_sha256": sha256_file(local / "config_remote.json"), "protocol_id": relocated.get("protocol_id", "STRICT_VERIFIED_ACQUISITION_SPLIT"), "payload_sha256": {relative: digest for relative, (_, digest) in files.items()}, "excluded_raw_files_copied": False, "active_images": sum(sample["split"] in {"train", "val", "test"} for sample in relocated["samples"]), "formal_training_started": False}
    write_json(local / "data_payload.json", marker)
    total_size = sum(path.stat().st_size for path, _ in files.values())
    if shutil.disk_usage(local).free < total_size + 2 * 1024**3:
        raise RuntimeError("Insufficient local disk space for the gated raw-data archive")
    archive = local / "approved_raw_data.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
        for relative, (path, _) in sorted(files.items()):
            stream.add(path, arcname=relative, recursive=False)
        stream.add(local / "data_payload.json", arcname="data_payload.json", recursive=False)
    return archive, marker


def data(config_path=DEFAULT_CONFIG):
    config, manifest_path, manifest, blocked = _local_data_gate(config_path)
    if blocked:
        return blocked
    archive, marker = _data_archive(config, manifest_path, manifest)
    archive_sha = sha256_file(archive)
    _execute(_base_code() + "(root/'staging').mkdir(parents=True,exist_ok=True)\n")
    name = "raw_data_" + marker["origin_manifest_sha256"][:16] + ".tar.gz"
    _copy(archive, REMOTE + "/staging/" + name, timeout=3600)
    code = _base_code() + "archive=root/'staging'/" + repr(name) + "\narchive_sha=" + repr(archive_sha) + "\n" + """
import tarfile
if file_sha(archive)!=archive_sha: raise RuntimeError('Raw-data archive transport hash mismatch')
result=root/'results'/'sonar_fcos_pilot_20261008'/'remote'
if (result/'formal_launch.json').exists(): raise RuntimeError('Cannot replace data after a formal launch')
with tarfile.open(archive,'r:gz') as stream:
 members=stream.getmembers();names=[member.name for member in members]
 if len(names)!=len(set(names)): raise RuntimeError('Duplicate data archive member')
 for member in members:
  name=Path(member.name);destination=(root/name).resolve()
  allowed=(member.name.startswith('data/UATD/raw/') and name.suffix.lower() in {'.bmp','.png','.jpg','.jpeg','.xml'}) or member.name in {'data_payload.json','approvals/origin_manifest.json','approvals/origin_config.json','approvals/protocol_acknowledgment.json','configs/sonar_fcos_pilot_remote_v1.json','results/sonar_fcos_pilot_20261008/data_audit/manifest.json'}
  if not member.isfile() or name.is_absolute() or '..' in name.parts or not destination.is_relative_to(root) or not allowed: raise RuntimeError('Unsafe data member: '+member.name)
 marker=json.loads(stream.extractfile('data_payload.json').read())
 if set(names)!=set(marker['payload_sha256'])|{'data_payload.json'}: raise RuntimeError('Data member list differs from receipt')
 for member in members:
  if member.name=='data_payload.json': continue
  destination=root/member.name
  if destination.exists():
   digest=hashlib.sha256()
   with destination.open('rb') as source:
    for chunk in iter(lambda:source.read(1024*1024),b''): digest.update(chunk)
   if digest.hexdigest()!=marker['payload_sha256'][member.name]: raise RuntimeError('Existing pilot data differs: '+member.name)
  else:
   destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.transport.tmp')
   digest=hashlib.sha256()
   with stream.extractfile(member) as source,temporary.open('wb') as output:
    for chunk in iter(lambda:source.read(1024*1024),b''): digest.update(chunk);output.write(chunk)
   if digest.hexdigest()!=marker['payload_sha256'][member.name]: raise RuntimeError('Raw data content hash mismatch: '+member.name)
   temporary.replace(destination)
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'vendor'))
from tools.sonar_fcos_pilot.common import load_config,require_manifest,sha256_file
config=load_config(root/'configs'/'sonar_fcos_pilot_remote_v1.json');manifest_path,manifest=require_manifest(config)
for sample in manifest['samples']:
 if sample['split'] in {'train','val','test'}:
  for field,digest_field in [('image_path','sha256'),('annotation_path','annotation_sha256')]:
   if sha256_file(sample[field])!=sample[digest_field]: raise RuntimeError('Migrated raw input changed: '+sample[field])
 elif sample.get('excluded') and Path(sample['image_path']).exists(): raise RuntimeError('Excluded raw input was unexpectedly staged')
report={**marker,'status':'DATA_STAGE_PASS','archive_sha256':archive_sha,'source_freeze_sha256':sha256_file(root/'source_freeze.json'),'remote_config':str(root/'configs'/'sonar_fcos_pilot_remote_v1.json'),'remote_manifest':str(manifest_path),'formal_training_started':False}
result.mkdir(parents=True,exist_ok=True);(result/'data_stage.json').write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps(report))
"""
    value = _json(code, timeout=1800)
    write_json(LOCAL / "data_stage.json", value)
    return value


def launch(config_path=DEFAULT_CONFIG):
    local_config, local_manifest_path, _, blocked = _local_data_gate(config_path)
    if blocked:
        write_json(LOCAL / "formal_launch_gate.json", blocked)
        return blocked
    code = _base_code() + "python=" + repr(PYTHON) + "\nexpected_origin_sha=" + repr(sha256_file(local_manifest_path)) + "\nexpected_origin_config_sha=" + repr(local_config["config_sha256"]) + "\n" + """
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'vendor'))
from tools.sonar_fcos_pilot.common import load_config,require_manifest,sha256_file
result=root/'results'/'sonar_fcos_pilot_20261008'/'remote'
config_path=root/'configs'/'sonar_fcos_pilot_remote_v1.json'
config=load_config(config_path);manifest_path,manifest=require_manifest(config)
data=json.loads((result/'data_stage.json').read_text())
if data['status']!='DATA_STAGE_PASS' or sha256_file(manifest_path)!=data['remote_manifest_sha256'] or sha256_file(config_path)!=data['remote_config_sha256']: raise RuntimeError('Passed migrated data/config binding is required')
if data['origin_manifest_sha256']!=expected_origin_sha or data['origin_config_sha256']!=expected_origin_config_sha: raise RuntimeError('Remote data differs from the currently approved local protocol')
marker=json.loads((root/'source_freeze.json').read_text());freeze_sha=sha256_file(root/'source_freeze.json')
for relative,digest in marker['source_sha256'].items():
 if sha256_file(root/relative)!=digest: raise RuntimeError('Source freeze changed: '+relative)
if data['source_freeze_sha256']!=freeze_sha: raise RuntimeError('Data and source were not staged together')
card=json.loads((result/'ready_card.json').read_text());binding=card['binding']
if card['status']!='ENVIRONMENT_AND_MODEL_WITNESS_PASS' or binding['source_freeze_sha256']!=freeze_sha: raise RuntimeError('Current model witness is required')
for relative,key in [('environment_lock.json','environment_lock_sha256'),('model_preflight/preflight.json','preflight_sha256'),('model_preflight/initialization_audit.json','initialization_audit_sha256')]:
 if sha256_file(result/relative)!=binding[key]: raise RuntimeError('Model/environment witness bytes changed')
review=json.loads((result/'independent_doc_review.json').read_text())
if review.get('status')!='PASS': raise RuntimeError('Independent documented-command review is required')
for key in ['source_freeze_sha256','environment_spec_sha256','environment_lock_sha256','preflight_sha256','initialization_audit_sha256']:
 if review.get(key)!=binding[key]: raise RuntimeError('Independent review belongs to different evidence: '+key)
for sample in manifest['samples']:
 if sample['split'] in {'train','val','test'}:
  if sha256_file(sample['image_path'])!=sample['sha256'] or sha256_file(sample['annotation_path'])!=sample['annotation_sha256']: raise RuntimeError('Approved raw input changed')
if (result/'formal_launch.json').exists() or (Path(config['output_dir'])/'campaign.lock').exists(): raise RuntimeError('Existing launch/lock; an automatic duplicate launch is forbidden')
for process in Path('/proc').iterdir():
 if not process.name.isdigit(): continue
 try:
  if int(process.name)!=os.getpid() and (process/'cwd').resolve()==root:
   command=(process/'cmdline').read_bytes()
   if b'tools.sonar_fcos_pilot' in command: raise RuntimeError('Another pilot process is active')
 except (PermissionError,FileNotFoundError,ProcessLookupError): pass
gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
selected=next([int(value.strip()) for value in line.split(',')] for line in gpu_lines.splitlines() if int(line.split(',')[0])==1)
if selected[1]>1024 or selected[2]-selected[1]<4096 or selected[3]>10: raise RuntimeError('Physical GPU 1 is occupied; launch refused')
env=os.environ.copy();env.update(PYTHONPATH=str(root/'vendor')+os.pathsep+str(root),CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS=str(config['torch_threads']),MKL_NUM_THREADS=str(config['torch_threads']),OPENBLAS_NUM_THREADS=str(config['torch_threads']),PYTHONUNBUFFERED='1')
command=[python,'-m','tools.sonar_fcos_pilot.run','campaign','--config',str(config_path)]
claim=os.open(result/'formal_launch.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
os.write(claim,json.dumps({'source_freeze_sha256':freeze_sha,'data_manifest_sha256':sha256_file(manifest_path)}).encode());os.close(claim)
with (result/'formal_stdout.log').open('ab') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'status':'LAUNCHED','pid':child.pid,'physical_gpu':1,'command':command,'source_freeze_sha256':freeze_sha,'data_stage_sha256':sha256_file(result/'data_stage.json'),'remote_config_sha256':sha256_file(config_path),'independent_doc_review_sha256':sha256_file(result/'independent_doc_review.json'),'max_formal_runs':config['max_formal_runs'],'epochs_per_run':config['epochs'],'formal_training_started':True}
(result/'formal_launch.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
"""
    value = _json(code, timeout=600)
    write_json(LOCAL / "formal_launch.json", value)
    return value


def training_status():
    code = _base_code() + """
output=root/'results'/'sonar_fcos_pilot_20261008';result=output/'remote';payload={'formal_training_started':False}
for relative in ['remote/formal_launch.json','campaign_status.json','baseline_gate.json','selection.json','comparison.json']:
 path=output/relative
 if path.is_file(): payload[relative]=json.loads(path.read_text())
launch=payload.get('remote/formal_launch.json')
if launch:
 payload['formal_training_started']=True
 process=Path('/proc')/str(launch['pid']);payload['controller_alive']=process.exists()
payload['runs']={}
for path in sorted((output/'runs').glob('*/status.json')):
 payload['runs'][path.parent.name]=json.loads(path.read_text())
log=result/'formal_stdout.log'
if log.is_file():
 with log.open('rb') as stream: stream.seek(max(0,log.stat().st_size-6000));payload['stdout_tail']=stream.read().decode(errors='replace')
print(json.dumps(payload))
"""
    value = _json(code)
    write_json(LOCAL / "training_status.json", value)
    return value


def fetch_training():
    """Fetch compact receipts and 12 pictures, never giant model checkpoints."""
    code = _base_code() + """
import base64
output=root/'results'/'sonar_fcos_pilot_20261008'
paths=[]
names={'campaign_status.json','baseline_gate.json','selection.json','comparison.json','test_authorization.json','eight_combinations.csv','第一轮结果.md','formal_launch.json','data_stage.json','independent_doc_review.json','status.json','summary.json','history.json','run_manifest.json','test_result.json'}
for path in output.rglob('*'):
 if path.is_file() and path.name in names and path.stat().st_size<=12*1024*1024: paths.append(path)
for path in sorted(output.rglob('*.png'))[:12]: paths.append(path)
values={}
for path in paths:
 if not path.resolve().is_relative_to(output.resolve()): raise RuntimeError('Evidence symlink escapes pilot output')
 raw=path.read_bytes();values[path.relative_to(root).as_posix()]={'raw_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(values))
"""
    records = _json(code, timeout=180)
    destination = LOCAL / "training_snapshots" / utc_now().replace(":", "").replace("+", "_")
    for relative, record in records.items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise ValueError("Unsafe training evidence path")
        raw = base64.b64decode(record["raw_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Training evidence transport hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    receipt = {"status": "FETCHED", "destination": str(destination), "files": len(records), "checkpoint_files_copied": False, "raw_sha256": {key: value["sha256"] for key, value in records.items()}}
    write_json(LOCAL / "fetch_training_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "stage", "environment", "smoke", "status", "fetch", "data", "launch", "training-status", "fetch-training"])
    parser.add_argument("--weight-path", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    function = {"probe": probe, "stage": stage, "environment": environment, "smoke": smoke, "status": status, "fetch": fetch, "data": data, "launch": launch, "training-status": training_status, "fetch-training": fetch_training}[args.mode]
    value = function(args.weight_path) if args.mode == "stage" else function(args.config) if args.mode in {"data", "launch"} else function()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
