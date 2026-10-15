"""Remote-only, finite reference campaign; no test mount, no recurring jobs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('cylinder2', 'uatd', 'sctd')
SOURCES = {
    'cylinder2': '/home/jiajia/experiments/sonar_phase2c_20260914_v1/data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8',
    'uatd': '/home/jiajia/experiments/sonar_phase3c_20260918_v1/datasets/UATD/clean_phase3b',
    'sctd': '/home/jiajia/experiments/sonar_phase3c_20260918_v1/datasets/SCTD/clean_phase3b_v2',
}
RESIDENTS = {2: 5406, 3: 5300}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def source_binding():
    return sha(ROOT / 'source_manifest.json')


def verify_sources():
    for rel, expected in json.loads((ROOT / 'source_manifest.json').read_text()).items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Source changed: ' + rel)


def stage_data():
    import yaml
    manifests = json.loads((ROOT / 'data_files.json').read_text())
    records = {}
    for dataset in DATASETS:
        src = Path(SOURCES[dataset])
        dest = ROOT / 'data' / dataset
        source_yaml = yaml.safe_load((src / 'data.yaml').read_text())
        entries = manifests[dataset]['files']
        counts = {}
        for row in entries:
            rel = Path(row['path'])
            if rel.parts[0] not in ('train', 'valid') or rel.parts[1] not in ('images', 'labels'):
                raise ValueError('Unexpected split: ' + str(rel))
            source = src / rel
            if source.stat().st_size != row['size'] or sha(source) != row['sha256']:
                raise ValueError('Remote data mismatch: ' + str(source))
            link = dest / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != source.resolve():
                    raise ValueError('Link changed')
            elif link.exists():
                raise FileExistsError(link)
            else:
                link.symlink_to(source.resolve())
            counts['/'.join(rel.parts[:2])] = counts.get('/'.join(rel.parts[:2]), 0) + 1
        if (dest / 'test').exists():
            raise RuntimeError('Test mounted')
        payload = {'path': str(dest), 'train': 'train/images', 'val': 'valid/images',
                   'nc': source_yaml['nc'], 'names': source_yaml['names']}
        (dest / 'data.yaml').write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
        # Small real-image view only for execution validation; never used as formal results.
        smoke = ROOT / 'data_smoke' / dataset
        for split, limit in [('train', 16), ('valid', 8)]:
            rows = sorted((dest / split / 'images').iterdir())[:limit]
            for image in rows:
                label = dest / split / 'labels' / (image.stem + '.txt')
                for kind, target in [('images', image), ('labels', label)]:
                    link = smoke / split / kind / target.name
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if not link.exists():
                        link.symlink_to(target)
        smoke_payload = dict(payload, path=str(smoke))
        (smoke / 'data.yaml').write_text(yaml.safe_dump(smoke_payload, sort_keys=False), encoding='utf-8')
        records[dataset] = {'counts': counts, 'yaml': str(dest / 'data.yaml'),
                            'yaml_sha256': sha(dest / 'data.yaml'), 'test_mounted': False}
    result = {'status': 'PASS', 'datasets': records, 'data_files_sha256': sha(ROOT / 'data_files.json'), 'time': now()}
    dump('data_stage.json', result)
    return result


def environment():
    import torch
    import ultralytics
    from ultralytics.utils.downloads import attempt_download_asset
    spec = json.loads((ROOT / 'environment_spec.json').read_text())
    for name, version in spec['base_pins'].items():
        if md.version(name) != version:
            raise RuntimeError('Base pin drift: ' + name)
    if ultralytics.__version__ != '8.3.221':
        raise RuntimeError('Unexpected Ultralytics')
    weights = ROOT / 'weights'
    weights.mkdir(exist_ok=True)
    receipt = {}
    for model in ('yolov8n', 'yolo11n'):
        path = weights / (model + '.pt')
        downloaded = Path(attempt_download_asset(str(path)))
        if not downloaded.is_file() or downloaded.stat().st_size < 1000000:
            raise RuntimeError('Incomplete weights: ' + model)
        receipt[model] = {'path': str(path), 'sha256': sha(path),
                          'source': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/' + path.name}
    package_names = ['torch', 'torchvision', 'numpy', 'Pillow', 'ultralytics', 'ultralytics-thop',
                     'polars', 'polars-runtime-32', 'psutil', 'opencv-python-headless', 'scipy', 'matplotlib']
    result = {'status': 'IMPORTS_AND_WEIGHTS_READY_GPU_WITNESS_PENDING', 'python': sys.version,
              'packages': {p: md.version(p) for p in package_names}, 'weights': receipt,
              'cuda': torch.version.cuda, 'spec_sha256': sha(ROOT / 'environment_spec.json')}
    dump('environment_lock.json', result)
    return result


def resources(gpu):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    record = next(line.split(',') for line in raw.splitlines() if int(line.split(',')[0]) == gpu)
    uuid = record[1].strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    selected = [line.split(',') for line in apps.splitlines() if line.split(',')[0].strip() == uuid]
    unexpected = [r for r in selected if int(r[1]) != RESIDENTS[gpu]]
    if unexpected or int(record[2]) < 16000 or int(record[3]) > 10:
        raise RuntimeError('GPU unavailable; refusing contention with new work: ' + str(gpu))
    for row in selected:
        if int(row[2]) > 2048:
            raise RuntimeError('Resident memory changed')
        proc = Path('/proc') / row[1].strip()
        cmd = (proc / 'cmdline').read_bytes().split(b'\0')[0].decode()
        if cmd != '/home/sqg/seed-vc-live/env/bin/python':
            raise RuntimeError('Resident identity changed')
    return {'gpu': gpu, 'uuid': uuid, 'free_mib': int(record[2]), 'resident_pids': [int(r[1]) for r in selected]}


def run_command(dataset, model, gpu, *, smoke=False, imgsz=None, init=None):
    project = ROOT / 'results' / ('smoke' if smoke else 'formal')
    name = dataset if not smoke else f'{dataset}_{model}_{init}_{imgsz}'
    data = ROOT / ('data_smoke' if smoke else 'data') / dataset / 'data.yaml'
    command = [sys.executable, str(ROOT / 'scripts/run_reference_baselines.py'), '--data', str(data),
               '--name', name, '--models', model, '--init', *([init] if smoke else ['scratch', 'coco']),
               '--imgsz', *([str(imgsz)] if smoke else ['640', '1280']),
               '--seeds', *(['42'] if smoke else ['42', '43', '44']),
               '--epochs', '1' if smoke else '300', '--batch', '16', '--device', '0',
               '--project', str(project), '--workers', '2', '--weights-dir', str(ROOT / 'weights')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    return command, env


def spawn_case(dataset, model, gpu, *, smoke=False, imgsz=None, init=None):
    verify_sources()
    resource = resources(gpu)
    cmd, env = run_command(dataset, model, gpu, smoke=smoke, imgsz=imgsz, init=init)
    label = f'{dataset}_{model}' + (f'_{init}_{imgsz}_smoke' if smoke else '_formal')
    path = ROOT / 'logs' / (label + '.log')
    path.parent.mkdir(exist_ok=True)
    with path.open('x') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return child, {'pid': child.pid, 'gpu': gpu, 'command': cmd, 'log': str(path), 'resource_before': resource}


def smoke():
    # Four Cylinder2 shapes cover both models/resolutions/initialization types;
    # the other four cases exercise 10- and 3-class targets with both models.
    cases = [('cylinder2', 'yolov8n', 640, 'scratch'), ('cylinder2', 'yolo11n', 640, 'coco'),
             ('cylinder2', 'yolov8n', 1280, 'coco'), ('cylinder2', 'yolo11n', 1280, 'scratch'),
             ('uatd', 'yolov8n', 640, 'scratch'), ('uatd', 'yolo11n', 640, 'coco'),
             ('sctd', 'yolov8n', 640, 'coco'), ('sctd', 'yolo11n', 640, 'scratch')]
    completed = []
    for i in range(0, len(cases), 2):
        pair = []
        for gpu, case in zip((2, 3), cases[i:i+2]):
            dataset, model, size, init = case
            child, row = spawn_case(dataset, model, gpu, smoke=True, imgsz=size, init=init)
            pair.append((child, row))
        dump('smoke_status.json', {'status': 'RUNNING', 'cases': [r for _, r in pair], 'completed': completed})
        for child, row in pair:
            row['returncode'] = child.wait()
            completed.append(row)
        if any(row['returncode'] for _, row in pair):
            dump('smoke_status.json', {'status': 'FAILED', 'completed': completed})
            raise RuntimeError('Real-data smoke failed; inspect logs')
    result = {'status': 'PASS', 'completed': completed, 'test_evaluated': False,
              'source_manifest_sha256': source_binding(), 'environment_lock_sha256': sha(ROOT / 'environment_lock.json'),
              'data_stage_sha256': sha(ROOT / 'data_stage.json'), 'formal_results': False, 'time': now()}
    dump('smoke_gate.json', result)
    dump('smoke_status.json', {'status': 'PASS', 'count': len(completed)})
    return result


def verify():
    """Read-only documented invocation for the independent reviewer."""
    import yaml
    verify_sources()
    lock = json.loads((ROOT / 'environment_lock.json').read_text())
    for name, version in lock['packages'].items():
        if md.version(name) != version:
            raise RuntimeError('Package drift')
    stage = json.loads((ROOT / 'data_stage.json').read_text())
    for ds in DATASETS:
        path = ROOT / 'data' / ds / 'data.yaml'
        cfg = yaml.safe_load(path.read_text())
        if 'test' in cfg or (path.parent / 'test').exists():
            raise RuntimeError('Test exposed')
        if sha(path) != stage['datasets'][ds]['yaml_sha256']:
            raise RuntimeError('Data YAML drift')
    gate = json.loads((ROOT / 'smoke_gate.json').read_text())
    if gate['status'] != 'PASS' or gate['source_manifest_sha256'] != source_binding():
        raise RuntimeError('Smoke gate missing or stale')
    return {'status': 'PASS', 'source_manifest_sha256': source_binding(),
            'environment_lock_sha256': sha(ROOT / 'environment_lock.json'), 'data_stage_sha256': sha(ROOT / 'data_stage.json'),
            'smoke_cases': len(gate['completed']), 'test_mounted': False}


def campaign():
    verify()
    completed = []
    for dataset in DATASETS:
        # A hard dataset barrier: both model queues finish before the next dataset.
        pair = []
        for gpu, model in ((2, 'yolov8n'), (3, 'yolo11n')):
            child, row = spawn_case(dataset, model, gpu)
            pair.append((child, row))
        dump('campaign_status.json', {'status': 'RUNNING', 'dataset': dataset, 'active': [r for _, r in pair],
                                     'completed': completed, 'planned_runs': 72, 'test_evaluated': False, 'time': now()})
        for child, row in pair:
            row['returncode'] = child.wait()
            completed.append(row)
        if any(row['returncode'] for _, row in pair):
            dump('campaign_status.json', {'status': 'FAILED', 'dataset': dataset, 'completed': completed, 'time': now()})
            raise RuntimeError('Model queue failed; preserved all results; next dataset not started')
    result = {'status': 'COMPLETE', 'completed': completed, 'planned_runs': 72, 'test_evaluated': False, 'time': now()}
    dump('campaign_status.json', result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['stage-data', 'environment', 'smoke', 'verify', 'campaign'])
    mode = ap.parse_args().mode.replace('-', '_')
    print(json.dumps(globals()[mode](), indent=2))
