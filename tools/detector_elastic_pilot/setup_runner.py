"""One-time creation of an isolated pilot runner from audited legacy entries.

Never edits the legacy source. New files must not already exist.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / 'tools/detector_elastic_pilot'
RESULTS = ROOT / 'results/detector_elastic_pilot_20261005'


def put(relative, text):
    path = ROOT / relative
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    legacy = ROOT / 'results/detector_b1_light_20261003/remote_snapshot/raw_bytes/execution_manifest.json'
    payload = json.loads(legacy.read_text(encoding='utf-8'))
    hashes = payload['source_sha256']
    changed = [name for name, digest in hashes.items() if sha(ROOT / name) != digest]
    if changed:
        raise ValueError(f'Legacy light source already drifted: {changed}')
    preserved = dict(legacy_manifest_sha256=sha(legacy), source_sha256=hashes,
                     verified_source_files=len(hashes), status='PASS_before_pilot_edits')
    (RESULTS / 'legacy_preservation_before.json').write_text(json.dumps(preserved, indent=2), encoding='utf-8')

    for filename in ('train.py', 'queue_runner.py', 'remote.py'):
        text = (ROOT / 'tools/detector_b1_light' / filename).read_text(encoding='utf-8')
        text = text.replace('detector_b1_light', 'detector_elastic_pilot')
        text = text.replace('20261003', '20261005')
        text = text.replace('B1-light', 'ELASTIC-inspired B1 pilot')
        text = text.replace('B1_LIGHT_SOURCE_V1', 'ELASTIC_PILOT_SOURCE_V1')
        text = text.replace('B1-neck/head', 'ELASTIC-inspired B1-neck/head')
        text = text.replace('L1', 'E0').replace('L0', 'E1')
        text = text.replace('neck_sha256', 'common_neck_sha256')
        text = text.replace('shared_B1_backbone_neck_prediction_per_seed_independent_DS_tower',
                            'shared_B1_backbone_common_topdown_prediction_per_seed_independent_PAN')
        if filename == 'train.py':
            text = text.replace("SEEDS = (42, 43, 44)", "SEEDS = (42,)")
            text = text.replace('seeds 42/43/44', 'seed 42')
            text = text.replace("cfg['model_spec'], cfg['baseline']", "cfg['environment_spec'], cfg['model_spec'], cfg['baseline']")
        if filename == 'remote.py':
            text = text.replace("archived=root/'results/detector_elastic_pilot_20261005/preflight_failures/transport01/staging_manifest.json'\npreserved_failure=previous.is_file() and archived.is_file() and previous.read_bytes()==archived.read_bytes()\n", '')
            text = text.replace(' and not preserved_failure', '')
        put('tools/detector_elastic_pilot/' + filename, text)

    text = (ROOT / 'configs/detector_b1_light_training.yaml').read_text(encoding='utf-8')
    text = text.replace('detector_b1_light', 'detector_elastic_pilot').replace('20261003', '20261005')
    text = text.replace('[L1, L0]', '[E0, E1]').replace('[42, 43, 44]', '[42]')
    text = text.replace('shared_B1_backbone_neck_prediction_per_seed_independent_DS_tower',
                        'shared_B1_backbone_common_topdown_prediction_per_seed_independent_PAN')
    text = text.replace('model_spec:', 'environment_spec: configs/detector_elastic_pilot_environment.yaml\nmodel_spec:')
    put('configs/detector_elastic_pilot_training.yaml', text)

    text = (ROOT / 'tests/test_detector_b1_light_runner.py').read_text(encoding='utf-8')
    text = text.replace('detector_b1_light', 'detector_elastic_pilot')
    text = text.replace('B1-light', 'ELASTIC-inspired B1 pilot').replace('L1', 'E0').replace('L0', 'E1')
    text = text.replace('neck_sha256', 'common_neck_sha256').replace('tower_sha256', 'full_neck_sha256')
    text = text.replace('train.SEEDS==(42,43,44)', 'train.SEEDS==(42,)')
    put('tests/test_detector_elastic_pilot_runner.py', text)
    print(json.dumps(dict(status='CREATED', legacy_sources_preserved=len(hashes))))


if __name__ == '__main__':
    main()
