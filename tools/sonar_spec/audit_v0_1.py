"""Read-only source/specification audit; no training, search or synthesis."""
import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/search_space_sonar_v0_1.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/search_space_v0_1')
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'reports/baseline_sonar_frozen/freeze_manifest.json').read_text())
    frozen = ROOT / 'configs/baseline_sonar_frozen.yaml'
    drift = [p for p, sha in manifest['source_sha256'].items() if digest(ROOT / p.replace('\\', '/')) != sha]
    assert not drift, f'Frozen source drift: {drift}'
    assert digest(frozen) == manifest['config_sha256'], 'Frozen configuration drift'
    source = json.loads((ROOT / 'artifacts/datasets/roboflow_cylider2_v6_source_index.json').read_text())
    originals = {Path(r['source_filename']).stem: r for r in source['records']}
    with (ROOT / 'reports/sonar_analysis_full/split_integrity_manifest.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    matched = mismatched = 0
    groups = defaultdict(Counter)
    for row in rows:
        stem = re.sub(r'_bmp$', '', row['source_stem'])
        if stem in originals:
            matched += 1
            mismatched += originals[stem]['split'] != row['split']
        groups[stem.rsplit('_', 1)[0]][row['split']] += 1
    archives = list((ROOT / 'data/external_redownload_audit_20260909/roboflow_cylider2/raw').glob('*.zip'))
    inventory = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            inventory[archive.name] = [n for n in z.namelist() if not n.endswith('/') and not re.search(r'/(images|labels)/', n)]
    x = spec['architecture_space']
    baseline = yaml.safe_load(frozen.read_text(encoding='utf-8'))
    assert [x['anchors'][p] for p in ['P3', 'P4', 'P5']] == baseline['detection']['anchor_sizes']
    assert x['anchors']['P2'] == x['anchors']['P3']
    assert x['head']['prediction_channels'] == x['head']['anchors_per_location'] * (x['num_classes'] + 5)
    assert not spec['executable_search_config']
    assert spec['fpga_compatibility']['active_hardware_constraints'] is None
    for stage, original in zip(x['stages'], manifest['architecture']['stages']):
        block = original['blocks'][0]
        assert stage['channels'] == original['channels'] and stage['stride'] == original['stride']
        assert stage['op'] == block['op'] and stage['depth'] == original['depth']
        if stage['op'] == 'mbconv':
            assert stage['expansion'] == block['expand_ratio'] and block['kernel_size'] == 3
    cases, costs = [], []
    for combo in x['output_combinations']:
        levels = combo['levels']
        features = [x['features'][p] for p in levels]
        assert [f['stage'] for f in features] == sorted({f['stage'] for f in features})
        assert features[-1]['stage'] == combo['last_active_stage']
        assert all(b['stride'] == 2*a['stride'] for a,b in zip(features, features[1:]))
        active = [s for s in x['stages'] if s['index'] <= combo['last_active_stage']]
        variable = [s for s in active if s['op'] == 'mbconv']
        assert all(s['kernel_choices'] == [3, 5] and s['expansion'] == 3 for s in variable)
        assert len(levels) == len([x['anchors'][p] for p in levels])
        assert all(len(x['anchors'][p]) == 3 for p in levels)
        for kernels in itertools.product(*[s['kernel_choices'] for s in variable]):
            cases.append((combo['id'], kernels))
        # Batch-1 Conv/Linear MACs, baseline k=3; no execution or architecture scoring.
        macs = 320*320*32*9
        cin, side = 32, 320
        for stage in active:
            cout, stride = stage['channels'], stage['stride']
            out = side // stride
            if stage['op'] == 'conv':
                macs += out*out*cin*cout*stage['kernel']**2
            elif stage['op'] == 'mbconv':
                expanded = 3*cin
                macs += side*side*cin*expanded + out*out*expanded*(9+cout)
            else:
                assert cin == cout and stride == 1
            cin, side = cout, out
        backbone_macs = macs
        for f in features:
            macs += f['spatial']**2*(f['channels']*96 + 2*96*96*9 + 96*21)
        elements = sum(96*f['spatial']**2 for f in features)
        costs.append(dict(combination=combo['id'], active_kernel_variables=len(variable), configurations=2**len(variable),
                          backbone_macs_k3=backbone_macs, total_macs_k3=macs, flops_2macs_k3=2*macs,
                          one_neck_feature_set_elements=elements, fp32_feature_set_bytes=4*elements,
                          int8_feature_set_bytes=elements, raw_anchor_predictions=sum(3*f['spatial']**2 for f in features)))
    assert len(cases) == len(set(cases)) == x['unique_architecture_count'] == 144
    assert costs[0]['total_macs_k3'] == 1961715200
    sys.path.insert(0, str(ROOT / 'src'))
    from hwnas_fpga.models.builder import DMDCConvSonarBlock, CFEMSonarBlock
    from hwnas_fpga.deploy.export import get_model_info
    import torch
    torch.manual_seed(42)
    torch.set_num_threads(2)
    operators = {name: get_model_info(cls(32, 32, stride=1), (1, 32, 80, 80))
                 for name, cls in [('dmdc_conv_sonar', DMDCConvSonarBlock), ('cfem_sonar', CFEMSonarBlock)]}
    # Existing source manifest remains authoritative before and after lightweight eval-only hooks.
    assert all(digest(ROOT / p.replace('\\', '/')) == sha for p, sha in manifest['source_sha256'].items())
    lut = json.loads((ROOT / 'hls_lut_builder/results/detection_multiscale_640_zcu104/formal_lut_status.json').read_text())
    report = dict(config_sha256=digest(args.config), baseline_config_sha256=digest(frozen),
                  baseline_source_files_checked=len(manifest['source_sha256']), baseline_unchanged=True,
                  split_case='B_filename_proxy_only', current_images=len(rows), original_filename_matches=matched,
                  original_split_mismatches=mismatched, original_metadata_fields=sorted(set().union(*(r.keys() for r in source['records']))),
                  filename_groups=dict(groups), archive_non_image_label_entries=inventory,
                  legal_architectures=len(cases), p2_static_costs=costs, historical_operator_estimates=operators,
                  detection_lut_status_counts=dict(Counter(e['status'] for e in lut['entries'])),
                  test_performance_accessed=False, synthesis_executed=False, search_executed=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audit_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (args.output / 'p2_static_costs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(costs[0]))
        writer.writeheader()
        writer.writerows(costs)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
