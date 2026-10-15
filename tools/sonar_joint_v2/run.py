"""Isolated v2: budget preflight -> budget-level PTQ -> joint promotion.

Only ``run`` launches training. ``plan`` writes an isolated hardware-screening
report. Never resume v1 as v2 and never interpret operator peaks as routed fit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sonar_search import run as legacy
from tools.sonar_joint_v2 import policy
from tools.sonar_search_finalize.run import bounded

CONFIG = ROOT / 'configs/search/sonar_joint_v2.yaml'
sha, digest, save, read = legacy.sha, legacy.digest, legacy.save, legacy.read
FIELDS = ('architecture_id', 'architecture_hash', 'graph_hash', 'mapping_hash',
          'quant_contract_hash', 'profile', 'mapping_policy', 'status',
          'measurement_valid', 'timing_eligible', 'Latency', 'LUT', 'DSP', 'BRAM',
          'DSP_sum', 'LUT_sum', 'BRAM18_sum', 'measurement_ids', 'latency_scope',
          'resource_scope', 'shell_reservation', 'target_device', 'clock_constraint_ns')


def load_config(path, require_budget=False):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    policy.validate_config(cfg, require_budget)
    if cfg['space'] != 'configs/search_space_sonar_v0_1.yaml' or cfg['baseline'] != 'configs/baseline_sonar_frozen.yaml':
        raise ValueError('Only the existing immutable detector/data contract is supported')
    validate_budget_sources(cfg)
    return cfg


def validate_budget_sources(cfg):
    """Check units and fixed engineering ceilings against retained primary records."""
    budget = cfg['hardware_budget']
    source = read(ROOT / cfg['deployment_budget']['source'])
    resource = source['resource_gate']
    expected = dict(LUT=resource['lut_max'], DSP=resource['dsp_max'], BRAM=resource['bram18_max'])
    if budget.get('resource_source') != cfg['deployment_budget']['source']:
        raise ValueError('Screen and deployment resource provenance differ')
    for key, maximum in expected.items():
        value = budget['limits'][key]
        if value is not None and value > maximum:
            raise ValueError('Screen budget exceeds preregistered resource ceiling: ' + key)
    baseline = yaml.safe_load((ROOT / budget['latency_source']).read_text(encoding='utf-8'))
    latency = budget['limits']['Latency']
    if latency is not None and latency > baseline['constraints']['max_latency_ms'] * 1000:
        raise ValueError('Screen latency exceeds retained detection-search ceiling')
    deployment = cfg['deployment_budget']
    if deployment['limits'] != dict(LUT=expected['LUT'], DSP=expected['DSP'], BRAM18=expected['BRAM']):
        raise ValueError('Deployment resource gate differs from preregistration')
    timing = source['timing_gate']; board = source['board_performance_gate']
    for key, value in dict(wns_ns_min=timing['wns_ns_min'], tns_ns_required=timing['tns_ns_required'],
                          board_e2e_ms_max=board['latency_board_ms_max'],
                          dynamic_power_w_max=board['dynamic_power_w_max'],
                          dynamic_energy_mj_max=board['dynamic_energy_mj_per_op_max']).items():
        if deployment[key] != value:
            raise ValueError('Deployment gate differs from preregistration: ' + key)


def source_inputs(config_path, cfg):
    config_path = Path(config_path).resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('Config must be inside the repository for source freezing')
    files = legacy.inputs(config_path)
    paths = list((ROOT / 'tools/sonar_joint_v2').glob('*.py'))
    # H4 dynamically imports H3 validation/contract code; freeze those actual
    # dependencies as well as the primary measured reports.
    paths.extend((ROOT / 'tools/fpga_h3').glob('*.py'))
    paths.append(ROOT / 'hls_lut_builder/scripts/common.py')
    paths.extend([ROOT / 'tools/sonar_search_finalize/run.py',
                  ROOT / cfg['hardware_budget']['latency_source'],
                  ROOT / cfg['deployment_budget']['source']])
    for path in paths:
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(sorted(files.items()))


def preflight(cfg, db_factory=legacy.CostDatabase):
    db = db_factory()
    rows = []
    for aid in sorted(legacy.architectures()):
        for profile in cfg['quant_profiles']:
            for mapping in cfg['mapping_policies']:
                looked = db.estimate_cost(aid, profile, mapping)
                row = {key: looked.get(key) for key in FIELDS}
                # Discard historical Accuracy/admission entirely: this campaign
                # supplies its own actual checkpoint-level PTQ scores.
                row.update(architecture_id=aid, profile=profile, mapping_policy=mapping,
                           deployment='NOT_CLAIMED', board_measured=False)
                rows.append(row)
    screened = policy.filter_costs(rows, cfg['hardware_budget'])
    eligible = sorted({r['architecture_id'] for r in screened['rows']})
    return dict(version=cfg['version'], rows=rows, eligible_rows=screened['rows'],
                rejected=screened['rejected'], architectures=len(legacy.architectures()),
                triples=len(rows), looked_up=sum(r['status'] == 'LOOKED_UP' for r in rows),
                eligible_architectures=eligible, eligible_triples=len(screened['rows']),
                status=screened['status'], hardware_budget=cfg['hardware_budget'],
                platform=legacy.PLATFORM, test_evaluated=False,
                training_started=False, deployment='NOT_CLAIMED')


def plan(cfg, out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Plan requires a new empty output directory')
    snapshot = preflight(cfg)
    save(out / 'cost_snapshot.json', snapshot)
    save(out / 'plan.json', dict(protocol=cfg, status=snapshot['status'],
         architectures=snapshot['architectures'], triples=snapshot['triples'],
         eligible_architectures=len(snapshot['eligible_architectures']),
         eligible_triples=snapshot['eligible_triples'],
         max_training_epoch_equivalents=11520,
         max_ptq_profile_evaluations=(144 + 108 + 72 + 6 + 12 * 2) * 3,
         calibration_split='train', evaluation_split='valid', test_evaluated=False,
         training_started=False, deployment='NOT_CLAIMED',
         additional_quantization_cost='PTQ now runs at each budget; not a wall-time saving claim'))
    return snapshot


def freeze(config_path, out):
    cfg = load_config(config_path, require_budget=True)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Freeze requires a new empty campaign directory')
    snapshot = preflight(cfg)
    if not snapshot['eligible_architectures']:
        raise ValueError('No hardware-screened architectures; refuse training freeze')
    dataset = legacy.data_binding(cfg)
    files = source_inputs(config_path, cfg)
    save(out / 'cost_snapshot.json', snapshot)
    with zipfile.ZipFile(out / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for rel in files:
            archive.write(ROOT / rel, rel)
    archive_sha = sha(out / 'source_snapshot.zip')
    save(out / 'snapshot.json', dict(sha256=archive_sha))
    save(out / 'freeze.json', dict(protocol=cfg, files=files, dataset=dataset,
         config_path=Path(config_path).resolve().relative_to(ROOT).as_posix(),
         config_hash=digest(cfg), data_hash=digest(dataset),
         code_hash=digest(files), cost_snapshot_sha256=sha(out / 'cost_snapshot.json'),
         source_snapshot_sha256=archive_sha,
         platform=legacy.PLATFORM, test_evaluated=False, training_started=False,
         deployment='NOT_CLAIMED', predecessor='sonar_joint_v1 preserved; no historical checkpoint import'))
    return snapshot


def verify(config_path, out):
    cfg = load_config(config_path, require_budget=True)
    binding = read(out / 'freeze.json')
    if binding['protocol'] != cfg or binding['config_hash'] != digest(cfg):
        raise ValueError('Frozen v2 protocol changed')
    files = source_inputs(config_path, cfg)
    if files != binding['files'] or digest(files) != binding['code_hash']:
        raise ValueError('Frozen source/cost provenance changed')
    dataset = legacy.data_binding(cfg)
    if dataset != binding['dataset'] or digest(dataset) != binding['data_hash']:
        raise ValueError('Dataset drift')
    if sha(out / 'cost_snapshot.json') != binding['cost_snapshot_sha256']:
        raise ValueError('Cost snapshot drift')
    if (sha(out / 'source_snapshot.zip') != binding['source_snapshot_sha256']
            or read(out / 'snapshot.json')['sha256'] != binding['source_snapshot_sha256']):
        raise ValueError('Source archive drift')
    with zipfile.ZipFile(out / 'source_snapshot.zip') as archive:
        if sorted(archive.namelist()) != sorted(files):
            raise ValueError('Source archive membership mismatch')
        import hashlib
        for rel, expected in files.items():
            if hashlib.sha256(archive.read(rel)).hexdigest() != expected:
                raise ValueError('Source archive member drift: ' + rel)
    legacy.CostDatabase()  # Reverify primary reports without claiming accuracy.
    return binding


def verify_receipt(out, mode, aid, seed, budget):
    receipt = read(out / 'tasks' / f'{mode}_{aid}_{seed}_{budget}.json')
    if receipt['freeze_sha256'] != sha(out / 'freeze.json') or not receipt['outputs']:
        raise ValueError('Receipt freeze/output binding changed')
    for rel, expected in receipt['outputs'].items():
        if sha(bounded(out, rel)) != expected:
            raise ValueError('Task artifact drift: ' + rel)
    result = receipt['result']
    if result.get('architecture_id') != aid or result.get('seed') != seed or result.get('budget') != budget or result.get('test_evaluated') is not False:
        raise ValueError('Task candidate/seed/budget binding changed')
    checkpoint = f'training/{aid}/seed_{seed}/best_{budget}.pt'
    if result.get('checkpoint') != checkpoint or receipt['outputs'].get(checkpoint) != result.get('checkpoint_sha256'):
        raise ValueError('Checkpoint receipt mismatch')
    if mode == 'quant':
        rel = f'quantization/{aid}/seed_{seed}/budget_{budget}/result.json'
        if rel not in receipt['outputs'] or read(out / rel) != result or result.get('split') != 'valid':
            raise ValueError('Quantization result receipt mismatch')
        train = verify_receipt(out, 'train', aid, seed, budget)
        if train['checkpoint_sha256'] != result['checkpoint_sha256']:
            raise ValueError('PTQ did not evaluate the bound budget checkpoint')
        if result['freeze_sha256'] != receipt['freeze_sha256']:
            raise ValueError('PTQ freeze mismatch')
        cal = 'calibration/calibration_manifest.json'
        if receipt['outputs'].get(cal) != result['calibration_sha256']:
            raise ValueError('Calibration receipt mismatch')
        manifest = read(out / cal)
        binding = read(out / 'freeze.json')
        if (manifest.get('split') != 'train' or manifest.get('entries') != binding['dataset']['train'][:64]
                or any(manifest.get(k) is not False for k in ('test_evaluated', 'valid_used_for_selection', 'augmentation'))):
            raise ValueError('Calibration frozen-train identity mismatch')
        for q in policy.PROFILES:
            profile = f'quantization/{aid}/seed_{seed}/budget_{budget}/{q}_profile.json'
            if receipt['outputs'].get(profile) != result['profiles'][q]['profile_sha256']:
                raise ValueError('Quantization profile receipt mismatch')
            record = read(out / profile)
            if (record.get('precision_profile') != q
                    or any(record.get('binding', {}).get(k) != result[k] for k in
                           ('architecture_id', 'seed', 'budget', 'checkpoint', 'checkpoint_sha256',
                            'freeze_sha256', 'calibration_sha256', 'split', 'test_evaluated'))):
                raise ValueError('Quantization profile checkpoint/budget binding mismatch')
            from tools.sonar_joint_v2.worker import _check_profile
            _check_profile(record, q)
            policy.score(result['profiles'][q]['mAP50_95'])
        policy.score(result['fp32']['mAP50_95'])
    else:
        policy.score(result['mAP50_95'])
    return result


def task(out, mode, aid, seed, budget, device):
    receipt = out / 'tasks' / f'{mode}_{aid}_{seed}_{budget}.json'
    if not receipt.exists():
        log = out / 'logs' / f'{mode}_{aid}_{seed}_{budget}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as stream:
            subprocess.run([sys.executable, str(Path(__file__).with_name('worker.py')), mode,
                '--output', str(out), '--architecture', aid, '--seed', str(seed),
                '--budget', str(budget), '--device', device], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
    return verify_receipt(out, mode, aid, seed, budget)


def confirmed_points(evaluations, costs, cfg):
    points, rejected = [], []
    for aid, by_seed in sorted(evaluations.items()):
        if sorted(by_seed) != cfg['confirmation_seeds']:
            raise ValueError('Three paired confirmation seeds required')
        runs = [by_seed[s] for s in cfg['confirmation_seeds']]
        # Rebuild from this campaign, never the old B1/C/D accuracy allowlist.
        for seed, r in zip(cfg['confirmation_seeds'], runs):
            if (r['architecture_id'] != aid or r['budget'] != 80 or r.get('seed') != seed
                    or r.get('split') != 'valid' or r.get('test_evaluated') is not False):
                raise ValueError('Confirmation identity mismatch')
            if set(r['profiles']) != set(cfg['quant_profiles']):
                raise ValueError('Incomplete confirmation precision profiles')
        staged = policy.joint_points([runs[0]], costs, cfg['hardware_budget'])
        fp = [r['fp32']['mAP50_95'] for r in runs]
        uniform = [r['profiles']['W8A8']['mAP50_95'] for r in runs]
        for q in cfg['quant_profiles']:
            values = [r['profiles'][q]['mAP50_95'] for r in runs]
            for value in fp + uniform + values:
                policy.score(value)
            if not legacy.admit(fp, uniform, values, q, cfg['admission']):
                rejected.append(dict(architecture_id=aid, profile=q, reason='PAIRED_ACCURACY_GATE', values=values))
                continue
            mean = sum(values) / 3
            for point in staged:
                if point['profile'] != q:
                    continue
                point = deepcopy(point)
                point.update(Accuracy=mean, mean_mAP50_95=mean,
                     std_mAP50_95=(sum((v - mean)**2 for v in values) / 2)**.5,
                     checkpoint_sha256=[r['checkpoint_sha256'] for r in runs],
                     quant_profile_sha256=[r['profiles'][q]['profile_sha256'] for r in runs],
                     calibration_sha256=[r['calibration_sha256'] for r in runs],
                     n_seeds=3, accuracy_admission='PAIRED_VALIDATION_PASS',
                     confirmation_seeds=cfg['confirmation_seeds'], deployment='NOT_CLAIMED')
                points.append(point)
    return dict(points=points, rejected=rejected, pareto=policy.fronts(points)[0] if points else [],
                scope='confirmed_valid_PTQ_accuracy_with_screened_operator_costs',
                test_evaluated=False, deployment='NOT_CLAIMED')


def execute(config_path, out, device, resume=False):
    if (out / 'state.json').exists() and not resume:
        raise FileExistsError('Run exists; use --resume')
    binding = verify(config_path, out)
    cfg = binding['protocol']
    costs = read(out / 'cost_snapshot.json')['rows']
    selected = sorted({r['architecture_id'] for r in policy.filter_costs(costs, cfg['hardware_budget'])['rows']})
    audit_ids, last = [], None
    try:
        for stage, budget in enumerate(cfg['budgets']):
            results = []
            for aid in selected:
                save(out / 'state.json', dict(status='RUNNING', stage='quantized_screening',
                     budget=budget, architecture_id=aid, completed=len(results), total=len(selected)))
                task(out, 'train', aid, 42, budget, device)
                results.append(task(out, 'quant', aid, 42, budget, device))
            points = policy.joint_points(results, costs, cfg['hardware_budget'])
            count = cfg['populations'][stage + 1] if stage < 2 else cfg['shortlist']
            last = policy.promote(points, count, cfg['promotion'])
            save(out / f'stage_{budget}.json', dict(evaluations=results, promotion=last))
            if not last['selected']:
                raise ValueError('No quantized budget-feasible promotion candidates')
            if stage < 2:
                removed = set(selected) - set(last['selected'])
                for macro in ('P3_P4_P5', 'P2_P3_P4', 'P2_P3_P4_P5'):
                    pool = [a for a in removed if a.split('_s')[0] == macro]
                    if pool:
                        audit_ids.append(min(pool, key=lambda a: digest([42, stage, a])))
            selected = last['selected']
        audit_results = []
        for aid in sorted(set(audit_ids)):
            task(out, 'train', aid, 42, 80, device)
            audit_results.append(task(out, 'quant', aid, 42, 80, device))
        audit_points = policy.joint_points(audit_results, costs, cfg['hardware_budget'])
        # Conservative existing top5 loss guard, evaluated with PTQ rather than FP32.
        best = sorted((max(p['Accuracy'] for p in last['points'] if p['architecture_id'] == aid)
                       for aid in {p['architecture_id'] for p in last['points']}), reverse=True)
        threshold = best[min(4, len(best) - 1)]
        failures = sorted({p['architecture_id'] for p in audit_points if p['Accuracy'] > threshold})
        save(out / 'rejection_audit.json', dict(evaluations=audit_results, points=audit_points,
             quantized_top5_threshold=threshold, violations=failures, test_evaluated=False))
        if failures:
            raise ValueError('Quantized rejection audit exceeded promoted top5; selection requires review')
        save(out / 'shortlist.json', dict(architectures=selected, scope='screened_promoted_population_not_global_optimum'))
        evaluations = {}
        for aid in selected:
            evaluations[aid] = {}
            for seed in cfg['confirmation_seeds']:
                task(out, 'train', aid, seed, 80, device)
                evaluations[aid][seed] = task(out, 'quant', aid, seed, 80, device)
        report = confirmed_points(evaluations, costs, cfg)
        save(out / 'joint_pareto.json', report)
        queue = [dict(architecture_id=p['architecture_id'], profile=p['profile'], mapping_policy=p['mapping_policy'],
                      graph_hash=p['graph_hash'], mapping_hash=p['mapping_hash'],
                      quant_contract_hash=p['quant_contract_hash'], freeze_sha256=sha(out / 'freeze.json'),
                      checkpoint_sha256=p['checkpoint_sha256'][0], quant_profile_sha256=p['quant_profile_sha256'][0],
                      calibration_sha256=p['calibration_sha256'][0], hardware_seed=42,
                      status='AWAITING_MATCHED_INTEGER_EXPORT_ROUTE_BOARD_EVIDENCE',
                      deployment_budget=cfg['deployment_budget']) for p in report['pareto']]
        save(out / 'deployment_queue.json', dict(points=queue, launch_hardware=False,
             stages=['integer_export', 'integer_parity', 'full_route', 'valid99_board_accuracy', 'e2e_latency', 'power_energy'],
             preregistration=read(ROOT / cfg['deployment_budget']['source']), deployment='NOT_CLAIMED'))
        save(out / 'state.json', dict(status='COMPLETE_SCREENING' if report['points'] else 'COMPLETE_NO_ADMITTED_POINTS',
             test_evaluated=False, deployment='NOT_CLAIMED'))
    except Exception as error:
        save(out / 'state.json', dict(status='FAILED_RESUMABLE', error=str(error), deployment='NOT_CLAIMED'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'freeze', 'run'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out == (ROOT / 'results/sonar_joint_v1').resolve() or 'sonar_joint_v1' in out.parts:
        raise ValueError('V1 output is immutable to this runner')
    cfg = load_config(args.config, args.mode != 'plan')
    if args.mode == 'plan':
        result = plan(cfg, out)
    elif args.mode == 'freeze':
        result = freeze(args.config, out)
    else:
        with legacy.run_lock(out):
            execute(args.config, out, args.device, args.resume)
        return
    print(json.dumps({k: result[k] for k in ('status', 'architectures', 'triples', 'looked_up', 'eligible_triples')}, indent=2))


if __name__ == '__main__':
    main()
