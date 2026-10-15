"""Prospective v2 amendment: paired endpoints, coverage safeguards and image CIs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import statistics

from tools.sonar_fcos_pilot.common import (ROOT, environment, read_json, require_manifest,
    resolve_path, sha256_file, utc_now, write_json)

DEFAULT_CONFIG = ROOT / 'configs/sonar_fcos_quality_20261012_v2.json'
PREREGISTRATION_SHA256 = '888cc683f3263d8e8287a608e45cba2b517ef0ddf38b66f19333771d53cfe374'
TRAINING_KEYS = ('image_size', 'epochs', 'batch_size', 'gradient_accumulation',
    'learning_rate', 'momentum', 'weight_decay', 'learning_rate_drop_epochs',
    'warmup_epochs', 'warmup_start_factor', 'amp', 'amp_initial_scale', 'workers',
    'torch_threads', 'pretrained', 'diagnostic_score_threshold', 'device')


def load_config(path=DEFAULT_CONFIG):
    experiment = read_json(path)
    base_path = Path(experiment['baseline_config'])
    baseline = read_json(base_path)
    if sha256_file(base_path) != 'c1bbeb42ac23ae8be243a1974194acac746504776a6fa19e869759df243aeb8b':
        raise ValueError('Historical AAA baseline config changed')
    from tools.sonar_fcos_quality.model import QUALITY_PARAMETERS
    if experiment['quality'] != QUALITY_PARAMETERS:
        raise ValueError('Quality model hyperparameters differ from frozen config')
    if experiment['variants'] != ['AAA', 'QFL_DFL'] or experiment['seeds'] != [42, 43, 44]:
        raise ValueError('Only the authorized two variants and three seeds are allowed')
    if experiment['epochs'] != 30 or experiment['max_formal_runs'] != 6:
        raise ValueError('The authorized experiment is exactly six 30-epoch runs')
    if experiment['checkpoint_selection'] != 'earliest_best_validation_coco_map50_95':
        raise ValueError('Checkpoint selection is frozen')
    if experiment['acceptance'] != {'mean_paired_ap75_gain_pp_min': 2.0, 'mean_nms_first_loss_relative_reduction_min': .25}:
        raise ValueError('User acceptance criteria changed')
    prereg_path = resolve_path(experiment['preregistration_path'])
    if sha256_file(prereg_path) != PREREGISTRATION_SHA256:
        raise ValueError('Frozen v2 preregistration bytes changed')
    prereg = read_json(prereg_path)
    if (prereg['protocol_id'] != 'FCOS_UATD_PREREG_V2' or prereg['seeds'] != [42,43,44]
            or prereg['primary_checkpoint_views'] != ['best','epoch30']
            or prereg['bootstrap']['resamples'] != 2000 or prereg['bootstrap']['seed'] != 20261012):
        raise ValueError('Frozen v2 preregistration changed')
    config = {**baseline, **experiment, 'config_path': str(Path(path).resolve()),
              'config_sha256': sha256_file(path), 'baseline_config_sha256': sha256_file(base_path)}
    config.pop('recovery_parent', None)
    for key in TRAINING_KEYS:
        if config[key] != baseline[key]:
            raise ValueError('Baseline training setting changed: ' + key)
    manifest_path, manifest = require_manifest(config)
    if sha256_file(manifest_path) != config['expected_manifest_sha256']:
        raise ValueError('Historical UATD manifest differs')
    validation = [r for r in manifest['samples'] if r['split'] == 'val']
    if len(validation) != 1517 or sum(len(r['boxes']) for r in validation) != 2692:
        raise ValueError('Not the same 1517-image, 2692-object validation split')
    return config


def source_binding(config):
    paths = list((ROOT / 'tools/sonar_fcos_pilot').glob('*.py'))
    paths += list((ROOT / 'tools/sonar_fcos_quality').glob('*.py'))
    paths += list((ROOT / 'tools/sonar_fcos_quality_v2').glob('*.py'))
    return {'config_sha256': config['config_sha256'],
            'baseline_config_sha256': config['baseline_config_sha256'],
            'data_manifest_sha256': config['expected_manifest_sha256'],
            'preregistration_sha256': sha256_file(resolve_path(config['preregistration_path'])),
            'parent_execution_manifest_sha256': config['parent_execution_manifest_sha256'],
            'pretrained_checkpoint_sha256': sha256_file(resolve_path(config['weight_cache']) / 'checkpoints' /
                                                       'fcos_resnet50_fpn_coco-99b0c9b7.pth'),
            'sources': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}}


def verify_development_inputs(manifest):
    for row in manifest['samples']:
        if row['split'] in ('train', 'val'):
            for field, hash_field in [('image_path', 'sha256'), ('annotation_path', 'annotation_sha256')]:
                if hash_field not in row or sha256_file(row[field]) != row[hash_field]:
                    raise RuntimeError('Development input changed: ' + str(row.get(field)))


def prepare(config):
    manifest_path, manifest = require_manifest(config)
    verify_development_inputs(manifest)
    output = resolve_path(config['output_dir'])
    receipt = {'status': 'PASS', 'source_binding': source_binding(config),
               'environment': environment(), 'training_settings': {k: config[k] for k in TRAINING_KEYS},
               'validation_images': 1517, 'validation_objects': 2692,
               'test_evaluated': False, 'protocol_id': manifest.get('protocol_id'),
               'preregistration': read_json(resolve_path(config['preregistration_path'])),
               'limitations': ['Development validation; no verified acquisition-sequence separation',
                              'Does not establish effectiveness on the two-class lightweight route',
                              'FCOS weights archive only; C01c deferred and requires separate E0 UATD pretraining'],
               'run_order': [{'variant': v, 'seed': s} for s in config['seeds'] for v in config['variants']]}
    path = output / 'execution_manifest.json'
    if path.exists() and read_json(path) != receipt:
        raise RuntimeError('Campaign already prepared with different source/config/environment')
    write_json(path, receipt)
    summarize(config)
    return receipt


def view_decision(rows):
    by = {(r['variant'], r['seed']): r for r in rows}
    expected = {(v,s) for v in ('AAA','QFL_DFL') for s in (42,43,44)}
    if set(by) != expected or len(rows) != 6 or any(r['trained_epochs'] != 30 for r in rows):
        return {'status':'PENDING'}
    deltas = [by['QFL_DFL',s]['ap75']-by['AAA',s]['ap75'] for s in (42,43,44)]
    metrics = ('ap75','ap85','map','nms_first_loss','raw_no_good_box','final_no_good_box','wrong_class_fp')
    stats = {v:{m:{'mean':statistics.mean(by[v,s][m] for s in (42,43,44)),
                   'sample_sd':statistics.stdev(by[v,s][m] for s in (42,43,44))}
                for m in metrics} for v in ('AAA','QFL_DFL')}
    a,q=stats['AAA'],stats['QFL_DFL']
    denominator=a['nms_first_loss']['mean']
    reduction=(denominator-q['nms_first_loss']['mean'])/denominator if denominator>0 else None
    gates={'ap75':statistics.mean(deltas)>=2.,
           'nms':reduction is not None and reduction>=.25,
           'final_coverage':q['final_no_good_box']['mean']<a['final_no_good_box']['mean'],
           'raw_coverage':q['raw_no_good_box']['mean']<=a['raw_no_good_box']['mean']}
    return {'status':'PASS' if all(gates.values()) else 'FAIL','gates':gates,
            'paired_ap75_gains_pp':deltas,'all_three_seed_ap75_positive':all(d>0 for d in deltas),
            'mean_ap75_gain_pp':statistics.mean(deltas),'nms_relative_reduction':reduction,
            'group_statistics':stats}


def paired_decision(rows):
    keys=[(r['view'],r['variant'],r['seed']) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate endpoint/run rows')
    views={v:view_decision([r for r in rows if r['view']==v]) for v in ('best','epoch30','mean26_30')}
    primary=[views[v]['status'] for v in ('best','epoch30')]
    if 'PENDING' in primary:
        status='PENDING'
    else:
        count=primary.count('PASS')
        status='PASS' if count==2 else 'NON_ROBUST' if count==1 else 'FAIL'
    return {'status':status,'views':views,'mandatory_views':['best','epoch30'],
            'stability_view_is_supplementary':True,'ci_is_auxiliary_not_an_additional_gate':True}


def _verify(artifacts):
    for a in artifacts.values():
        if sha256_file(a['path'])!=a['sha256']:
            raise RuntimeError('Committed artifact changed: '+a['path'])


def summarize(config):
    from .analysis import bootstrap_suite
    output=resolve_path(config['output_dir'])
    binding=source_binding(config)
    rows,summaries,workpoints=[],[],[]
    for seed in config['seeds']:
        for variant in config['variants']:
            path=output/'runs'/f'{variant}_seed{seed}'/'summary.json'
            if not path.exists():
                continue
            item=read_json(path)
            if item['source_binding']!=binding or item['status']!='COMPLETE' or item['trained_epochs']!=30:
                raise RuntimeError('Invalid completed v2 receipt')
            _verify(item['best_artifacts'])
            _verify(item['evaluation_manifests'])
            if set(item['retained_epochs'])!={str(e) for e in range(26,31)}:
                raise RuntimeError('Incomplete late-epoch retention')
            for artifacts in item['retained_epochs'].values():
                _verify(artifacts)
            for name,key in [('best','best'),('epoch30','epoch30'),('mean26_30','epochs26_30_mean')]:
                endpoint=item['endpoints'][key]
                m,d=endpoint['metrics'],endpoint['diagnostics']
                rows.append({'view':name,'variant':variant,'seed':seed,'trained_epochs':30,
                    'selected_epoch':item['best_epoch'] if name=='best' else 30 if name=='epoch30' else '26-30',
                    **{k:m[k] for k in ('ap75','ap85','map')},
                    **{k:d[k] for k in ('nms_first_loss','raw_no_good_box','final_no_good_box','wrong_class_fp')}})
                if name!='mean26_30':
                    workpoints.append({'view':name,'variant':variant,'seed':seed,
                        'fixed_score_025':endpoint['fixed_score_025'],'matched_fppi_05':endpoint['workpoint']})
            for epoch,artifacts in item['retained_epochs'].items():
                if epoch=='30':
                    continue  # Already present as the primary epoch30 endpoint.
                endpoint=read_json(artifacts['evaluation']['path'])
                workpoints.append({'view':'epoch'+epoch,'variant':variant,'seed':seed,
                    'fixed_score_025':endpoint['fixed_score_025'],'matched_fppi_05':endpoint['workpoint']})
            summaries.append(item)
    decision=paired_decision(rows)
    result={'updated_at':utc_now(),'rows':rows,'decision':decision,'completed_runs':len(summaries),
        'source_binding':binding,'test_evaluated':False,'workpoints':workpoints,
        'historical_reference_only':config['historical_reference_only'],'confidence_intervals':{},
        'claim_boundaries':read_json(resolve_path(config['preregistration_path']))['claim_boundaries']}
    output.mkdir(parents=True,exist_ok=True)
    if len(summaries)==6:
        _,manifest=require_manifest(config)
        records=[r for r in manifest['samples'] if r['split']=='val']
        for view in ('best','epoch30','mean26_30'):
            prediction_artifacts={}
            for item in summaries:
                key=(item['variant'],item['seed'])
                prediction_artifacts[key]=([item['best_artifacts']['predictions']] if view=='best' else
                    [item['retained_epochs']['30']['predictions']] if view=='epoch30' else
                    [item['retained_epochs'][str(e)]['predictions'] for e in range(26,31)])
            import hashlib
            identity={'source_binding':binding,'view':view,
                'prediction_sha256':{f'{v}_{s}':[a['sha256'] for a in values]
                                     for (v,s),values in prediction_artifacts.items()}}
            cache_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            ci_path=output/'confidence_intervals'/f'{view}.json'
            if ci_path.exists():
                ci=read_json(ci_path)
                if ci['input_sha256']!=cache_sha:
                    raise RuntimeError('Bootstrap cache belongs to different predictions')
            else:
                write_json(output/'analysis_status.json',{'status':'BOOTSTRAP_RUNNING','view':view,'resamples':2000,
                                                         'updated_at':utc_now()})
                predictions={key:([read_json(a['path']) for a in values] if view=='mean26_30'
                                  else read_json(values[0]['path'])) for key,values in prediction_artifacts.items()}
                ci={'input_sha256':cache_sha,'identity':identity,
                    'result':bootstrap_suite(records,predictions,len(manifest['classes']),n_resamples=2000,random_seed=20261012)}
                write_json(ci_path,ci)
            if abs(ci['result']['mean_paired_ap75_gain_pp']-decision['views'][view]['mean_ap75_gain_pp'])>1e-8:
                raise RuntimeError('Saved endpoint AP75 differs from bootstrap prediction-cache point estimate')
            result['confidence_intervals'][view]=ci
        selected=max(summaries,key=lambda s:s['validation']['map'])
        package=output/'pretrained'
        package.mkdir(exist_ok=True)
        src=Path(selected['best_artifacts']['checkpoint']['path'])
        destination=package/'uatd_fcos_best.pt'
        if destination.exists() and sha256_file(destination)!=sha256_file(src):
            raise RuntimeError('Do not replace an exported archive')
        if not destination.exists():
            shutil.copy2(src,destination)
        write_json(package/'pretrained_manifest.json',{'checkpoint':str(destination),'sha256':sha256_file(destination),
            'variant':selected['variant'],'seed':selected['seed'],'epoch':selected['best_epoch'],
            'selection':'maximum best-validation mAP across six runs; archive only, not efficacy gate',
            'source_binding':binding,'model_contract':selected['model'],'decision':decision['status'],
            'transfer_status':'FCOS_ARCHIVE_ONLY_NOT_E0_PRETRAINING','C01c':'DEFERRED_NOT_STARTED','test_evaluated':False})
        write_json(output/'analysis_status.json',{'status':'COMPLETE','updated_at':utc_now()})
    write_json(output/'branch_results.json',result)
    columns=('view','variant','seed','trained_epochs','selected_epoch','ap75','ap85','map',
             'nms_first_loss','raw_no_good_box','final_no_good_box','wrong_class_fp')
    with (output/'branch_results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    write_json(output/'workpoint_diagnostics.json',{'source_binding':binding,'rows':workpoints})
    lines=['# 实验2 FCOS / UATD 结果（预注册v2）','',
        f"状态：{decision['status']}；完整运行{len(summaries)}/6。",'',
        'best与固定第30轮都满足AP75、NMS与两个候选覆盖防护条件才为PASS；仅一个口径达标为不稳健。','',
        '|口径|方案|seed|轮次|AP75|AP85|mAP50–95|NMS首次丢失|原始无好框|最终无好框|错类FP(.25)|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|{view}|{variant}|{seed}|{selected_epoch}|{ap75:.4f}|{ap85:.4f}|{map:.4f}|{nms_first_loss}|{raw_no_good_box}|{final_no_good_box}|{wrong_class_fp}|'.format(**r))
    for v,d in decision['views'].items():
        lines+=['',f"**{v}：{d['status']}**"]
        if d['status']!='PENDING':
            lines += [f"AP75配对均值增量：{d['mean_ap75_gain_pp']:.4f}个百分点；三个seed差值：{d['paired_ap75_gains_pp']}。",
                      f"NMS相对下降：{d['nms_relative_reduction']}；四项条件：{d['gates']}。"]
        if v in result['confidence_intervals']:
            ci=result['confidence_intervals'][v]['result']
            lines += [f"图像配对bootstrap 95%区间：[{ci['ci95_low_pp']:.4f}, {ci['ci95_high_pp']:.4f}]个百分点；"
                      f"各seed方向：{ci['seed_signs']}。"]
    lines+=['','各口径/seed均值及样本标准差、2000次配对图像bootstrap和完整抽样差值见branch_results.json。',
        '固定0.25与目标FPPI=0.5的实际阈值、达到的误检率及候选覆盖见workpoint_diagnostics.json；离散阈值不保证恰好0.5。',
        '26–30为分别评价后的算术平均，不是集成；其CI使用共同抽样索引。',
        '本结果仅限UATD开发验证集，采集序列隔离未核实；不能外推双类轻量网络有效性。',
        'FCOS权重只归档；C01c暂不执行，需另行E0＋新头UATD预训练。']
    (output/'支线结果表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    workpoint_lines=['# 分数工作点并列诊断','',
        'FPPI为每图误检数。等FPPI列采用目标0.5下不超过758个误检的最宽松整组阈值，实际值可能略低；不得据较低分数直接归因于分类改善。',
        '候选覆盖在同一原生阶段链上施加阈值，不重跑NMS/top-k；这些列是辅助诊断，主门槛仍采用原生链。','',
        '|口径|方案|seed|工作点|阈值|实际FPPI|错类FP|NMS首次丢失|原始无好框|最终无好框|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in workpoints:
        for key,label in [('fixed_score_025','固定0.25'),('matched_fppi_05','目标FPPI0.5')]:
            endpoint=row[key]; diag=endpoint['diagnostics']; counts=endpoint['counts']
            if diag['status']!='AVAILABLE':
                raise RuntimeError('Required workpoint stage evidence unavailable')
            workpoint_lines.append(f"|{row['view']}|{row['variant']}|{row['seed']}|{label}|"
                f"{endpoint['threshold']:.8g}|{counts['fppi']:.6f}|{counts['wrong_class_fp']}|"
                f"{diag['nms_first_loss']}|{diag['raw_no_good_box']}|{diag['final_no_good_box']}|")
    (output/'工作点诊断表.md').write_text('\n'.join(workpoint_lines)+'\n',encoding='utf-8')
    return result
