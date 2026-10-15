"""Offline D09 CAL -> locked host limits -> independent VAL; no board I/O.

All files are immutable inputs. A derived output is created exclusively and is
never overwritten. Raw CPU-control evidence is verified before statistics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.hardware_measurement_pilot.cpu_conditions import parse_cpus


CASE_IDS = ('pw1x1_int8_reference_high', 'dw5_int8_reference_low', 'body8_to_head16_reference_high')
CONTRACT = dict(schema='d09_host_statistics_contract_v1', case_ids=list(CASE_IDS),
    calibration_batches_per_case=3, validation_batches_per_case=3, warmup_per_batch=50,
    measured_per_batch=1000, group_size=1, sample_sd_ddof=1, percentile='linear_(n-1)*q',
    margin_multiplier='1.25', cv_floor='0.010', cv_step='0.001', cv_cap='0.050',
    tail_excess_floor='0.020', tail_excess_step='0.005', p95_over_median_cap='1.100',
    drift_floor='0.005', drift_step='0.001', drift_cap='0.020',
    transfer_cv_limit=0.05, transfer_median_relative_limit=0.01,
    thresholds_shared_across_cases=True, validation_uses_fixed_calibration_centers=True,
    outlier_deletion=False, formal_cost_eligible=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, label):
    number = float(value)
    require(not isinstance(value, bool) and math.isfinite(number) and number > 0, label+' must be finite and positive')
    return number


def utc(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, 'timestamps must carry a timezone')
    return parsed.astimezone(timezone.utc)


def stats(values):
    require(len(values) >= 2, 'at least two observations required')
    values = [positive(value, 'observation') for value in values]
    ordered = sorted(values)
    position = (len(values)-1)*0.95
    lower = int(position)
    p95 = ordered[lower]+(ordered[min(lower+1,len(values)-1)]-ordered[lower])*(position-lower)
    mean,median = statistics.mean(values),statistics.median(values)
    return dict(n=len(values),mean_ns=mean,median_ns=median,p95_ns=p95,
                sample_std_ns=statistics.stdev(values),cv=statistics.stdev(values)/mean,
                p95_over_median=p95/median,minimum_ns=ordered[0],maximum_ns=ordered[-1])


def ceil_step(value, step):
    step = Decimal(step)
    return (value/step).to_integral_value(rounding=ROUND_CEILING)*step


def load_baseline(path):
    baseline = read(path)
    require(set(baseline['cases']) == set(CASE_IDS), 'PL baseline must contain exactly the three fixed cases')
    for cid,item in baseline['cases'].items():
        source = Path(item['source_dir'])
        for name, expected in item['files'].items():
            target = (source/name).resolve()
            require(target.is_relative_to(source.resolve()) and digest(target)==expected, 'prior baseline source hash differs: '+cid+'/'+name)
        prior = read(source/'summary.json')
        row = next(row for row in prior['cases'] if row['case_id']==cid)
        require(row['scopes']['transfer']['median_ns']==item['prior_transfer_median_ns'], 'prior transfer median binding differs')
        require(item['clock_hz']==187498125, 'prior clock differs')
    return baseline


def check_cpu(record, config, label):
    require(record.get('label')==label and record.get('status')=='PASS_CPU_CONDITIONS' and
            record.get('condition_deviations')==[], 'CPU observation status/label differs: '+label)
    utc(record['timestamp_utc'])
    index,frequency=config['cpu_index'],config['frequency_khz']
    tids=record.get('thread_ids',[])
    require(len(tids)==record.get('thread_count')==1 and record.get('current_cpu')==index and
            record.get('affinities')=={str(tids[0]):[index]}, 'CPU affinity/current CPU differs: '+label)
    require(record.get('cpu_governor')==record.get('requested_mode')==config['mode'] and
            record.get('requested_frequency_khz')==frequency and index in record.get('policy_related_cpus',[]),
            'CPU governor/request differs: '+label)
    policy='/sys/devices/system/cpu/cpufreq/'+config['policy_name']+'/'
    fields=['scaling_min_freq','scaling_max_freq','scaling_cur_freq','cpuinfo_cur_freq']
    if config['mode']=='userspace': fields.append('scaling_setspeed')
    require(record.get('frequency_raw_khz')=={policy+field:str(frequency) for field in fields},
            'actual CPU frequency/bounds differ: '+label)
    policies=record.get('policies',[])
    require(len(policies)==1 and policies[0].get('path')==policy+'scaling_governor' and
            policies[0].get('actual')==config['mode'] and config['mode'] in policies[0].get('available',[]),
            'CPU governor policy evidence differs: '+label)


def check_external(record, config, label):
    require(record.get('label')==label and record.get('status')=='PASS_EXTERNAL_CONDITIONS' and
            record.get('condition_deviations')==[], 'external observation status/label differs: '+label)
    utc(record['timestamp_utc'])
    expected=config['system_conditions']
    require(record.get('boot_id')==expected['boot_id'], 'boot ID differs')
    require(all(record.get('software_versions',{}).get(key)==value for key,value in config['expected_versions'].items()),
            'actual software version differs')
    require(record.get('runtime_files')==config['runtime_files'], 'actual runtime file hashes differ')
    services=record.get('services',{})
    require(set(services)==set(expected['stopped_units']) and all(value.get('LoadState')=='loaded' and
            value.get('ActiveState')=='inactive' for value in services.values()), 'actual services not stopped')
    irqs=record.get('irq_affinity',{})
    require(set(irqs)==set(expected['irq_affinity']), 'IRQ evidence coverage differs')
    for irq,wanted in expected['irq_affinity'].items():
        actual=irqs[irq]
        require(actual.get('requested')==wanted and parse_cpus(actual['configured'])==parse_cpus(wanted) and
                parse_cpus(actual['effective'])<=parse_cpus(wanted) and config['cpu_index'] not in parse_cpus(actual['effective']),
                'actual IRQ affinity differs: '+irq)


def load_cpu_evidence(root, config_path, session, manifest, summary, sources):
    require(config_path is not None, 'an explicit bound CPU config path is required')
    config_path=Path(config_path).resolve()
    config=read(config_path)
    require(digest(config_path)==session['cpu_conditions_config_sha256'], 'actual CPU config hash differs')
    sources['cpu_conditions.json']=dict(path=config_path.as_posix(),sha256=digest(config_path))
    protocol_path=(config_path.parent/config['protocol']['path']).resolve()
    require(protocol_path.is_relative_to(config_path.parent) and digest(protocol_path)==session['d09_protocol_sha256']==config['protocol']['sha256'],
            'actual common protocol hash differs')
    protocol=read(protocol_path)
    sources['d09_protocol.json']=dict(path=protocol_path.as_posix(),sha256=digest(protocol_path))
    require(protocol.get('host_statistics_contract')==CONTRACT, 'CPU package statistics contract differs')
    for key in ('run_id','experiment_phase'):
        require(config[key]==session[key], 'actual CPU config run identity differs: '+key)
    for key in ('mode','cpu_index','policy_name','frequency_khz'):
        require(config[key]==protocol['cpu_conditions'][key], 'actual CPU config condition differs: '+key)
    for key in ('system_conditions','runtime_files','expected_versions'):
        require(config[key]==protocol[key] and config[key], 'actual CPU config environment differs: '+key)
    require(config['mode']=='userspace' and config['cpu_index']==3 and config['frequency_khz']==1199999 and
            config['policy_name']=='policy0', 'CPU evidence differs from authorized fixed condition')
    require(config.get('userspace_authorization',{}).get('authorized') is True, 'userspace authorization absent')
    require(all(config.get(key)==value for key,value in (('batches',3),('warmup_calls',50),('measured_calls',1000),('group_size',1))),
            'actual CPU config sampling sizes differ')
    require(config['pl_baseline']['sha256']==session['pl_baseline_sha256'], 'actual CPU config baseline differs')
    if session['experiment_phase']=='validation':
        require(config['calibration_freeze']['sha256']==session['host_thresholds_sha256'], 'actual CPU config threshold binding differs')
    for record in (session,manifest):
        require(record.get('source_files')=={name:protocol['source_files'][name] for name in
                ('board_runner.py','measurement.py','cpu_conditions.py','d09_cpu_runner.py')}, 'measurement source binding differs')
    def artifact(name):
        path=root/name
        sources[name]=dict(path=path.as_posix(),sha256=digest(path))
        return read(path)
    setup=artifact('cpu_conditions_setup.json')
    check_cpu(setup,config,'after_apply')
    require(utc(setup['timestamp_utc'])<=min(utc(item['started_utc']) for item in manifest['batches']), 'CPU setup after batch start')
    check_external(artifact('system_conditions_startup.json'),config,'startup')
    restored=artifact('cpu_conditions_restore.json')
    transaction=artifact('cpu_conditions_transaction.json')
    require(restored.get('status')=='PASS_RESTORED' and summary.get('cpu_restoration_status')=='PASS_RESTORED' and
            restored.get('failures')==[] and restored.get('no_mutations') is False and
            restored.get('original')==restored.get('restored')==transaction.get('original') and
            restored.get('original_affinity')==restored.get('restored_affinity')==transaction.get('original_affinity') and
            bool(restored.get('original')) and bool(restored.get('original_affinity')) and transaction.get('state')=='RESTORED',
            'CPU restoration evidence missing, inconsistent or failed')
    require(all(transaction.get('config',{}).get(key)==value for key,value in config.items()), 'CPU transaction/config differs')
    for item in manifest['batches']:
        batch=item['batch_id']
        require(item.get('condition_deviations')==[], 'batch condition deviation exists')
        check_cpu(artifact(batch+'/cpu_before_reload.json'),config,batch+':before_reload')
        for boundary in ('start','end'):
            environment=artifact(batch+'/environment_'+boundary+'.json')
            check_cpu(environment['cpu'],config,batch+':'+boundary)
            check_external(environment['external_conditions'],config,batch+':'+boundary)
            for observed in (environment['cpu'],environment['external_conditions']):
                require(utc(item['started_utc'])<=utc(observed['timestamp_utc'])<=utc(item['finished_utc']),
                        'CPU/external boundary observation outside batch interval')
            require(environment['clock_hz']==item['clock_hz'] and
                    environment.get('software_versions')==environment['external_conditions']['software_versions'],
                    'batch boundary PL clock/version evidence differs')
            temperature=environment['temperature']
            require(temperature.get('condition_deviations')==[] and math.isfinite(float(temperature['temperature_c'])),
                    'batch temperature evidence unavailable')
    path=root/'cpu_group_observations.jsonl'
    sources[path.name]=dict(path=path.as_posix(),sha256=digest(path))
    expected=[f"{batch['batch_id']}:{phase}:{iteration}:{boundary}" for batch in manifest['batches']
              for phase,count in (('warmup',50),('measured',1000)) for iteration in range(count) for boundary in ('before','after')]
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    require(len(records)==len(expected), 'CPU group observations incomplete')
    previous=None
    intervals={item['batch_id']:(utc(item['started_utc']),utc(item['finished_utc'])) for item in manifest['batches']}
    for record,label in zip(records,expected):
        check_cpu(record,config,label)
        timestamp=utc(record['timestamp_utc'])
        start,end=intervals[label.split(':')[0]]
        require(start<=timestamp<=end, 'CPU group observation outside batch interval')
        require(previous is None or timestamp>=previous, 'CPU group observation timestamps regress')
        previous=timestamp
    require(utc(restored['timestamp_utc'])>=previous, 'CPU restoration predates observations')
    return dict(status='PASS_RAW_CPU_CONDITIONS_AND_RESTORATION',group_observations=len(records),
                configuration_path=config_path.as_posix(),statistical_exclusions=0)


def load_run(directory, phase, protocol_sha, baseline, baseline_sha, thresholds_sha=None, cpu_config_path=None):
    root = Path(directory).resolve()
    names = ('timing_raw.csv','timing_manifest.json','summary.json','session.json','counter_raw.jsonl')
    sources = {name:dict(path=(root/name).as_posix(),sha256=digest(root/name)) for name in names}
    manifest,summary,session = (read(root/name) for name in ('timing_manifest.json','summary.json','session.json'))
    require(len(manifest['cases'])==1, 'a run must contain exactly one case')
    spec = manifest['cases'][0]
    cid = spec['case_id']
    require(cid in CASE_IDS, 'unexpected case')
    prior = baseline['cases'][cid]
    for key in ('case_sha256','numeric_contract_sha256','bitstream_sha256','expected_output_sha256'):
        require(spec[key]==prior[key], 'new case identity differs from prior: '+key)
    for key, expected in (('experiment_phase',phase),('d09_protocol_sha256',protocol_sha),
                          ('pl_baseline_sha256',baseline_sha),('host_thresholds_sha256',thresholds_sha)):
        require(manifest.get(key)==expected and session.get(key)==expected, 'session/manifest binding differs: '+key)
    run_id = session.get('run_id')
    require(isinstance(run_id,str) and run_id and manifest.get('run_id')==run_id, 'run_id absent or differs')
    config_sha = session.get('cpu_conditions_config_sha256')
    require(isinstance(config_sha,str) and len(config_sha)==64 and manifest.get('cpu_conditions_config_sha256')==config_sha,
            'CPU condition config identity differs')
    require(summary.get('raw_csv_sha256')==sources['timing_raw.csv']['sha256'], 'summary/raw CSV hash differs')
    cpu_evidence=load_cpu_evidence(root,cpu_config_path,session,manifest,summary,sources)
    require(summary.get('status')=='PASS' and summary.get('d09_cpu_conditions_passed') is True and
            summary.get('pl_compute_baseline_passed') is True and summary.get('phase_clock_evidence_verified') is True,
            'upstream numeric/CPU/compute/phase gate not passed')
    require(len(summary.get('cases',[]))==1 and summary['cases'][0].get('case_id')==cid and summary['cases'][0].get('status')=='PASS',
            'upstream case status differs')
    metadata = {item['batch_id']:item for item in manifest['batches']}
    require(len(metadata)==len(manifest['batches'])==3, 'exactly three unique batches required')
    intervals, independent_ids = [], []
    for item in metadata.values():
        require(item['case_id']==cid and item['independence_event'] in ('reload_bitstream','restart_runtime'), 'batch independence event missing')
        require(item['clock_hz']==item['clock_hz_end']==prior['clock_hz'], 'batch PL frequency differs')
        independent_ids.append(item['independence_id'])
        intervals.append((utc(item['started_utc']),utc(item['finished_utc'])))
    intervals.sort()
    require(len(set(independent_ids))==3 and all(isinstance(value,str) and value for value in independent_ids), 'batch independence IDs repeated/absent')
    require(all(start<end for start,end in intervals) and all(intervals[i][1]<=intervals[i+1][0] for i in range(2)), 'batch time windows overlap/invalid')
    counters = {}
    for line in (root/'counter_raw.jsonl').read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        key = (item['batch_id'],item['phase'],int(item['iteration']))
        require(key not in counters, 'duplicate raw counter iteration')
        counters[key] = item
    observed, groups, last_sequence = set(), set(), {}
    values = defaultdict(lambda:defaultdict(list))
    with (root/'timing_raw.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            batch, row_phase = row['batch_id'],row['phase']
            require(batch in metadata and row_phase in ('warmup','measured') and row['case_id']==cid, 'unexpected raw case/batch/phase')
            key=(batch,row_phase,int(row['iteration']))
            require(key not in observed and key in counters, 'duplicate iteration or missing counter')
            observed.add(key)
            require(int(row['group_size'])==1, 'N must stay exactly one')
            group_key=(batch,row_phase,row['group_id'])
            require(group_key not in groups, 'duplicate host group')
            groups.add(group_key)
            for name in ('case_sha256','numeric_contract_sha256','bitstream_sha256','expected_output_sha256','board_id'):
                require(row[name]==spec[name], 'raw identity differs: '+name)
            require(int(row['mismatched_elements'])==0 and int(row['compared_elements'])==spec['expected_elements'] and
                    row['observed_output_sha256']==spec['expected_output_sha256'], 'raw output parity differs')
            clock=positive(row['clock_hz'],'PL clock')
            counter=counters[key]
            require(clock==counter['clock_hz']==prior['clock_hz'], 'raw PL frequency differs')
            require(counter['compute_cycles']==prior['compute_cycles'] and counter['phase_counts_verified'] is True and
                    counter['timer_status']==1 and counter['sequence']==counter['sequence_before']+1 and
                    counter['total_active_cycles']==counter['compute_cycles']+counter['transfer_cycles'], 'counter or compute baseline gate failed')
            require(batch not in last_sequence or counter['sequence']==last_sequence[batch]+1, 'counter sequence not consecutive')
            last_sequence[batch]=counter['sequence']
            for scope in ('compute','transfer'):
                number=positive(row[scope+'_ns'],scope)
                expected=counter[scope+'_cycles']*1e9/clock
                require(math.isclose(number,expected,rel_tol=1e-12,abs_tol=1e-6), 'CSV/counter duration differs')
            host=positive(row['group_call_total_ns'],'host group total')
            require(positive(row['call_ns'],'raw call time')==host, 'N=1 call_ns differs from group_call_total_ns')
            require(host+1e-6>=float(row['compute_ns'])+float(row['transfer_ns']), 'PL intervals exceed N=1 host group time')
            values[batch][row_phase].append(dict(host=host,compute=float(row['compute_ns']),transfer=float(row['transfer_ns'])))
    require(observed==set(counters), 'CSV/counter coverage differs')
    batch_stats, transfer_values, pl_failures = [], [], []
    for batch in metadata:
        require(len(values[batch]['warmup'])==50 and len(values[batch]['measured'])==1000, 'batch must have exactly 50 warmup + 1000 measured calls')
        host=stats([row['host'] for row in values[batch]['measured']])
        transfer=stats([row['transfer'] for row in values[batch]['measured']])
        compute=stats([row['compute'] for row in values[batch]['measured']])
        transfer_values.extend(row['transfer'] for row in values[batch]['measured'])
        deviation=abs(transfer['median_ns']/prior['prior_transfer_median_ns']-1)
        if transfer['cv']>0.05 or compute['cv']>0.05 or deviation>0.01:
            pl_failures.append(batch)
        batch_stats.append(dict(batch_id=batch,host=host,transfer=transfer,compute=compute,transfer_median_relative_deviation=deviation))
    pooled=stats(transfer_values)
    pooled_deviation=abs(pooled['median_ns']/prior['prior_transfer_median_ns']-1)
    if pooled['cv']>0.05 or pooled_deviation>0.01:
        pl_failures.append('pooled')
    center=statistics.median(row['host']['median_ns'] for row in batch_stats)
    return dict(case_id=cid,run_id=run_id,experiment_phase=phase,root=root.as_posix(),sources=sources,
        cpu_conditions_config_sha256=config_sha,protocol_sha256=protocol_sha,pl_baseline_sha256=baseline_sha,
        first_started_utc=intervals[0][0].isoformat(),last_finished_utc=intervals[-1][1].isoformat(),
        independence_ids=independent_ids,batches=batch_stats,median_of_batch_medians_ns=center,
        within_run_median_relative_deviation=max(abs(row['host']['median_ns']/center-1) for row in batch_stats),
        pooled_transfer=pooled,pooled_transfer_median_relative_deviation=pooled_deviation,
        pl_consistency_passed=not pl_failures,pl_failed_groups=pl_failures,
        upstream_cpu_conditions_passed=True,cpu_evidence=cpu_evidence,raw_rows=len(observed),statistical_exclusions=0)


def check_run_set(runs, phase):
    require(len(runs)==3 and {row['case_id'] for row in runs}==set(CASE_IDS), 'exactly the three fixed cases required')
    require(all(row['experiment_phase']==phase for row in runs), 'wrong phase in run set')
    require(len({row['run_id'] for row in runs})==3, 'run IDs must be unique')
    ids=[value for row in runs for value in row['independence_ids']]
    require(len(ids)==len(set(ids))==9, 'independence IDs repeated across case runs')


def calibrate(runs, protocol_sha, baseline_sha):
    check_run_set(runs,'calibration')
    require(all(row['protocol_sha256']==protocol_sha and row['pl_baseline_sha256']==baseline_sha for row in runs), 'CAL shared protocol/baseline differs')
    cmax=max(row['host']['cv'] for run in runs for row in run['batches'])
    emax=max(Decimal(str(row['host']['p95_over_median']))-1 for run in runs for row in run['batches'])
    dmax=max(run['within_run_median_relative_deviation'] for run in runs)
    margin=Decimal(CONTRACT['margin_multiplier'])
    cv=ceil_step(max(Decimal(CONTRACT['cv_floor']),margin*Decimal(str(cmax))),CONTRACT['cv_step'])
    ratio=Decimal(1)+ceil_step(max(Decimal(CONTRACT['tail_excess_floor']),margin*emax),CONTRACT['tail_excess_step'])
    drift=ceil_step(max(Decimal(CONTRACT['drift_floor']),margin*Decimal(str(dmax))),CONTRACT['drift_step'])
    caps=(cv<=Decimal(CONTRACT['cv_cap']) and ratio<=Decimal(CONTRACT['p95_over_median_cap']) and drift<=Decimal(CONTRACT['drift_cap']))
    pl=all(row['pl_consistency_passed'] for row in runs)
    status='PASS_CALIBRATION_THRESHOLDS_LOCKED' if caps and pl else 'FAIL_CALIBRATION_NO_FREEZE'
    return dict(schema='d09_host_thresholds_v1',status=status,created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_status='DRAFT_PENDING_INDEPENDENT_VALIDATION',d09_protocol_sha256=protocol_sha,pl_baseline_sha256=baseline_sha,
        statistics_contract=CONTRACT,statistics_contract_sha256=canonical(CONTRACT),statistics_script_sha256=digest(__file__),
        calibration_run_ids=[row['run_id'] for row in runs],calibration_runs=runs,
        calibration_centers_ns={row['case_id']:row['median_of_batch_medians_ns'] for row in runs},
        observed_maxima=dict(cv=cmax,tail_excess=float(emax),median_relative_deviation=dmax),
        candidate_thresholds=dict(host_cv_limit=float(cv),host_p95_over_median_limit=float(ratio),host_median_relative_limit=float(drift)),
        host_cv_limit=float(cv) if caps and pl else None,host_p95_over_median_limit=float(ratio) if caps and pl else None,
        host_median_relative_limit=float(drift) if caps and pl else None,
        engineering_caps_passed=bool(caps),pl_consistency_passed=pl,statistical_exclusions=0,formal_cost_eligible=False)


def validate(runs, freeze, freeze_sha, protocol_sha, baseline_sha):
    check_run_set(runs,'validation')
    require(freeze.get('status')=='PASS_CALIBRATION_THRESHOLDS_LOCKED', 'CAL did not produce locked passing thresholds')
    require(freeze['d09_protocol_sha256']==protocol_sha and freeze['pl_baseline_sha256']==baseline_sha, 'threshold protocol/baseline differs')
    require(freeze['statistics_contract']==CONTRACT and freeze['statistics_contract_sha256']==canonical(CONTRACT) and
            freeze['statistics_script_sha256']==digest(__file__), 'threshold statistic algorithm/version differs')
    replay=calibrate(freeze['calibration_runs'],protocol_sha,baseline_sha)
    for field in ('status','host_cv_limit','host_p95_over_median_limit','host_median_relative_limit','calibration_centers_ns'):
        require(freeze[field]==replay[field], 'locked CAL threshold or center differs from deterministic replay: '+field)
    cal_ids=set(freeze['calibration_run_ids'])
    require(cal_ids=={row['run_id'] for row in freeze['calibration_runs']}, 'CAL run ID binding differs')
    cal_independent={value for row in freeze['calibration_runs'] for value in row['independence_ids']}
    cal_roots={row['root'] for row in freeze['calibration_runs']}
    for cal in freeze['calibration_runs']:
        for record in cal['sources'].values():
            require(digest(record['path'])==record['sha256'], 'CAL artifact changed after threshold locking')
    results=[]
    for run in runs:
        require(run['protocol_sha256']==protocol_sha and run['pl_baseline_sha256']==baseline_sha, 'VAL shared protocol/baseline differs')
        require(run['run_id'] not in cal_ids and run['root'] not in cal_roots and not(set(run['independence_ids'])&cal_independent), 'VAL reuses CAL run/batch')
        require(utc(run['first_started_utc'])>utc(freeze['created_utc']), 'VAL began before thresholds were locked')
        center=freeze['calibration_centers_ns'][run['case_id']]
        batch_results=[]
        for batch in run['batches']:
            host=batch['host']
            deviation=abs(host['median_ns']/center-1)
            batch_results.append(dict(batch_id=batch['batch_id'],host=host,calibration_center_ns=center,
                anchor_median_relative_deviation=deviation,
                cv_passed=host['cv']<=freeze['host_cv_limit'],p95_ratio_passed=host['p95_over_median']<=freeze['host_p95_over_median_limit'],
                anchor_median_passed=deviation<=freeze['host_median_relative_limit']))
        within=run['within_run_median_relative_deviation']<=freeze['host_median_relative_limit']
        passed=within and run['pl_consistency_passed'] and all(all(row[key] for key in ('cv_passed','p95_ratio_passed','anchor_median_passed')) for row in batch_results)
        results.append(dict(case_id=run['case_id'],run_id=run['run_id'],passed=passed,within_val_median_passed=within,
            within_val_median_relative_deviation=run['within_run_median_relative_deviation'],pl_consistency_passed=run['pl_consistency_passed'],batches=batch_results))
    return dict(schema='d09_host_validation_v1',status='PASS_INDEPENDENT_HOST_AND_PL_VALIDATION' if all(row['passed'] for row in results) else 'FAIL_VALIDATION_NO_FREEZE',
        created_utc=datetime.now(timezone.utc).isoformat(),host_thresholds_sha256=freeze_sha,d09_protocol_sha256=protocol_sha,
        pl_baseline_sha256=baseline_sha,statistics_script_sha256=digest(__file__),validation_runs=runs,case_results=results,
        thresholds_unchanged=True,statistical_exclusions=0,
        scope='Independent host/PL statistics and identity/time binding; CPU control evidence separately required for D09 freeze',formal_cost_eligible=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('calibrate','validate'))
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--pl-baseline',type=Path,required=True)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--cpu-config',type=Path,action='append',required=True)
    parser.add_argument('--thresholds',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite an existing derived result')
    protocol=read(args.protocol)
    require(protocol.get('host_statistics_contract')==CONTRACT, 'common protocol must bind the exact host_statistics_contract')
    baseline=load_baseline(args.pl_baseline)
    protocol_sha,baseline_sha=digest(args.protocol),digest(args.pl_baseline)
    freeze_sha=digest(args.thresholds) if args.thresholds else None
    require((args.mode=='validate')==(args.thresholds is not None), '--thresholds is required only for validate')
    require(len(args.cpu_config)==len(args.run), 'one --cpu-config is required for each --run in the same order')
    runs=[load_run(path,'calibration' if args.mode=='calibrate' else 'validation',protocol_sha,baseline,baseline_sha,freeze_sha,config)
          for path,config in zip(args.run,args.cpu_config)]
    if args.mode=='validate':
        freeze=read(args.thresholds)
        recomputed=[load_run(row['root'],'calibration',protocol_sha,baseline,baseline_sha,None,row['cpu_evidence']['configuration_path'])
                    for row in freeze['calibration_runs']]
        require(recomputed==freeze['calibration_runs'], 'frozen CAL statistics differ from independent raw CSV replay')
        result=validate(runs,freeze,freeze_sha,protocol_sha,baseline_sha)
    else:
        result=calibrate(runs,protocol_sha,baseline_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],output=args.output.as_posix(),sha256=digest(args.output)),ensure_ascii=False))
    return 0 if result['status'].startswith('PASS_') else 2


if __name__=='__main__':
    raise SystemExit(main())
