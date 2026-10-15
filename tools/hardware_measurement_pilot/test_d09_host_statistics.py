"""Synthetic engineering-boundary tests; no board acceptance evidence."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.hardware_measurement_pilot import d09_host_statistics as ds
from tools.hardware_measurement_pilot.measurement import CSV_FIELDS


def summaries(phase='calibration',cv=.02,ratio=1.04,drift=.004):
    result=[]
    for index,cid in enumerate(ds.CASE_IDS):
        center=(index+1)*1000000
        result.append(dict(case_id=cid,run_id=phase+cid,experiment_phase=phase,root='/synthetic/'+phase+cid,sources={},
            protocol_sha256='a'*64,pl_baseline_sha256='b'*64,first_started_utc='2099-01-01T00:00:00+00:00',
            last_finished_utc='2099-01-01T00:00:03+00:00',independence_ids=[phase+cid+str(i) for i in range(3)],
            batches=[dict(batch_id='batch'+str(i),host=dict(n=1000,mean_ns=center,median_ns=center,p95_ns=center*ratio,cv=cv,p95_over_median=ratio)) for i in range(3)],
            median_of_batch_medians_ns=center,within_run_median_relative_deviation=drift,pl_consistency_passed=True))
    return result


class ThresholdBoundaryTests(unittest.TestCase):
    def test_floor_and_decimal_rounding(self):
        floor=ds.calibrate(summaries(cv=0,ratio=1,drift=0),'a'*64,'b'*64)
        self.assertEqual((floor['host_cv_limit'],floor['host_p95_over_median_limit'],floor['host_median_relative_limit']),(.01,1.02,.005))
        regular=ds.calibrate(summaries(),'a'*64,'b'*64)
        self.assertEqual((regular['host_cv_limit'],regular['host_p95_over_median_limit'],regular['host_median_relative_limit']),(.025,1.05,.005))

    def test_all_exact_engineering_caps_pass_without_float_extra_step(self):
        result=ds.calibrate(summaries(cv=.04,ratio=1.08,drift=.016),'a'*64,'b'*64)
        self.assertEqual(result['status'],'PASS_CALIBRATION_THRESHOLDS_LOCKED')
        self.assertEqual((result['host_cv_limit'],result['host_p95_over_median_limit'],result['host_median_relative_limit']),(.05,1.10,.02))

    def test_cap_failure_is_not_clipped_or_frozen(self):
        for argument,value in [('cv',.0401),('ratio',1.0801),('drift',.0161)]:
            with self.subTest(argument=argument):
                result=ds.calibrate(summaries(**{argument:value}),'a'*64,'b'*64)
                self.assertEqual(result['status'],'FAIL_CALIBRATION_NO_FREEZE')
                self.assertIsNone(result['host_cv_limit'])
                self.assertIsNone(result['host_median_relative_limit'])

    def test_shared_worst_case_and_pl_failure(self):
        runs=summaries(cv=.01)
        runs[2]['batches'][1]['host']['cv']=.032
        result=ds.calibrate(runs,'a'*64,'b'*64)
        self.assertEqual(result['host_cv_limit'],.04)
        runs[0]['pl_consistency_passed']=False
        self.assertEqual(ds.calibrate(runs,'a'*64,'b'*64)['status'],'FAIL_CALIBRATION_NO_FREEZE')

    def test_validation_cannot_hide_collective_shift_by_recentering(self):
        freeze=ds.calibrate(summaries(),'a'*64,'b'*64)
        runs=summaries('validation')
        for run in runs:
            run['within_run_median_relative_deviation']=0
            for batch in run['batches']:
                batch['host']['median_ns']*=1.02
        result=ds.validate(runs,freeze,'c'*64,'a'*64,'b'*64)
        self.assertEqual(result['status'],'FAIL_VALIDATION_NO_FREEZE')
        self.assertTrue(all(row['within_val_median_passed'] for row in result['case_results']))
        self.assertTrue(all(not batch['anchor_median_passed'] for row in result['case_results'] for batch in row['batches']))

    def test_independent_validation_and_locked_parameters(self):
        freeze=ds.calibrate(summaries(),'a'*64,'b'*64)
        self.assertEqual(ds.validate(summaries('validation'),freeze,'c'*64,'a'*64,'b'*64)['status'],'PASS_INDEPENDENT_HOST_AND_PL_VALIDATION')
        for change in ('threshold','center','cal_id','time'):
            frozen,runs=copy.deepcopy(freeze),summaries('validation')
            if change=='threshold': frozen['host_cv_limit']=.05
            if change=='center': frozen['calibration_centers_ns'][ds.CASE_IDS[0]]=123
            if change=='cal_id': runs[0]['run_id']=frozen['calibration_run_ids'][0]
            if change=='time': runs[0]['first_started_utc']='2000-01-01T00:00:00+00:00'
            with self.subTest(change=change),self.assertRaises(ValueError):
                ds.validate(runs,frozen,'c'*64,'a'*64,'b'*64)


class RawIngestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cid=ds.CASE_IDS[0]
        clock=187498125
        self.prior=dict(case_sha256='1'*64,numeric_contract_sha256='2'*64,bitstream_sha256='3'*64,
                        expected_output_sha256='4'*64,compute_cycles=67218,clock_hz=clock,prior_transfer_median_ns=100000*1e9/clock)
        self.baseline=dict(cases={self.cid:self.prior})
        spec=dict(self.prior,case_id=self.cid,board_id='synthetic-test',expected_elements=5)
        shared=dict(experiment_phase='calibration',run_id='synthetic-cal',d09_protocol_sha256='a'*64,
                    pl_baseline_sha256='b'*64,cpu_conditions_config_sha256='c'*64,host_thresholds_sha256=None)
        self.config=dict(mode='userspace',cpu_index=3,policy_name='policy0',frequency_khz=1199999,
            batches=3,warmup_calls=50,measured_calls=1000,group_size=1,pl_baseline={'sha256':'b'*64},
            userspace_authorization={'authorized':True},run_id='synthetic-cal',experiment_phase='calibration',
            system_conditions=dict(boot_id='synthetic-boot',stopped_units=['cron.service'],irq_affinity={'39':'0-2'}),
            expected_versions=dict(kernel_release='test',python_version='3.10',pynq='3',numpy='1'),
            runtime_files={'/synthetic/python':'e'*64})
        source_files={name:'d'*64 for name in ('board_runner.py','measurement.py','cpu_conditions.py','d09_cpu_runner.py')}
        protocol=dict(cpu_conditions={key:self.config[key] for key in ('mode','cpu_index','policy_name','frequency_khz')},
            source_files=source_files,host_statistics_contract=ds.CONTRACT,
            **{key:self.config[key] for key in ('system_conditions','runtime_files','expected_versions')})
        self.save('d09_protocol.json',protocol)
        self.protocol_sha=ds.digest(self.root/'d09_protocol.json')
        self.config['protocol']=dict(path='d09_protocol.json',sha256=self.protocol_sha)
        self.save('cpu_conditions.json',self.config)
        shared.update(d09_protocol_sha256=self.protocol_sha,cpu_conditions_config_sha256=ds.digest(self.root/'cpu_conditions.json'),source_files=source_files)
        policy='/sys/devices/system/cpu/cpufreq/policy0/'
        self.cpu=dict(status='PASS_CPU_CONDITIONS',condition_deviations=[],timestamp_utc='2020-01-01T00:00:00+00:00',
            cpu_governor='userspace',requested_mode='userspace',requested_frequency_khz=1199999,
            thread_ids=[42],thread_count=1,affinities={'42':[3]},current_cpu=3,policy_related_cpus=[0,1,2,3],
            policies=[dict(path=policy+'scaling_governor',actual='userspace',available=['userspace'])],
            frequency_raw_khz={policy+field:'1199999' for field in ('scaling_min_freq','scaling_max_freq','scaling_cur_freq','cpuinfo_cur_freq','scaling_setspeed')})
        self.external=dict(status='PASS_EXTERNAL_CONDITIONS',condition_deviations=[],timestamp_utc='2020-01-01T00:00:00+00:00',
            boot_id='synthetic-boot',software_versions=self.config['expected_versions'],runtime_files=self.config['runtime_files'],
            services={'cron.service':dict(LoadState='loaded',ActiveState='inactive')},
            irq_affinity={'39':dict(requested='0-2',configured='0-2',effective='0')})
        self.save('cpu_conditions_setup.json',dict(self.cpu,label='after_apply'))
        self.save('system_conditions_startup.json',dict(self.external,label='startup'))
        self.save('cpu_conditions_restore.json',dict(status='PASS_RESTORED',failures=[],no_mutations=False,
            original={'governor':'userspace'},restored={'governor':'userspace'},original_affinity={'42':[3]},restored_affinity={'42':[3]},
            timestamp_utc='2020-01-01T00:01:00+00:00'))
        self.save('cpu_conditions_transaction.json',dict(state='RESTORED',config=self.config,
            original={'governor':'userspace'},original_affinity={'42':[3]}))
        metadata=[]
        self.raw=[]
        self.counters=[]
        for batch in range(3):
            bid='batch'+str(batch+1)
            metadata.append(dict(case_id=self.cid,batch_id=bid,independence_event='reload_bitstream',independence_id=bid,
                                 clock_hz=clock,clock_hz_end=clock,started_utc=f'2020-01-01T00:00:{batch*2:02d}+00:00',
                                 finished_utc=f'2020-01-01T00:00:{batch*2+1:02d}+00:00',condition_deviations=[]))
            self.save(bid+'/cpu_before_reload.json',dict(self.cpu,label=bid+':before_reload'))
            for boundary in ('start','end'):
                timestamp=f'2020-01-01T00:00:{batch*2:02d}+00:00'
                self.save(bid+'/environment_'+boundary+'.json',dict(cpu=dict(self.cpu,label=bid+':'+boundary,timestamp_utc=timestamp),
                    external_conditions=dict(self.external,label=bid+':'+boundary,timestamp_utc=timestamp),software_versions=self.config['expected_versions'],
                    clock_hz=clock,temperature=dict(temperature_c=40,condition_deviations=[])))
            sequence=5
            for phase,count in [('warmup',50),('measured',1000)]:
                for iteration in range(count):
                    sequence+=1
                    row=dict(case_id=self.cid,iteration=iteration,phase=phase,**{key:self.prior[key] for key in ('case_sha256','numeric_contract_sha256','bitstream_sha256','expected_output_sha256')},
                        clock_hz=clock,board_id='synthetic-test',compute_ns=67218*1e9/clock,transfer_ns=100000*1e9/clock,
                        call_ns=1000000+(iteration%2)*1000,compared_elements=5,mismatched_elements=0,observed_output_sha256='4'*64,
                        batch_id=bid,group_id=phase+str(iteration),group_size=1,group_call_total_ns=1000000+(iteration%2)*1000)
                    self.raw.append(row)
                    self.counters.append(dict(batch_id=bid,phase=phase,iteration=iteration,compute_cycles=67218,transfer_cycles=100000,
                        total_active_cycles=167218,sequence=sequence,sequence_before=sequence-1,timer_status=1,phase_counts_verified=True,clock_hz=clock))
        self.manifest=dict(shared,cases=[spec],batches=metadata)
        self.summary=dict(status='PASS',cases=[dict(case_id=self.cid,status='PASS')],d09_cpu_conditions_passed=True,
                          pl_compute_baseline_passed=True,phase_clock_evidence_verified=True,cpu_restoration_status='PASS_RESTORED')
        (self.root/'session.json').write_text(json.dumps(shared))
        self.observations=[dict(self.cpu,label=f'{row["batch_id"]}:{row["phase"]}:{row["iteration"]}:{boundary}',
                               timestamp_utc=f'2020-01-01T00:00:{(int(row["batch_id"][-1])-1)*2:02d}+00:00')
                           for row in self.raw for boundary in ('before','after')]
        self.write()

    def save(self,name,value):
        path=self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def write(self):
        with (self.root/'timing_raw.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.raw)
        self.summary['raw_csv_sha256']=ds.digest(self.root/'timing_raw.csv')
        (self.root/'summary.json').write_text(json.dumps(self.summary))
        (self.root/'timing_manifest.json').write_text(json.dumps(self.manifest))
        (self.root/'counter_raw.jsonl').write_text('\n'.join(json.dumps(row) for row in self.counters)+'\n')
        (self.root/'cpu_group_observations.jsonl').write_text('\n'.join(json.dumps(row) for row in self.observations)+'\n')

    def load(self):
        return ds.load_run(self.root,'calibration',self.protocol_sha,self.baseline,'b'*64,cpu_config_path=self.root/'cpu_conditions.json')

    def test_complete_csv_counter_and_all_rows_preserved(self):
        result=self.load()
        self.assertEqual(result['raw_rows'],3150)
        self.assertEqual(result['statistical_exclusions'],0)
        self.assertTrue(result['pl_consistency_passed'])
        self.assertEqual([row['host']['n'] for row in result['batches']],[1000]*3)

    def test_slow_observation_enters_statistics_without_deletion(self):
        self.raw[50]['group_call_total_ns']=1000000000
        self.raw[50]['call_ns']=1000000000
        self.write()
        result=self.load()
        self.assertEqual(result['batches'][0]['host']['n'],1000)
        self.assertEqual(result['batches'][0]['host']['maximum_ns'],1000000000)
        self.assertGreater(result['batches'][0]['host']['cv'],.05)

    def test_group_size_and_nonfinite_values_rejected(self):
        for key,value in [('group_size',16),('call_ns','nan'),('group_call_total_ns',0)]:
            prior=self.raw[50][key]
            self.raw[50][key]=value
            self.write()
            with self.subTest(key=key),self.assertRaises(ValueError): self.load()
            self.raw[50][key]=prior

    def test_changed_counter_or_upstream_cpu_gate_rejected(self):
        self.counters[50]['compute_cycles']-=1
        self.write()
        with self.assertRaises(ValueError): self.load()
        self.counters[50]['compute_cycles']+=1
        self.summary['d09_cpu_conditions_passed']=False
        self.write()
        with self.assertRaises(ValueError): self.load()

    def test_n1_call_group_disagreement_rejected(self):
        self.raw[50]['call_ns']+=1
        self.write()
        with self.assertRaisesRegex(ValueError,'N=1 call_ns'): self.load()

    def test_summary_pass_cannot_hide_raw_cpu_frequency_or_affinity_failure(self):
        for key,value in [('current_cpu',2),('affinities',{'42':[0,1,2,3]}),('frequency_raw_khz',{})]:
            original=self.observations[100][key]
            self.observations[100][key]=value
            self.write()
            with self.subTest(key=key),self.assertRaises(ValueError): self.load()
            self.observations[100][key]=original

    def test_missing_cpu_observations_and_changed_boundary_rejected(self):
        original=self.observations.pop()
        self.write()
        with self.assertRaisesRegex(ValueError,'observations incomplete'): self.load()
        self.observations.append(original)
        self.write()
        self.save('batch2/environment_end.json',dict(cpu=dict(self.cpu,label='batch2:end',current_cpu=2)))
        with self.assertRaisesRegex(ValueError,'CPU affinity'): self.load()

    def test_external_versions_service_irq_hash_and_restore_checked_from_raw(self):
        for key,value in [('software_versions',{}),('services',{}),('irq_affinity',{'39':dict(requested='0-2',configured='0-2',effective='3')}),('runtime_files',{})]:
            self.save('system_conditions_startup.json',dict(self.external,label='startup',**{key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError): self.load()
        self.save('system_conditions_startup.json',dict(self.external,label='startup'))
        restored=ds.read(self.root/'cpu_conditions_restore.json')
        restored['restored_affinity']={'42':[0,1,2,3]}
        self.save('cpu_conditions_restore.json',restored)
        with self.assertRaisesRegex(ValueError,'restoration evidence'): self.load()

    def test_cpu_observation_from_prior_run_rejected_even_with_pass_status(self):
        self.observations[100]['timestamp_utc']='2019-01-01T00:00:00+00:00'
        self.write()
        with self.assertRaisesRegex(ValueError,'outside batch interval'): self.load()


if __name__=='__main__':
    unittest.main()
