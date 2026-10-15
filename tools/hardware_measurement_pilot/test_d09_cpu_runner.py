"""D09 safety/integration tests; all OS and hardware boundaries are mocked."""
from contextlib import ExitStack, redirect_stdout
import copy
import csv
from datetime import datetime, timezone, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.hardware_measurement_pilot import d09_cpu_runner as runner
from tools.hardware_measurement_pilot.test_cpu_conditions import config


VERSIONS=dict(kernel_release="test-kernel",python_version="3.10.4",pynq="3.0.1",numpy="1.21.5")
SYSTEMS=dict(boot_id="11111111-2222-3333-4444-555555555555",stopped_units=["test-idle.service"],irq_affinity={"39":"0-2"})


class FakeExternal:
    def __init__(self):
        self.version_data=dict(VERSIONS)
        self.unit=dict(LoadState="loaded",ActiveState="inactive",SubState="dead")
        self.values={"/proc/sys/kernel/random/boot_id":SYSTEMS["boot_id"],
                     "/proc/irq/39/smp_affinity_list":"0-2","/proc/irq/39/effective_affinity_list":"0"}
        self.digest="f"*64
    def versions(self): return self.version_data
    def unit_state(self,unit): return self.unit
    def read(self,path): return self.values[str(path)]
    def sha_file(self,path): return self.digest


class FakeController:
    instances=[]
    failure_label=None
    restore_status="PASS_RESTORED"
    def __init__(self,settings,journal_callback=None):
        self.settings=settings; self.restores=0; self.labels=[]
        self.instances.append(self)
    def apply(self): return self.snapshot("setup")
    def snapshot(self,label):
        self.labels.append(label)
        errors=["INJECTED_CPU_FREQUENCY_DRIFT"] if label==self.failure_label else []
        return dict(cpu_governor=self.settings["mode"],condition_deviations=errors,current_cpu=3,
                    affinities={"10":[3]},thread_count=1,label=label,status="FAIL" if errors else "PASS")
    def require(self,snapshot): runner.cpu.CpuConditions.require(snapshot)
    def restore(self):
        self.restores+=1
        return dict(status=self.restore_status,no_mutations=False,failures=[] if self.restore_status=="PASS_RESTORED" else ["injected"])
    def state_record(self): return dict(state="MOCK")


class FakeSession:
    instances=[]
    def __init__(self,root,case,metadata,registers,timeout,progress=None):
        self.index=len(self.instances); self.instances.append(self)
        self.clock_hz=187498125.0; self.abi={}; self.unsafe=False; self.closed=False
        self.independence_id="reload_bitstream:mock"+str(self.index)
        self.downloaded_at_utc=runner.board._utc()
    def _clock(self): return self.clock_hz
    def select_fixture(self,index): pass
    def close(self):
        if self.unsafe:
            if self not in runner.board.QUARANTINE: runner.board.QUARANTINE.append(self)
            raise runner.board.UnsafeDeviceState("mock DMA still owns buffers")
        self.closed=True
        if self in runner.board.QUARANTINE: runner.board.QUARANTINE.remove(self)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)/"case"; self.root.mkdir()
        (self.root/"board_bundle.json").write_text("{}")
        (self.root/"cpu_conditions.json").write_text("{}")
        self.output=Path(self.temporary.name)/"output"
        FakeController.instances=[]; FakeController.failure_label=None; FakeController.restore_status="PASS_RESTORED"
        FakeSession.instances=[]
        runner.board.QUARANTINE.clear(); runner.board.GOVERNOR_QUARANTINE.clear()
        self.settings=config(pl_compute_expected_cycles=100)
        self.settings=runner.cpu.validate_config(self.settings)
        self.fixture=dict(expected_elements=1,expected_output_sha256="d"*64)
        self.case=dict(case_id="body8_to_head16_reference_high",case_sha256="a"*64,numeric_contract_sha256="b"*64,
                       fixtures=[dict(fixture_id="test"+str(i),**self.fixture) for i in range(5)])
        self.bundle=dict(files={"vivado_shell/top.bit":"c"*64,"phase_validation.json":"e"*64},phase_validation="phase_validation.json")
        self.bundle["files"].update({"board_runner.py":runner.board._sha(runner.board.__file__),
                                    "measurement.py":runner.board._sha(runner.board.measurement.__file__)})
        self.binding=dict(experiment_phase="calibration",run_id="cal_test",d09_protocol_sha256="1"*64,
            cpu_conditions_config_sha256="2"*64,host_thresholds_sha256=None,pl_baseline_sha256="3"*64,
            external_conditions_evidence_sha256=None,source_files={})

    def tearDown(self):
        runner.board.QUARANTINE.clear(); runner.board.GOVERNOR_QUARANTINE.clear()
        self.temporary.cleanup()

    def mock_group(self,session,base,batch_id,phase,first,count,group_id,writer,stream,counters):
        compute=100*1e9/session.clock_hz; transfer=200*1e9/session.clock_hz
        row=dict(base,iteration=first,phase=phase,compute_ns=compute,transfer_ns=transfer,call_ns=1000000,
                 compared_elements=1,mismatched_elements=0,observed_output_sha256=base["expected_output_sha256"],
                 batch_id=batch_id,group_id=group_id,group_size=1,group_call_total_ns=1000000)
        counters.write(json.dumps(dict(compute_cycles=100,transfer_cycles=200,total_active_cycles=300,
                                      compute_ns=compute,transfer_ns=transfer,phase=phase,iteration=first,batch_id=batch_id))+"\n")
        writer.writerow(row); stream.flush(); counters.flush()
        return 1000000

    def context(self,group=None):
        stack=ExitStack()
        stack.enter_context(patch.object(runner,"validate_runtime",return_value=(self.root,self.bundle,self.case,dict(clock_hz=187500000),{},self.settings,self.binding)))
        stack.enter_context(patch.object(runner.board,"verified_board_identity",return_value=dict(board_id="machine-id:mock")))
        stack.enter_context(patch.object(runner.cpu,"external_snapshot",return_value=dict(status="PASS_EXTERNAL_CONDITIONS",condition_deviations=[],software_versions=VERSIONS)))
        stack.enter_context(patch.object(runner.board,"temperature_snapshot",return_value=dict(temperature_c=30,condition_deviations=[])))
        stack.enter_context(patch.object(runner.board,"BoardSession",FakeSession))
        stack.enter_context(patch.object(runner.board,"_correctness",return_value=[dict(compute_cycles=100+i) for i in range(5)]))
        stack.enter_context(patch.object(runner.board,"_record_group",side_effect=group or self.mock_group))
        # Keep synthetic batch timestamps ordered without changing real statistical code.
        start=datetime.now(timezone.utc)
        tick=[0]
        def utc():
            tick[0]+=1
            return (start+timedelta(seconds=tick[0]*10)).isoformat()
        stack.enter_context(patch.object(runner.board,"_utc",side_effect=utc))
        return stack

    def execute(self):
        return runner.run_case(self.root,self.output,"machine-id:mock",self.root/"cpu_conditions.json",controller_factory=FakeController)

    def test_complete_three_batch_mock_retains_legacy_condition_and_restores(self):
        # No actual hardware, sysfs, services or OS affinity touched by this mock.
        with self.context(): result=self.execute()
        self.assertEqual(result["status"],"PASS")
        self.assertTrue(result["d09_cpu_conditions_passed"])
        self.assertFalse(result["protocol_conditions_met"],"userspace never masquerades as old performance condition")
        self.assertFalse(result["hardware_measurement_eligible"])
        with (self.output/"timing_raw.csv").open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),3150)
        self.assertEqual(sum(row["phase"]=="measured" for row in rows),3000)
        self.assertEqual(len((self.output/"cpu_group_observations.jsonl").read_text().splitlines()),6300)
        self.assertEqual(len(FakeSession.instances),3)
        self.assertTrue(all(s.closed for s in FakeSession.instances))
        self.assertEqual(FakeController.instances[0].restores,1)
        self.assertEqual(json.loads((self.output/"cpu_conditions_restore.json").read_text())["status"],"PASS_RESTORED")

    def test_drift_after_call_preserves_completed_raw_row_and_restores(self):
        FakeController.failure_label="batch1:warmup:0:after"
        with self.context(),self.assertRaises(runner.cpu.CpuConditionError): self.execute()
        with (self.output/"timing_raw.csv").open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),1)
        self.assertEqual(len((self.output/"counter_raw.jsonl").read_text().splitlines()),1)
        self.assertEqual(FakeController.instances[0].restores,1)
        self.assertTrue((self.output/"cpu_condition_failure.json").exists())

    def test_pl_mismatch_after_call_preserves_row(self):
        def mismatch(*args):
            value=self.mock_group(*args)
            args[-1].last_record["compute_cycles"]=101
            return value
        with self.context(mismatch),self.assertRaises(runner.cpu.CpuConditionError): self.execute()
        with (self.output/"timing_raw.csv").open() as f: self.assertEqual(len(list(csv.DictReader(f))),1)
        self.assertEqual(FakeController.instances[0].restores,1)

    def test_pre_cpu_validation_failure_writes_positive_no_mutation_receipt(self):
        with patch.object(runner,"validate_runtime",side_effect=ValueError("bad bundle")),self.assertRaises(ValueError): self.execute()
        receipt=json.loads((self.output/"cpu_conditions_restore.json").read_text())
        self.assertEqual(receipt["status"],"PASS_RESTORED")
        self.assertTrue(receipt["no_mutations"])
        self.assertEqual(FakeController.instances,[])

    def test_restore_failure_changes_result_and_requires_retained_lock(self):
        FakeController.restore_status="FAIL_RESTORE"
        with self.context(): result=self.execute()
        self.assertEqual(result["status"],"FAIL")
        self.assertTrue((self.output/"cpu_conditions_retention.json").exists())

    def test_dma_quarantine_defers_restore_until_original_idle_loop(self):
        def unsafe(session,*args):
            session.unsafe=True
            raise runner.board.UnsafeDeviceState("injected timeout")
        with self.context(unsafe),self.assertRaises(runner.board.UnsafeDeviceState): self.execute()
        self.assertEqual(FakeController.instances[0].restores,0)
        receipt=json.loads((self.output/"cpu_conditions_restore.json").read_text())
        self.assertEqual(receipt["status"],"DEFERRED_DMA_QUARANTINE")
        self.assertEqual(len(runner.board.QUARANTINE),1)
        FakeSession.instances[0].unsafe=False
        with redirect_stdout(io.StringIO()): runner.board._retain_cli(self.output)
        self.assertEqual(FakeController.instances[0].restores,1)
        self.assertEqual(json.loads((self.output/"cpu_conditions_restore.json").read_text())["status"],"PASS_RESTORED")
        self.assertEqual(runner.board.QUARANTINE,[])

    def test_external_reads_fail_closed_for_version_service_irq_boot_and_hash(self):
        settings=config(expected_versions=VERSIONS,system_conditions=SYSTEMS,runtime_files={"/runtime/file.py":"f"*64})
        system=FakeExternal()
        self.assertEqual(runner.cpu.external_snapshot(settings,"mock",system=system)["status"],"PASS_EXTERNAL_CONDITIONS")
        mutations=[lambda s:s.version_data.update(numpy="other"),lambda s:s.unit.update(ActiveState="active"),
                   lambda s:s.values.update({"/proc/irq/39/effective_affinity_list":"3"}),
                   lambda s:s.values.update({"/proc/sys/kernel/random/boot_id":"other"}),lambda s:setattr(s,"digest","0"*64),
                   lambda s:s.values.pop("/proc/irq/39/effective_affinity_list")]
        for mutate in mutations:
            system=FakeExternal(); mutate(system)
            snapshot=runner.cpu.external_snapshot(settings,"mock",system=system)
            self.assertEqual(snapshot["status"],"FAIL_EXTERNAL_CONDITIONS")
            with self.assertRaises(runner.cpu.CpuConditionError): runner.cpu.CpuConditions.require(snapshot)

    def test_extra_hash_common_protocol_freeze_and_baseline_binding(self):
        for name,source in (("d09_cpu_runner.py",runner.__file__),("cpu_conditions.py",runner.cpu.__file__)):
            (self.root/name).write_bytes(Path(source).read_bytes())
        settings=config(expected_versions=VERSIONS,system_conditions=SYSTEMS,runtime_files={"/runtime/file.py":"f"*64},
                        pl_compute_expected_cycles=100)
        common=dict(cpu_conditions={k:settings[k] for k in ("mode","cpu_index","policy_name","frequency_khz")},
                    expected_versions=VERSIONS,system_conditions=SYSTEMS,runtime_files=settings["runtime_files"])
        self.bundle["files"].update({name:runner.board._sha(self.root/name) for name in ("d09_cpu_runner.py","cpu_conditions.py")})
        def bind(name,value):
            path=self.root/name; path.write_text(json.dumps(value),encoding="utf-8")
            self.bundle["files"][name]=runner.board._sha(path)
            return dict(path=name,sha256=self.bundle["files"][name])
        settings["protocol"]=bind("d09_protocol.json",common)
        settings["pl_baseline"]=bind("pl_baseline.json",dict(cases={self.case["case_id"]:dict(compute_cycles=100,
            case_sha256=self.case["case_sha256"],bitstream_sha256=self.bundle["files"]["vivado_shell/top.bit"])}))
        def validate():
            bind("cpu_conditions.json",settings)
            with patch.object(runner.board,"validate_bundle",return_value=(self.root,self.bundle,self.case,{},{})):
                return runner.validate_runtime(self.root,self.root/"cpu_conditions.json")
        self.assertEqual(validate()[5]["mode"],"userspace")
        settings["frequency_khz"]=600000
        with self.assertRaises(ValueError): validate()
        settings["frequency_khz"]=1200000
        settings.update(experiment_phase="validation",run_id="val_test",host_median_relative_limit=.02)
        freeze=dict(host_median_relative_limit=.02,calibration_run_ids=["cal_test"],
                    d09_protocol_sha256=settings["protocol"]["sha256"],created_utc="2026-01-01T00:00:00+00:00")
        settings["calibration_freeze"]=bind("host_thresholds.json",freeze)
        self.assertEqual(validate()[5]["experiment_phase"],"validation")
        settings["host_median_relative_limit"]=.03
        with self.assertRaises(ValueError): validate()
        settings["host_median_relative_limit"]=.02; settings["run_id"]="cal_test"
        with self.assertRaises(ValueError): validate()


if __name__ == "__main__": unittest.main()
