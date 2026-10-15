"""CPU condition transactions exercised entirely against an in-memory OS."""
import copy
from pathlib import Path
import unittest

from tools.hardware_measurement_pilot import cpu_conditions as cpu


def config(**updates):
    result=dict(schema="d09_cpu_conditions_v1",campaign_id="d09_test",run_id="cal_test",
        experiment_phase="calibration",cpu_index=3,policy_name="policy0",mode="userspace",frequency_khz=1200000,
        userspace_authorization=dict(authorized=True,source="Explicit offline test authorization"),settle_seconds=0)
    result.update(updates)
    return result


class FakeSystem:
    def __init__(self):
        self.policy_root=Path("/virtual/cpufreq")
        self.environ={name:"1" for name in cpu.THREAD_ENV}
        self.values=dict(scaling_available_governors="userspace performance",related_cpus="0 1 2 3",
            scaling_available_frequencies="300000 600000 1200000",scaling_governor="userspace",
            scaling_min_freq="300000",scaling_max_freq="1200000",scaling_setspeed="600000",
            scaling_cur_freq="600000",cpuinfo_cur_freq="600000")
        self.affinities={10:{0,1,2,3}}
        self.writes=[]
        self.fail_field=None
        self.fail_once=False
        self.force_wrong_frequency=False

    def read(self,path):
        return self.values[Path(path).name]

    def exists(self,path):
        return Path(path).name in self.values

    def write(self,path,value):
        name=Path(path).name
        self.writes.append((name,str(value)))
        if self.fail_field == name:
            if self.fail_once:
                self.fail_field=None
            raise OSError("injected sysfs write failure")
        if name == "scaling_min_freq" and int(value)>int(self.values["scaling_max_freq"]):
            raise OSError("min exceeds installed max")
        if name == "scaling_max_freq" and int(value)<int(self.values["scaling_min_freq"]):
            raise OSError("max below installed min")
        self.values[name]=str(value)
        wanted=int(self.values["scaling_max_freq"]) if self.values["scaling_governor"] == "performance" else int(self.values["scaling_setspeed"])
        actual=min(max(wanted,int(self.values["scaling_min_freq"])),int(self.values["scaling_max_freq"]))
        if self.force_wrong_frequency:
            actual=600000
        self.values["scaling_cur_freq"]=self.values["cpuinfo_cur_freq"]=str(actual)

    def thread_ids(self): return sorted(self.affinities)
    def get_affinity(self,tid): return set(self.affinities[tid])
    def set_affinity(self,tid,cpus): self.affinities[tid]=set(cpus)
    def current_cpu(self): return min(self.affinities[10])
    def online_cpus(self): return {0,1,2,3}


class CpuTests(unittest.TestCase):
    def test_explicit_userspace_apply_and_exact_restore(self):
        system=FakeSystem(); original=copy.deepcopy(system.values)
        events=[]; controller=cpu.CpuConditions(config(),system=system,journal_callback=events.append)
        self.assertEqual(controller.apply()["status"],"PASS_CPU_CONDITIONS")
        self.assertEqual(system.affinities[10],{3})
        self.assertEqual(system.values["scaling_min_freq"],"1200000")
        self.assertEqual(system.values["scaling_max_freq"],"1200000")
        result=controller.restore()
        self.assertEqual(result["status"],"PASS_RESTORED")
        self.assertEqual(system.values,original)
        self.assertEqual(system.affinities[10],{0,1,2,3})
        self.assertIs(controller.restore(),result)
        self.assertTrue(any(e["state"] == "APPLIED" for e in events))

    def test_unavailable_default_performance_fails_before_any_mutation(self):
        system=FakeSystem(); system.values["scaling_available_governors"]="userspace"
        settings=config(); settings.pop("mode")
        controller=cpu.CpuConditions(settings,system=system)
        with self.assertRaises(cpu.CpuConditionError): controller.apply()
        self.assertEqual(system.writes,[])
        self.assertEqual(system.affinities[10],{0,1,2,3})
        self.assertEqual(controller.restore(),dict(status="PASS_RESTORED",no_mutations=True,changes="NONE_PREFLIGHT_DID_NOT_PASS",failures=[]))

    def test_userspace_without_explicit_authorization_rejected(self):
        with self.assertRaises(ValueError): cpu.CpuConditions(config(userspace_authorization={}),system=FakeSystem())

    def test_partial_sysfs_failure_is_reversible(self):
        system=FakeSystem(); before=copy.deepcopy(system.values)
        system.fail_field="scaling_min_freq"; system.fail_once=True
        controller=cpu.CpuConditions(config(mode="performance"),system=system)
        with self.assertRaises(OSError): controller.apply()
        self.assertEqual(controller.restore()["status"],"PASS_RESTORED")
        self.assertEqual(system.values,before)
        self.assertEqual(system.affinities[10],{0,1,2,3})

    def test_restore_failure_is_not_reported_as_success(self):
        system=FakeSystem(); controller=cpu.CpuConditions(config(),system=system); controller.apply()
        system.fail_field="scaling_min_freq"
        result=controller.restore()
        self.assertEqual(result["status"],"FAIL_RESTORE")
        self.assertTrue(result["failures"])
        self.assertEqual(system.affinities[10],{0,1,2,3},"still attempt independent affinity restoration")

    def test_frequency_and_affinity_drift_detected(self):
        system=FakeSystem(); controller=cpu.CpuConditions(config(),system=system); controller.apply()
        system.values["scaling_cur_freq"]="600000"; system.affinities[10]={2}
        record=controller.snapshot("injected_drift")
        self.assertIn("CPU_AFFINITY_OR_CURRENT_CPU_DRIFT",record["condition_deviations"])
        self.assertIn("CPU_FREQUENCY_DRIFT:scaling_cur_freq",record["condition_deviations"])
        with self.assertRaises(cpu.CpuConditionError): controller.require(record)
        self.assertEqual(controller.restore()["status"],"PASS_RESTORED")

    def test_new_thread_fails_closed_and_restores_inherited_affinity(self):
        system=FakeSystem(); controller=cpu.CpuConditions(config(),system=system); controller.apply()
        system.affinities[11]={3}
        self.assertIn("SINGLE_OS_THREAD_REQUIRED",controller.snapshot("thread")["condition_deviations"])
        self.assertEqual(controller.restore()["status"],"PASS_RESTORED")
        self.assertEqual(system.affinities[11],{0,1,2,3})

    def test_bad_environment_and_unrestorable_userspace_fail_before_writes(self):
        for modify in (lambda s:s.environ.update(OPENBLAS_NUM_THREADS="4"),
                       lambda s:s.values.update(scaling_setspeed="<unsupported>"),
                       lambda s:s.affinities.update({11:{0,1,2,3}})):
            system=FakeSystem(); modify(system)
            controller=cpu.CpuConditions(config(),system=system)
            with self.assertRaises(cpu.CpuConditionError): controller.apply()
            self.assertEqual(system.writes,[])

    def test_safe_bounds_order_for_downward_and_upward_moves(self):
        system=FakeSystem(); system.values.update(scaling_min_freq="1200000",scaling_max_freq="1200000",scaling_setspeed="1200000")
        controller=cpu.CpuConditions(config(frequency_khz=300000),system=system)
        controller.apply()
        self.assertEqual(system.writes[:2],[("scaling_min_freq","300000"),("scaling_max_freq","300000")])
        self.assertEqual(controller.restore()["status"],"PASS_RESTORED")
        self.assertEqual(system.values["scaling_min_freq"],"1200000")

    def test_settle_readback_mismatch_fails_and_restores(self):
        system=FakeSystem(); system.force_wrong_frequency=True
        controller=cpu.CpuConditions(config(),system=system)
        with self.assertRaises(cpu.CpuConditionError): controller.apply()
        system.force_wrong_frequency=False
        self.assertEqual(controller.restore()["status"],"PASS_RESTORED")

    def test_invalid_sampling_or_validation_configuration_rejected(self):
        for changes in (dict(group_size=16),dict(measured_calls=999),dict(experiment_phase="validation"),
                        dict(experiment_phase="calibration",host_median_relative_limit=.02)):
            with self.assertRaises(ValueError): cpu.validate_config(config(**changes))


if __name__ == "__main__": unittest.main()
