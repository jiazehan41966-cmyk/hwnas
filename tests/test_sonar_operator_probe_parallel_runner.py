"""Resource amendment and scheduling contracts; no research training is run."""
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from tools.sonar_operator_probe_parallel_20261014 import runner as r


def resident(pid=5406):
    return dict(pid=pid, start_ticks=123, uid=1000, cmdline_sha256='resident', state='S')


def amendment():
    return dict(gpus=[
        dict(index=1, uuid='GPU1', mode='exclusive', min_free_mib=12288, max_start_util=15, residents=[]),
        dict(index=2, uuid='GPU2', mode='shared', min_free_mib=12288, max_start_util=15, residents=[resident()]),
        dict(index=3, uuid='GPU3', mode='shared', min_free_mib=12288, max_start_util=15, residents=[resident(5300)])])


def snapshot(index=2, pids=None, free=30000, utilization=0):
    return dict(index=index, uuid=f'GPU{index}', name='same GPU', total_mib=32000,
                free_mib=free, utilization=utilization, compute_pids=pids if pids is not None else [5406])


class AdmissionTests(unittest.TestCase):
    def test_only_original_resident_and_own_worker_are_allowed(self):
        with patch.object(r, 'gpu_snapshot', return_value=snapshot(pids=[5406, 42])), \
             patch.object(r, 'process_identity', return_value=resident()):
            result = r.admission(amendment(), 2, own_pid=42)
        self.assertTrue(result['admitted'])
        self.assertEqual(result['mode'], 'shared')
        self.assertIn('not_hardware', result['performance_claim_scope'])

    def test_unknown_training_owner_blocks_without_signaling(self):
        with patch.object(r, 'gpu_snapshot', return_value=snapshot(pids=[5406, 999])), \
             patch.object(r, 'process_identity', return_value=resident()), \
             patch.object(r.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'other_compute_pids'):
                r.admission(amendment(), 2, own_pid=42)
            launch.assert_not_called()

    def test_minimum_free_memory_and_utilization_are_real_gates(self):
        for value in (snapshot(free=12287), snapshot(utilization=16)):
            with patch.object(r, 'gpu_snapshot', return_value=value), \
                 patch.object(r, 'process_identity', return_value=resident()):
                with self.assertRaises(RuntimeError):
                    r.admission(amendment(), 2)

    def test_resident_pid_reuse_missing_owner_and_uuid_drift_fail_closed(self):
        changed = dict(resident(), start_ticks=999)
        for actual, value in ((changed, snapshot()), (None, snapshot()), (resident(), snapshot(pids=[]))):
            with patch.object(r, 'gpu_snapshot', return_value=value), \
                 patch.object(r, 'process_identity', return_value=actual):
                with self.assertRaisesRegex(ValueError, 'resident'):
                    r.admission(amendment(), 2)
        with patch.object(r, 'gpu_snapshot', return_value=dict(snapshot(), uuid='other')):
            with self.assertRaisesRegex(ValueError, 'UUID'):
                r.admission(amendment(), 2)

    def test_gpu0_and_loosened_limits_are_never_authorized(self):
        with self.assertRaises(ValueError):
            r.admission(amendment(), 0)
        value = amendment()
        value['gpus'][1]['min_free_mib'] = 1024
        with self.assertRaises(ValueError):
            r.gpu_rules(value)
        value = amendment()
        value['gpus'][2]['max_start_util'] = 100
        with self.assertRaises(ValueError):
            r.gpu_rules(value)


class FrozenExecutionTests(unittest.TestCase):
    def test_real_frozen_modules_keep_code_fingerprints_after_resource_shim(self):
        # Exercise the actual large module/constant graph rather than a toy
        # module: it reproduced unstable marshal identities under Python3.13.
        from tools.sonar_operator_probe import train as old, model
        objects = {name: value for name, value in old.__dict__.items()
                   if isinstance(value, types.FunctionType) and value.__module__ == old.__name__
                   and name not in r.RESOURCE_FUNCTIONS}
        before = r.function_hashes(old, r.RESOURCE_FUNCTIONS)
        model_before = r.function_hashes(model)
        original_source = (r.sha(old.__file__), r.sha(model.__file__))
        try:
            # Remove these two function objects from the live module without
            # removing any frozen-source/code guard.
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}):
                report = r.install_resource_shim(old, amendment(), 2)
                self.assertEqual(old.physical_gpu({}, 'cuda'), 2)
            self.assertEqual(before, r.function_hashes(old, r.RESOURCE_FUNCTIONS))
            self.assertEqual(model_before, r.function_hashes(model))
            self.assertTrue(report['training_functions_unchanged'])
            self.assertTrue(all(old.__dict__[name] is value for name, value in objects.items()))
            self.assertEqual(original_source, (r.sha(old.__file__), r.sha(model.__file__)))
        finally:
            # Do not hold references to the replaced resource functions: that
            # would mask the refcount-sensitive marshal drift being regressed.
            r.importlib.reload(old)

    def test_structural_code_guard_detects_changed_nested_computation(self):
        first, second = types.ModuleType('guard_fixture'), types.ModuleType('guard_fixture')
        exec('def outer(x):\n def inner(y): return y+1\n return inner(x)\n', first.__dict__)
        exec('def outer(x):\n def inner(y): return y+2\n return inner(x)\n', second.__dict__)
        self.assertNotEqual(r.function_hashes(first), r.function_hashes(second))
        self.assertEqual(r.function_hashes(first), r.function_hashes(first))

    def test_shared_slot_cannot_launch_without_actual_bound_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            receipt = dict(amendment_sha256='amendment', wrapper_source_sha256='wrapper',
                           base_manifest_sha256='base', original_training_function_code_sha256={})
            with self.assertRaises(FileNotFoundError):
                r.require_slot_smoke(None, output, 2, receipt)
            r.dump(output / 'RESOURCE_AMENDMENT.json', amendment())
            r.dump(output / 'environment_lock.json', {'fingerprint': 'environment'})
            smoke_root = output / 'resource_smoke' / 'gpu_2'
            report_path = smoke_root / 'smoke' / 'control' / 'seed_42' / 'smoke_report.json'
            report = dict(status='PASS', training={'batches': 2, 'images': 8, 'gradient_finite': True},
                valid={'num_images': 4}, initial_state_sha256='initial', final_state_sha256='changed',
                binding={'execution_manifest_sha256': 'base', 'environment_fingerprint': 'environment'})
            r.dump(report_path, report)
            gate = dict(status='PASS', physical_gpu=2, gpu_uuid='GPU2', **receipt,
                original_environment_lock_sha256=r.sha(output / 'environment_lock.json'),
                smoke_report_sha256=r.sha(report_path), actual_two_optimizer_steps=True,
                actual_one_validation_batch=True, actual_paired_augmented_data=True,
                identical_control_initialization=True, environment_fingerprint='environment')
            r.dump(smoke_root / 'slot_smoke_gate.json', gate)
            self.assertEqual(r.require_slot_smoke(None, output, 2, receipt)['status'], 'PASS')
            report['training']['batches'] = 0
            r.dump(report_path, report)
            gate['smoke_report_sha256'] = r.sha(report_path)
            r.dump(smoke_root / 'slot_smoke_gate.json', gate)
            with self.assertRaisesRegex(ValueError, 'actual finite'):
                r.require_slot_smoke(None, output, 2, receipt)

    def test_coordinator_has_six_hour_wait_cap_without_spending_gpu_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output, base = Path(directory) / 'output', Path(directory) / 'base'
            output.mkdir()
            for index in (2, 3):
                r.dump(output / 'resource_smoke' / f'gpu_{index}' / 'slot_smoke_gate.json', {'status': 'fixture'})
            old = types.SimpleNamespace(ResourceUnavailable=RuntimeError,
                result_dir=lambda root, variant, seed: Path(root) / 'runs' / variant / f'seed_{seed}')
            cfg = {'variants': ['control'], 'comparisons': {}}
            value = dict(prior_gpu_seconds=0., legacy_active=[], legacy_controllers=[])
            receipt = {'amendment_sha256': 'amendment', 'base_manifest_sha256': 'base'}
            clock = {'seconds': 0.}

            def sleep(seconds):
                clock['seconds'] += 10800

            with patch.object(r, 'verify_prepared', return_value=(value, cfg, {}, receipt)), \
                 patch.object(r, 'require_slot_smoke', return_value={'smoke_gpu_seconds': 0}), \
                 patch.object(r, 'admission', side_effect=RuntimeError('occupied')), \
                 patch.object(r.time, 'monotonic', side_effect=lambda: clock['seconds']), \
                 patch.object(r.time, 'sleep', side_effect=sleep), \
                 patch.object(r.subprocess, 'Popen') as child:
                state = r.scheduler(old, None, base, output, base / 'config.yaml')
            self.assertEqual(state['status'], 'RESOURCE_WAIT_TIMEOUT_NO_AUTORETRY')
            self.assertEqual(state['waiting_resource_seconds'], 21600)
            self.assertEqual(state['aggregate_gpu_seconds'], 0)
            child.assert_not_called()

    def test_shim_preserves_original_training_function_objects_and_code(self):
        old = types.ModuleType('synthetic_frozen_runner')
        exec('def run_training():\n return physical_gpu({}, "cuda")\n'
             'def physical_gpu(cfg, device):\n return 1\n'
             'def gpu_admission(cfg, device):\n return {"physical_gpu":1}\n', old.__dict__)
        old.ResourceUnavailable = RuntimeError
        original = old.run_training
        code_hashes = r.function_hashes(old, r.RESOURCE_FUNCTIONS)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}):
            report = r.install_resource_shim(old, amendment(), 2)
            self.assertEqual(old.run_training(), 2)
        self.assertIs(old.run_training, original)
        self.assertEqual(code_hashes, r.function_hashes(old, r.RESOURCE_FUNCTIONS))
        self.assertTrue(report['training_functions_unchanged'])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            with self.assertRaises(ValueError):
                old.physical_gpu({}, 'cuda')

    def test_unique_job_claim_is_atomic_and_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            first = r.claim_job(directory, 'sonarop_stage7', 42, {'owner': 1})
            with self.assertRaises(FileExistsError):
                r.claim_job(directory, 'sonarop_stage7', 42, {'owner': 2})
            existing = r.read(Path(directory) / 'claims' / 'sonarop_stage7__seed_42.json')
            self.assertEqual(first, existing)

    def test_single_coordinator_os_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with r.coordinator_lock(directory):
                with self.assertRaises(OSError):
                    with r.coordinator_lock(directory):
                        self.fail('A duplicate coordinator acquired the lock')

    def test_aggregate_gpu_budget_sums_concurrent_and_legacy_workers(self):
        legacy = [{'started_at_unix': 30}]
        running = {2: {'started_monotonic': 20}, 3: {'started_monotonic': 25}}
        self.assertEqual(r.ledger_seconds(10, legacy, running, 5, monotonic=30, unix=100), 100)
        legacy[0]['finished_at_unix'] = 80
        self.assertEqual(r.ledger_seconds(10, legacy, running, 5, monotonic=30, unix=100), 80)
        # Resource waiting/CPU work cannot add time to an empty worker set.
        self.assertEqual(r.ledger_seconds(10, [], {}, 5, monotonic=99999, unix=99999), 15)

    def test_process_identity_keeps_hash_and_never_command_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / '7'
            proc.mkdir()
            tail = ['S'] + ['0'] * 18 + ['123'] + ['0'] * 5
            (proc / 'stat').write_text('7 (python with spaces) ' + ' '.join(tail), encoding='utf-8')
            (proc / 'status').write_text('Uid:\t1000\t1000\t1000\t1000\n', encoding='utf-8')
            command = b'/usr/bin/python\0speech.py\0sensitive-argument\0'
            (proc / 'cmdline').write_bytes(command)
            (root / 'stat').write_text('btime 100\n', encoding='utf-8')
            with patch.object(r.os, 'sysconf', return_value=100, create=True):
                identity = r.process_identity(7, root)
        self.assertEqual(identity['start_ticks'], 123)
        self.assertEqual(identity['started_at_unix'], 101.23)
        self.assertEqual(identity['cmdline_sha256'], hashlib.sha256(command).hexdigest())
        self.assertNotIn('sensitive-argument', str(identity))
        self.assertNotIn('cwd', identity)

    def test_active_legacy_run_cannot_be_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            base, output = Path(directory) / 'base', Path(directory) / 'output'
            folder = base / 'runs' / 'control' / 'seed_42'
            folder.mkdir(parents=True)
            r.dump(folder / 'status.json', {'status': 'RUNNING'})
            old = types.SimpleNamespace(result_dir=lambda root, variant, seed: Path(root) / 'runs' / variant / f'seed_{seed}')
            with self.assertRaisesRegex(ValueError, 'Active/incomplete'):
                r.import_completed(old, base, output, 'control', 42, base / 'config.yaml')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
