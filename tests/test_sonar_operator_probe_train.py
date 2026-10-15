"""Synthetic protocol/gate checks; no training metrics are manufactured here."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.sonar_operator_probe import train as runner


def row(name, seed=42, stable=.2, endpoint=.2):
    return dict(binding=dict(variant=name, seed=seed, execution_manifest_sha256='source',
        environment_fingerprint='environment', subset_contract_sha256='subset',
        validation_target_sha256='targets'),
        initialization=dict(shared_tensor_sha256={'head.weight': 'same', 'shared.weight': 'same'}),
        history=[dict(augmented_batches_sha256='pixels', image_order_sha256='order',
                      valid_batches_sha256='valid')],
        stable_metrics=dict(mAP50_95=stable), secondary_terminal_metrics=dict(mAP50_95=endpoint))


def config(names, comparisons, eligible=None):
    return dict(variants=names, comparisons=comparisons,
                confirmation_eligible_variants=eligible or list(comparisons),
                queue=dict(maximum_confirm_variants=3, allowed_gpus=[1], gpu_memory_admission_mib=1024))


class ScreeningGateTests(unittest.TestCase):
    def test_best_only_or_negative_endpoint_cannot_extend_seeds(self):
        cfg = config(['control', 'best_only', 'late_positive'],
                     {'best_only': ['control'], 'late_positive': ['control']})
        rows = [row('control'), row('best_only', stable=.19, endpoint=.3),
                row('late_positive', stable=.21, endpoint=.19)]
        selected = runner.select_confirmation(rows, cfg)
        self.assertEqual(selected['selected_variants'], [])
        self.assertEqual(selected['status'], 'NO_STAGE1_POSITIVE')

    def test_strict_stable_and_nonstrict_endpoint_screen_then_ranked_budget(self):
        names = ['control', 'parent', 'a', 'b', 'c', 'd', 'reference']
        cfg = config(names, {name: ['control', 'parent'] for name in names[2:]},
                     eligible=['a', 'b', 'c', 'd'])
        rows = [row('control'), row('parent', stable=.21)]
        rows += [row(name, stable=value, endpoint=.2) for name, value in
                 [('a', .23), ('b', .24), ('c', .25), ('d', .26), ('reference', .4)]]
        selected = runner.select_confirmation(rows, cfg)
        self.assertEqual(selected['selected_variants'], ['d', 'c', 'b'])
        self.assertEqual(selected['selected_controls'], ['control', 'parent'])
        self.assertEqual({seed for name, seed in selected['jobs']}, {43, 44})
        self.assertEqual(len(selected['jobs']), 10)
        self.assertNotIn('reference', {name for name, seed in selected['jobs']})

    def test_all_matching_controls_must_pass(self):
        cfg = config(['control', 'parent', 'candidate'], {'candidate': ['control', 'parent']})
        rows = [row('control', stable=.2), row('parent', stable=.3), row('candidate', stable=.25)]
        self.assertEqual(runner.select_confirmation(rows, cfg)['selected_variants'], [])

    def test_missing_stage1_arm_fails_closed(self):
        cfg = config(['control', 'candidate'], {'candidate': ['control']})
        with self.assertRaisesRegex(ValueError, 'all registered seed42'):
            runner.select_confirmation([row('control')], cfg)

    def test_final_gate_requires_two_endpoint_and_stable_positive_seeds(self):
        cfg = config(['control', 'candidate'], {'candidate': ['control']})
        selection = dict(selected_variants=['candidate'])
        rows = [row('control', seed) for seed in (42, 43, 44)]
        rows += [row('candidate', 42, .25, .3), row('candidate', 43, .22, .19),
                 row('candidate', 44, .19, .19)]
        result = runner.confirmation_decision(rows, cfg, selection)['candidate']
        self.assertEqual(result['status'], 'NO_STABLE_POSITIVE')
        self.assertEqual(result['parents']['control']['positive_endpoint_seeds'], 1)
        rows[-1]['secondary_terminal_metrics']['mAP50_95'] = .21
        result = runner.confirmation_decision(rows, cfg, selection)['candidate']
        self.assertEqual(result['status'], 'REVIEW_SOFTWARE_ONLY')
        self.assertFalse(result['formal_eligible'])

    def test_partial_confirmation_does_not_claim_positive(self):
        cfg = config(['control', 'candidate'], {'candidate': ['control']})
        result = runner.confirmation_decision([row('control'), row('candidate', stable=.4)], cfg,
                                             dict(selected_variants=['candidate']))
        self.assertEqual(result['candidate']['status'], 'INCOMPLETE_NO_CLAIM')


class PairingAndAdmissionTests(unittest.TestCase):
    def test_wait_retries_only_resource_occupancy_and_reports_waiting(self):
        cfg = config(['control'], {})
        clock = {'seconds': 0.}
        receipts = []

        def sleep(seconds):
            clock['seconds'] += seconds

        with patch.object(runner, 'gpu_admission', side_effect=[runner.ResourceUnavailable('other owner'),
                runner.ResourceUnavailable('other owner'), {'admitted': True}]) as admission, \
             patch.object(runner.time, 'monotonic', side_effect=lambda: clock['seconds']), \
             patch.object(runner.time, 'sleep', side_effect=sleep) as sleeper:
            result = runner.wait_for_gpu(cfg, 'cuda', on_wait=receipts.append,
                                         maximum_wait_seconds=120, poll_seconds=30)
        self.assertEqual(result['waiting_seconds'], 60)
        self.assertEqual(admission.call_count, 3)
        self.assertEqual(sleeper.call_count, 2)
        self.assertTrue(all(receipt['status'] == 'WAITING_RESOURCE' for receipt in receipts))
        self.assertEqual(receipts[-1]['remaining_wait_seconds'], 90)

    def test_wait_has_bounded_timeout_and_no_nonresource_retries(self):
        cfg = config(['control'], {})
        clock = {'seconds': 0.}

        def sleep(seconds):
            clock['seconds'] += seconds

        with patch.object(runner, 'gpu_admission', side_effect=runner.ResourceUnavailable('occupied')), \
             patch.object(runner.time, 'monotonic', side_effect=lambda: clock['seconds']), \
             patch.object(runner.time, 'sleep', side_effect=sleep):
            with self.assertRaises(runner.ResourceWaitTimeout):
                runner.wait_for_gpu(cfg, 'cuda', maximum_wait_seconds=45, poll_seconds=30)
        self.assertEqual(clock['seconds'], 45)
        with patch.object(runner, 'gpu_admission', side_effect=ValueError('malformed nvidia-smi')), \
             patch.object(runner.time, 'sleep') as sleeper:
            with self.assertRaisesRegex(ValueError, 'malformed'):
                runner.wait_for_gpu(cfg, 'cuda')
            sleeper.assert_not_called()

    def test_wait_budget_and_poll_limits_are_enforced(self):
        cfg = config(['control'], {})
        for maximum, interval in [(21601, 30), (-1, 30), (21600, 61), (21600, 0)]:
            with self.assertRaises(ValueError):
                runner.wait_for_gpu(cfg, 'cuda', maximum_wait_seconds=maximum, poll_seconds=interval)

    def test_resource_race_preserves_existing_worker_status(self):
        cfg = config(['control'], {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = runner.result_dir(root, 'control', 42)
            folder.mkdir(parents=True)
            (folder / 'run_binding.json').write_text('{}', encoding='utf-8')
            (folder / 'latest.pt').write_bytes(b'committed fixture')
            original = {'status': 'FAILED_NO_AUTORETRY', 'epoch': 13}
            runner.p.dump(folder / 'status.json', original)
            with patch.object(runner, 'verify_manifest', return_value=(cfg, {})), \
                 patch.object(runner, 'require_smoke'), \
                 patch.object(runner, 'gpu_admission', side_effect=runner.ResourceUnavailable('race')):
                with self.assertRaises(runner.ResourceUnavailable):
                    runner.run_training(root, 'control', 42, 'cuda', resume=True)
            self.assertEqual(runner.read_json(folder / 'status.json'), original)
            self.assertEqual(len(list(folder.glob('resource_admission_failure_*.json'))), 1)

    def test_queue_waiting_and_cpu_analysis_do_not_spend_gpu_budget(self):
        cfg = config(['control'], {})
        cfg['queue'].update(maximum_hours_per_run=2, maximum_gpu_hours=48,
                            resource_wait_maximum_hours=6, resource_poll_seconds=30)
        clock, snapshots = {'seconds': 0.}, []
        original_dump = runner.p.dump
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'execution_manifest.json').write_text('{}', encoding='utf-8')
            original_dump(root / 'smoke_gate.json', {'smoke_gpu_seconds': 10})

            def dump(path, value):
                if Path(path).name == 'queue_status.json':
                    snapshots.append(deepcopy(value))
                return original_dump(path, value)

            def wait(cfg, device, *, on_wait, **kwargs):
                clock['seconds'] += 120
                on_wait({'status': 'WAITING_RESOURCE', 'waiting_seconds': 120})
                return {'admitted': True, 'waiting_seconds': 120}

            class Child:
                pid = 999999

                def wait(self, **kwargs):
                    clock['seconds'] += 60
                    original_dump(runner.result_dir(root, 'control', 42) / 'result.json', row('control'))
                    return 0

            def result(*args, **kwargs):
                clock['seconds'] += 1000
                return row('control')

            with patch.object(runner, 'verify_manifest', return_value=(cfg, {})), \
                 patch.object(runner, 'require_smoke'), \
                 patch.object(runner, 'wait_for_gpu', side_effect=wait), \
                 patch.object(runner.time, 'perf_counter', side_effect=lambda: clock['seconds']), \
                 patch.object(runner.subprocess, 'Popen', return_value=Child()), \
                 patch.object(runner, 'run_result', side_effect=result), \
                 patch.object(runner, 'summarize', return_value={'confirmation_decisions': {}}), \
                 patch.object(runner.p, 'dump', side_effect=dump):
                final = runner.queue(root, 'cuda', config=root / 'config.yaml')
        waiting = [value for value in snapshots if value['status'] == 'WAITING_RESOURCE']
        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0]['gpu_hours_elapsed'], 10 / 3600)
        self.assertEqual(final['status'], 'COMPLETE')
        self.assertEqual(final['waiting_total_seconds'], 120)
        self.assertEqual(final['worker_gpu_seconds'], 60)
        self.assertEqual(final['gpu_hours_elapsed'], 70 / 3600)

    def test_prepare_seeds_determinism_before_environment_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / 'author').mkdir()
            (workspace / 'author' / 'source_receipt.json').write_text('{"commit":"abc"}', encoding='utf-8')
            prereg = workspace / 'PREREGISTRATION.md'
            prereg.write_text('synthetic fixture', encoding='utf-8')
            cfg = dict(scope='exploratory_operator_screen', author={'source_root': 'author', 'commit': 'abc'},
                       subset_contract_sha256='subset')
            flags = {'seeded': False}

            def seed(value):
                self.assertEqual(value, 42)
                flags['seeded'] = True

            def bind(*args):
                self.assertTrue(flags['seeded'], 'Environment fingerprint bound before deterministic flags')

            with patch.object(runner, 'ROOT', workspace), \
                 patch.object(runner, 'load_config', return_value=(cfg, {})), \
                 patch.object(runner, '_preregistration', return_value=prereg), \
                 patch.object(runner, 'source_hashes', return_value={}), \
                 patch.object(runner, 'gpu_admission', return_value={'admitted': True}), \
                 patch.object(runner.p, 'verify_inputs', return_value={}), \
                 patch.object(runner.t, 'load_subset_contract', return_value={'full_images': [], 'subset_id': 'test'}), \
                 patch.object(runner.t, 'frozen_validation_targets', return_value=[]), \
                 patch.object(runner.p, 'seed_all', side_effect=seed), \
                 patch.object(runner.t, 'bind_environment', side_effect=bind):
                manifest = runner.prepare(workspace / 'output', prereg, 'cpu')
            self.assertFalse(manifest['formal_eligible'])
            self.assertFalse(manifest['test_evaluated'])

    def test_actual_augmented_pixels_must_pair(self):
        rows = [row('control'), row('candidate')]
        rows[1]['history'][0]['augmented_batches_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'pixels/labels/order'):
            runner.verify_pairing(rows)

    def test_pairwise_shared_tensor_check_cannot_hide_behind_global_intersection(self):
        rows = [row('a'), row('b'), row('c')]
        rows[0]['initialization']['shared_tensor_sha256']['only_a_and_c'] = 'same'
        rows[2]['initialization']['shared_tensor_sha256']['only_a_and_c'] = 'different'
        with self.assertRaisesRegex(ValueError, 'initialization tensors'):
            runner.verify_pairing(rows)

    def test_environment_and_duplicate_arms_fail(self):
        rows = [row('control'), row('candidate')]
        rows[1]['binding']['environment_fingerprint'] = 'different'
        with self.assertRaisesRegex(ValueError, 'environment'):
            runner.verify_pairing(rows)
        with self.assertRaisesRegex(ValueError, 'Duplicated'):
            runner.verify_pairing([row('control'), row('control')])

    def test_gpu0_and_multi_gpu_visibility_are_rejected(self):
        cfg = config(['control'], {})
        for visible in ('0', '0,1', '', 'GPU-UUID'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}):
                with self.assertRaises(ValueError):
                    runner.physical_gpu(cfg, 'cuda')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            self.assertEqual(runner.physical_gpu(cfg, 'cuda'), 1)

    def test_gpu_compute_owner_blocks_admission_without_process_mutation(self):
        cfg = config(['control'], {})
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), \
             patch.object(runner.subprocess, 'check_output', side_effect=['UUID1, 29', 'UUID1, 999999']), \
             patch.object(runner.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                runner.gpu_admission(cfg, 'cuda')
            popen.assert_not_called()

    def test_idle_gpu_and_own_context_are_accepted(self):
        cfg = config(['control'], {})
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), \
             patch.object(runner.subprocess, 'check_output',
                          side_effect=['UUID1, 400', f'UUID1, {os.getpid()}']):
            report = runner.gpu_admission(cfg, 'cuda')
        self.assertTrue(report['admitted'])

    def test_missing_or_changed_late_weight_cannot_count_as_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            history = [dict(epoch=76)]
            with self.assertRaisesRegex(ValueError, 'weight archive'):
                runner._verify_late_weights(folder, history, {'run': 'x'}, 'init', {'same': True})
            path = folder / 'epoch_weights' / 'epoch_076.pt'
            path.parent.mkdir()
            torch.save(dict(binding={'run': 'other'}, epoch=76, initial_state_sha256='init',
                            initialization={'same': True}), path)
            history[0]['weight_archive'] = dict(path='epoch_weights/epoch_076.pt', sha256=runner.p.sha(path))
            with self.assertRaisesRegex(ValueError, 'another run'):
                runner._verify_late_weights(folder, history, {'run': 'x'}, 'init', {'same': True})


if __name__ == '__main__':
    unittest.main()
