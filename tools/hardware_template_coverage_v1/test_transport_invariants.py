"""Offline fail-closed gates for the new tiled ABI and finite route scope."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from tools.hardware_template_coverage_v1 import board_runner, parallel_build, remote_worker


class TransportInvariantTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        cases, files, native = [], {}, []
        for i in range(65):
            case = dict(case_id=f't6_case_{i:02d}', phase_counter_abi='P3T1',
                        phase_contract=remote_worker.PHASE_CONTRACT)
            case['case_sha256'] = remote_worker.canonical_case_sha(case)
            folder = self.root/'cases'/case['case_id']
            folder.mkdir(parents=True)
            remote_worker.atomic_json(folder/'case.json', case)
            files[f'cases/{case["case_id"]}/case.json'] = remote_worker.digest(folder/'case.json')
            cases.append(case)
            native.append(dict(case_id=case['case_id'], case_sha256=case['case_sha256'], status='PASS'))
        inventory = dict(representative_case_ids=[case['case_id'] for case in cases[:13]])
        remote_worker.atomic_json(self.root/'inventory.json', inventory)
        files['inventory.json'] = remote_worker.digest(self.root/'inventory.json')
        self.manifest = dict(schema='t6_tiled_source_v1', files=files, cases=cases)
        remote_worker.atomic_json(self.root/'source_manifest.json', self.manifest)
        remote_worker.atomic_json(self.root/'native_summary.json',
            dict(status='PASS', source_manifest_sha256=remote_worker.digest(self.root/'source_manifest.json'), cases=native))

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_tiled_identity_and_native_binding_pass(self):
        self.assertEqual(len(remote_worker.verify_inputs(self.root)['cases']), 65)

    def test_old_source_schema_rejected(self):
        self.manifest['schema'] = 'hardware_measurement_pilot_source_v1'
        remote_worker.atomic_json(self.root/'source_manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'new template coverage source schema'):
            remote_worker.verify_inputs(self.root)

    def test_old_case_abi_rejected_even_when_canonical_hash_is_valid(self):
        case = self.manifest['cases'][0]
        case['phase_counter_abi'] = 'P3M2'
        case['case_sha256'] = remote_worker.canonical_case_sha(case)
        path = self.root/'cases'/case['case_id']/'case.json'
        remote_worker.atomic_json(path, case)
        self.manifest['files'][path.relative_to(self.root).as_posix()] = remote_worker.digest(path)
        remote_worker.atomic_json(self.root/'source_manifest.json', self.manifest)
        native = json.loads((self.root/'native_summary.json').read_text())
        native['source_manifest_sha256'] = remote_worker.digest(self.root/'source_manifest.json')
        native['cases'][0]['case_sha256'] = case['case_sha256']
        remote_worker.atomic_json(self.root/'native_summary.json', native)
        with self.assertRaisesRegex(ValueError, 'tiled timer ABI/phase contract'):
            remote_worker.verify_inputs(self.root)

    def test_route_scope_is_exactly_13_and_hls_scope_is_65(self):
        args = SimpleNamespace(case_id=None, stage='hls')
        self.assertEqual(len(parallel_build.selected_case_ids(self.root, args, self.manifest)), 65)
        args.stage = 'route'
        self.assertEqual(len(parallel_build.selected_case_ids(self.root, args, self.manifest)), 13)
        self.assertEqual(len(remote_worker.route_case_ids(self.root, self.manifest)), 13)
        args.case_id = self.manifest['cases'][-1]['case_id']
        with self.assertRaisesRegex(ValueError, 'restricted to the explicit representative whitelist'):
            parallel_build.selected_case_ids(self.root, args, self.manifest)

    def test_unbound_route_whitelist_rejected(self):
        self.manifest['files'].pop('inventory.json')
        with self.assertRaisesRegex(ValueError, 'source-bound inventory'):
            parallel_build.selected_case_ids(self.root, SimpleNamespace(case_id=None, stage='route'), self.manifest)
        with self.assertRaisesRegex(ValueError, 'bound to source manifest'):
            remote_worker.route_case_ids(self.root, self.manifest)

    def test_coverage_scope_routes_13_and_never_routes_other_52(self):
        args = SimpleNamespace(case_id=None, stage='coverage')
        selected = parallel_build.selected_case_ids(self.root, args, self.manifest)
        self.assertEqual(len(selected), 65)
        plans = {name: parallel_build.stage_plan(self.root, args, self.manifest, name) for name in selected}
        self.assertEqual(sum(plan == ('hls', 'route') for plan in plans.values()), 13)
        self.assertEqual(sum(plan == ('hls',) for plan in plans.values()), 52)
        representatives = set(json.loads((self.root/'inventory.json').read_text())['representative_case_ids'])
        self.assertEqual({name for name,plan in plans.items() if 'route' in plan}, representatives)
        args.case_id = self.manifest['cases'][-1]['case_id']
        self.assertEqual(parallel_build.stage_plan(self.root, args, self.manifest, args.case_id), ('hls',))

    def test_parameter_load_and_tile_load_markers_are_both_admitted(self):
        row = self.manifest['cases'][0]
        directory = self.root/'cases'/row['case_id']/'hls_project/solution1/syn/verilog'
        directory.mkdir(parents=True)
        rtl = 'output phase_ap_vld;\n'
        for state, phase in [(2,1),(6,1),(10,2),(14,3),(18,0)]:
            rtl += f"if ((1'b1 == ap_CS_fsm_state{state})) begin\n phase = 8'd{phase};\nend\n"
        (directory/'pilot_kernel.v').write_text(rtl)
        witness = parallel_build.phase_structure(self.root, row)
        self.assertEqual(witness['phase_states']['1'], [2,6])
        self.assertIn('INTEGRATED_VALIDATION_PENDING', witness['status'])
        # HLS may consolidate equal-valued assignments into an OR condition.
        merged = rtl.replace("if ((1'b1 == ap_CS_fsm_state2)) begin\n phase = 8'd1;\nend\n", '')
        merged = merged.replace("if ((1'b1 == ap_CS_fsm_state6))", "if (((1'b1 == ap_CS_fsm_state2) | (1'b1 == ap_CS_fsm_state6)))")
        (directory/'pilot_kernel.v').write_text(merged)
        self.assertEqual(parallel_build.phase_structure(self.root, row)['phase_states']['1'], [2,6])

    def simulated_client(self, *, locked=False):
        """Execute the actual generated SSH bootstrap locally, with no child tool."""
        if locked:
            (self.root/'worker.lock').write_text('already owned')
        args = SimpleNamespace(attempt='attempt01', stage='coverage', jobs=3,
                               limit=65, case_id=None, retry_failed=False)
        remote = '/home/jiajia/experiments/hardware_template_coverage_20261013_attempt01'
        def fake_remote(command, *, input, **kwargs):
            code = input.replace(remote, self.root.as_posix())
            output = io.StringIO()
            original_path = list(sys.path)
            try:
                with redirect_stdout(output), patch.dict(sys.modules, {'remote_worker': remote_worker}):
                    exec(compile(code, '<actual-remote-bootstrap>', 'exec'), {})
                return SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr='')
            except Exception as error:
                return SimpleNamespace(returncode=1, stdout='', stderr=str(error))
            finally:
                sys.path[:] = original_path
        with patch.object(parallel_build.subprocess, 'run', side_effect=fake_remote), \
                patch.object(parallel_build.subprocess, 'Popen', return_value=SimpleNamespace(pid=123456)) as popen, \
                patch.object(parallel_build, 'atomic'), redirect_stdout(io.StringIO()):
            if locked:
                with self.assertRaisesRegex(RuntimeError, 'another worker holds the root lock'):
                    parallel_build.client(args)
                popen.assert_not_called()
            else:
                self.assertEqual(parallel_build.client(args), 0)
                popen.assert_called_once()

    def test_fresh_deployed_root_without_results_gets_log_directory(self):
        self.assertFalse((self.root/'results').exists())
        self.simulated_client()
        self.assertTrue((self.root/'results').is_dir())
        self.assertEqual(len(list((self.root/'results').glob('parallel_driver_*.log'))), 1)

    def test_locked_fresh_deployed_root_does_not_get_a_log_or_child(self):
        self.assertFalse((self.root/'results').exists())
        self.simulated_client(locked=True)
        self.assertFalse((self.root/'results').exists())

    def test_generated_hls_failure_evidence_never_overwritten(self):
        row = self.manifest['cases'][0]
        (self.root/'cases'/row['case_id']/'hls_project').mkdir()
        with patch.object(remote_worker.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(FileExistsError, 'preserve generated evidence'):
                remote_worker.run_stage(self.root, row, 'hls', 10, 'unused')
            popen.assert_not_called()

    def test_route_gate_rejects_old_timer_metadata(self):
        folder = self.root/'cases'/self.manifest['cases'][0]['case_id']
        shell = folder/'vivado_shell'
        shell.mkdir()
        (shell/'ROUTE_TIMING_PASS').write_text('static accepted marker')
        (shell/'top.bit').write_bytes(b'not-a-real-bitstream')
        remote_worker.atomic_json(shell/'shell_metadata.json', dict(schema_version=2,
            scope='PILOT_ONLY', phase_counter_abi='P3M2', timer_magic=0x50334D32,
            phase_contract='1_load_2_onchip_compute_3_store_0_idle', route_timing_drc='PASS'))
        evidence = self.root/'evidence'
        evidence.mkdir()
        with self.assertRaisesRegex(ValueError, 'new tiled timer ABI'):
            remote_worker.route_gate(folder, evidence)

    def test_board_admission_rejects_old_case_abi(self):
        remote_worker.atomic_json(self.root/'case.json',
            dict(phase_counter_abi='P3M2', phase_contract='1_load_2_onchip_compute_3_store_0_idle'))
        with self.assertRaisesRegex(ValueError, 'tiled P3T1 timer'):
            board_runner._case_identity(self.root)

    def test_missing_operators_have_no_guessed_registers(self):
        self.assertEqual(board_runner._required_arguments('conv3x3'), {'a','weight','bias','output','params'})
        self.assertEqual(board_runner._required_arguments('maxpool'), {'a','output'})

    def test_hls_capacity_separates_numeric_pass_from_physical_estimate(self):
        capacities = dict(BRAM_18K=624, DSP=1728, FF=460800, LUT=230400, URAM=96)
        def tree(bram):
            values = dict(capacities, BRAM_18K=bram)
            xml = '<Report><UserAssignments><Part>xczu7ev-ffvc1156-2-e</Part></UserAssignments><AreaEstimates>'
            xml += '<Resources>' + ''.join(f'<{k}>{v}</{k}>' for k,v in values.items()) + '</Resources>'
            xml += '<AvailableResources>' + ''.join(f'<{k}>{v}</{k}>' for k,v in capacities.items()) + '</AvailableResources>'
            return ET.ElementTree(ET.fromstring(xml + '</AreaEstimates></Report>'))
        self.assertEqual(remote_worker.hls_resource_fit(tree(623))['status'], 'FIT')
        self.assertEqual(remote_worker.hls_resource_fit(tree(624))['status'], 'OVERSIZED')
        self.assertEqual(remote_worker.hls_resource_fit(tree(625))['status'], 'OVERSIZED')
        oversized = tree(623)
        oversized.find('.//Resources/DSP').text = '1729'
        self.assertFalse(remote_worker.hls_resource_fit(oversized)['checks']['DSP'])


if __name__ == '__main__':
    unittest.main()
