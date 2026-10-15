"""Evidence admission must reject corruption and unsupported scope changes."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('detector_evidence_audit', ROOT / 'tools/detector_preflight/audit_evidence.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

TIMING = '''| Design Timing Summary
WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns)
1.118 0.000 0 76028 0.010 0.000 0 76028 1.166 0.000 0 26383
All user specified timing constraints are met.
| Clock Summary
Clock Waveform(ns) Period(ns) Frequency(MHz)
clk_pl_0 {0.000 2.667} 5.333 187.512
'''
ROUTE = '''# of routable nets..... : 38542 :
# of fully routed nets..... : 38542 :
# of nets with routing errors..... : 0 :
'''
RESOURCES = '''| Design State : Routed
| Device       : xczu7ev-ffvc1156-2-e
| h2_dma_bd_wrapper | (top) | 19987 | 18427 | 1334 | 226 | 23354 | 9 | 3 | 0 | 11 |
| core | h2_dma_bd_core_0 | 7772 | 7637 | 80 | 55 | 8834 | 1 | 1 | 0 | 11 |
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else value.encode())


def dump(path, value):
    write(path, json.dumps(value))


@pytest.fixture
def evidence(tmp_path):
    execution = tmp_path / 'execution'
    board, build = execution / 'board_evidence', execution / 'dma_attempt02'
    source = {'sources': {}}
    for name in ('h2_dma.cpp', 'build_dma_bd.tcl', 'numeric_v1.h', 'run_dma.tcl', 'tb_dma.cpp'):
        write(build / name, 'immutable source ' + name)
        source['sources'][name] = module.sha256(build / name)
    files = {}
    rows = []
    for case, mode in (('pw', 0), ('fpn', 1)):
        source[case] = dict(mode=mode, output_bytes=4, files={})
        for contract, filename in (('numeric_contract_hash', 'numeric_contract_v1.json'),
                                   ('interface_contract_hash', 'interface_contract_v1.json')):
            obj = json.loads((ROOT / 'reports/fpga_h2_contract' / filename).read_text())
            source[case][contract] = hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for name in ('input', 'reference', 'low', 'numeric', 'weights', 'biases'):
            path = build / case / (name + '.bin')
            write(path, b'\x01\x02\x03\x04')
            source[case]['files'][name + '.bin'] = module.sha256(path)
            files[case + '/' + name + '.bin'] = module.sha256(path)
        actual = board / (case + '_board_output.bin')
        write(actual, b'\x01\x02\x03\x04')
        rows.append(dict(case=case, mode=mode, bytes=4, mismatch_count=0,
                         end_to_end_submit_wait_ns=1000000,
                         latency_scope='single_transaction_no_cache_or_allocation',
                         output_sha256=module.sha256(actual)))
    dump(build / 'manifest.json', source)
    files['source_manifest.json'] = module.sha256(build / 'manifest.json')
    for name, content in dict(h2_dma__bit=b'bitstream', h2_dma__hwh=b'ip metadata',
                              timing__rpt=TIMING, route_status__rpt=ROUTE,
                              utilization_hier__rpt=RESOURCES).items():
        name = name.replace('__', '.')
        write(board / name, content)
        files[name] = module.sha256(board / name)
    driver = tmp_path / 'pynq_dma_runner.py'
    write(driver, 'driver source')
    files['pynq_dma_runner.py'] = module.sha256(driver)
    abi = {name.rstrip('_'): 16 + 12 * i for i, name in enumerate(module.ABI_MACROS)}
    header = '\n'.join(f'#define XH2_DMA_CONTROL_ADDR_{macro} {hex(abi[name.rstrip("_")])}' for name, macro in module.ABI_MACROS.items())
    write(build / 'evidence/core_abi.h', header)
    dump(board / 'board_manifest.json', dict(files=files, full_detector=False, timing_status='PASS',
         abi=abi, header_sha256=module.sha256(build / 'evidence/core_abi.h')))
    dump(board / 'board_results.json', rows)
    dump(board / 'timeout_probe.json', dict(status='PASS', scope='S2MM_without_producer; idle_core',
         dma_channels_halted=True, core_idle=True, timeout_seconds=0.05))
    dump(execution / 'status.json', dict(route='RUNNING', board_dma='PENDING'))
    return execution, driver


def run(evidence, tmp_path):
    execution, driver = evidence
    return module.audit(execution, tmp_path / 'audit', driver)


def test_valid_witness_preserves_scope_and_original_status(evidence, tmp_path):
    old = (evidence[0] / 'status.json').read_bytes()
    result = run(evidence, tmp_path)
    assert result['status'] == 'PASS_ISOLATED_DMA_WITNESS'
    assert result['resources']['shell_difference']['LUT'] == 12215
    assert result['resources']['shell_difference']['BRAM36_equivalent'] == 9
    assert result['trained_checkpoint_bound'] is False and result['full_detector'] is False
    assert result['timeout_witness']['proves_stuck_axi_master_recovery'] is False
    assert result['timing']['clocks'][0]['frequency_MHz'] == 187.512
    assert (evidence[0] / 'status.json').read_bytes() == old
    assert len(result['snapshots']) >= 30


@pytest.mark.parametrize('damage', ['output', 'source', 'missing_driver', 'forged_count', 'scope', 'abi'])
def test_broken_evidence_fails_closed(evidence, tmp_path, damage):
    execution, driver = evidence
    board = execution / 'board_evidence'
    if damage == 'output':
        write(board / 'pw_board_output.bin', b'\x01\x02\x03\x05')
    elif damage == 'source':
        write(execution / 'dma_attempt02/h2_dma.cpp', 'changed')
    elif damage == 'missing_driver':
        evidence = execution, tmp_path / 'nonexistent_driver.py'
    elif damage == 'forged_count':
        rows = json.loads((board / 'board_results.json').read_text())
        rows[0]['bytes'] = 5
        dump(board / 'board_results.json', rows)
    else:
        manifest = json.loads((board / 'board_manifest.json').read_text())
        if damage == 'scope':
            manifest['full_detector'] = True
        else:
            manifest['abi']['mode'] += 4
        dump(board / 'board_manifest.json', manifest)
    result = run(evidence, tmp_path)
    assert result['status'] == 'FAIL'
    assert any(row['status'] == 'FAIL' for row in result['checks'])


def test_changed_output_with_forged_hash_still_fails_byte_comparison(evidence, tmp_path):
    board = evidence[0] / 'board_evidence'
    write(board / 'fpn_board_output.bin', b'\x01\x02\x03\x05')
    rows = json.loads((board / 'board_results.json').read_text())
    rows[1]['output_sha256'] = module.sha256(board / 'fpn_board_output.bin')
    dump(board / 'board_results.json', rows)
    result = run(evidence, tmp_path)
    assert result['status'] == 'FAIL'
    assert next(row for row in result['checks'] if row['name'] == 'board_output_bytes_and_parity')['status'] == 'FAIL'


def test_timing_and_route_reject_nonpassing_reports():
    with pytest.raises(ValueError):
        module.parse_timing(TIMING.replace('1.118', '-0.001'))
    with pytest.raises(ValueError):
        module.parse_timing(TIMING.replace('0.010', '-0.001'))
    with pytest.raises(ValueError):
        module.parse_route(ROUTE.replace('errors..... : 0', 'errors..... : 1'))
    with pytest.raises(ValueError):
        module.parse_route(ROUTE.replace('routed nets..... : 38542', 'routed nets..... : 38541'))


def test_manifest_path_cannot_escape_snapshot(tmp_path):
    with pytest.raises(ValueError):
        module.safe_child(tmp_path, '../outside')


def test_refuses_existing_output_without_overwriting(evidence, tmp_path):
    output = tmp_path / 'audit'
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.audit(evidence[0], output, evidence[1])
