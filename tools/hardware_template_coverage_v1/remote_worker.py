"""Serial, immutable-input FPGA template coverage worker; runs only explicitly selected stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

HLS = '/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls'
VIVADO = '/opt/Xilinx/Vivado/2022.2/bin/vivado'
PHASE_CONTRACT = '1_load_2_onchip_compute_3_store_repeated_tiles_0_idle'
TIMER_MAGIC = 0x50335431


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def relative_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError(f'unsafe relative filename: {name}')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f'file escapes bundle: {name}')
    return path


def canonical_case_sha(case):
    case = {key: value for key, value in case.items() if key != 'case_sha256'}
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def verify_inputs(root):
    manifest = json.loads((root / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != 't6_tiled_source_v1':
        raise ValueError('new template coverage source schema required')
    if not manifest.get('files') or not manifest.get('cases'):
        raise ValueError('empty source manifest')
    if len(manifest['cases']) != 65:
        raise ValueError('exactly 65 source-bound primitive cases required')
    for name, expected in manifest['files'].items():
        if digest(relative_file(root, name)) != expected:
            raise ValueError(f'input hash mismatch: {name}')
    native = json.loads((root / 'native_summary.json').read_text(encoding='utf-8-sig'))
    if native.get('status') != 'PASS':
        raise ValueError('native parity summary is not PASS')
    if native.get('source_manifest_sha256') != digest(root / 'source_manifest.json'):
        raise ValueError('native summary refers to a different source manifest')
    native_rows = {row['case_id']: row for row in native['cases']}
    seen = set()
    for row in manifest['cases']:
        name = row['case_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in seen:
            raise ValueError('invalid or repeated case_id')
        seen.add(name)
        case = json.loads((root / 'cases' / name / 'case.json').read_text(encoding='utf-8-sig'))
        if case.get('phase_counter_abi') != 'P3T1' or case.get('phase_contract') != PHASE_CONTRACT:
            raise ValueError(f'tiled timer ABI/phase contract differs: {name}')
        expected = canonical_case_sha(case)
        if case.get('case_sha256') != expected or row['case_sha256'] != expected:
            raise ValueError(f'case manifest hash mismatch: {name}')
        parity = native_rows.get(name, {})
        if parity.get('status') != 'PASS' or parity.get('case_sha256') != expected:
            raise ValueError(f'native parity missing or stale: {name}')
    return manifest


def terminate_owned_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def substage_times(log):
    """Tcl wall-clock intervals are separate from process monotonic intervals."""
    starts, durations = {}, {}
    for action, name, milliseconds in re.findall(
            r'PILOT_STAGE_(START|END)\s+(\w+)\s+(\d+)', log):
        stamp = int(milliseconds)
        if action == 'START':
            starts[name] = stamp
        elif name in starts and stamp >= starts[name]:
            durations[name] = (stamp - starts.pop(name)) / 1000.0
    return {'completed_stage_seconds': durations, 'unfinished_stage_starts_ms': starts,
            'clock': 'Tcl clock milliseconds; process envelope uses time.monotonic_ns'}


def hls_resource_fit(tree):
    """Reject impossible core estimates; this never proves routed shell fit."""
    part = tree.findtext('.//UserAssignments/Part')
    if part != 'xczu7ev-ffvc1156-2-e':
        raise ValueError('HLS resource capacity report refers to another part')
    expected = dict(BRAM_18K=624, DSP=1728, FF=460800, LUT=230400, URAM=96)
    used, available, checks = {}, {}, {}
    for name, capacity in expected.items():
        values = tree.findall('.//AreaEstimates/Resources/' + name)
        capacities = tree.findall('.//AreaEstimates/AvailableResources/' + name)
        if len(values) != 1 or len(capacities) != 1:
            raise ValueError('missing/ambiguous HLS resource field: ' + name)
        used[name], available[name] = int(values[0].text), int(capacities[0].text)
        if used[name] < 0 or available[name] != capacity:
            raise ValueError('invalid HLS resource usage/capacity: ' + name)
        checks[name] = (used[name] < capacity if name == 'BRAM_18K' else used[name] <= capacity)
    return dict(status='FIT' if all(checks.values()) else 'OVERSIZED',
                used=used, available=available, checks=checks,
                bram_rule='BRAM_18K < 624; remaining shell requirements checked by route',
                evidence_layer='HLS_ESTIMATES_ONLY_NOT_ROUTED_INTEGRATED_FIT',
                required_next_gates=['complete_route_timing_drc_and_bitstream', 'bound_board_correctness'])


def hls_gate(case, evidence):
    results = list(case.glob('**/csim/build/c_result.json'))
    if len(results) != 1:
        raise ValueError(f'expected one C-simulation result, got {len(results)}')
    result = json.loads(results[0].read_text())
    fixtures = (case / 'fixtures.txt').read_text().split()
    if not fixtures or result.get('fixtures') != len(fixtures) or result.get('mismatches') != 0:
        raise ValueError('C-simulation numeric gate failed')
    witnesses = []
    for fixture in fixtures:
        expected = relative_file(case, fixture + '_expected.bin')
        actual = relative_file(results[0].parent, fixture + '_actual.bin')
        expected_sha, actual_sha = digest(expected), digest(actual)
        if expected_sha != actual_sha:
            raise ValueError(f'C-simulation output hash mismatch: {fixture}')
        witnesses.append({'fixture': fixture, 'bytes': expected.stat().st_size,
                          'expected_sha256': expected_sha, 'actual_sha256': actual_sha})
    reports = sorted(case.glob('**/syn/report/*_csynth.xml'))
    top_reports = [path for path in reports if path.name == 'pilot_kernel_csynth.xml']
    if not top_reports:
        raise ValueError('top-level HLS synthesis XML missing')
    exports = sorted(case.glob('**/impl/ip/component.xml')) + sorted(case.glob('*.zip'))
    if not exports:
        raise ValueError('HLS IP export artifact missing')
    leaves = {}
    tree = ET.parse(top_reports[0])
    capacity = hls_resource_fit(tree)
    for node in tree.iter():
        if not list(node) and node.text and node.text.strip():
            leaves.setdefault(node.tag, []).append(node.text.strip())
    rtl = sorted(case.glob('**/syn/verilog/pilot_kernel.v'))
    if not rtl:
        raise ValueError('top-level synthesized RTL missing')
    ports = [line.strip() for line in rtl[0].read_text(errors='replace').splitlines()
             if re.match(r'\s*(input|output|inout)\b', line)]
    for path in list(top_reports[0].parent.glob('*.rpt')) + list(top_reports[0].parent.glob('*.xml')):
        shutil.copy2(path, evidence / path.name)
    shutil.copy2(rtl[0], evidence / rtl[0].name)
    shutil.copy2(results[0], evidence / 'c_result.json')
    atomic_json(evidence / 'csim_parity.json', {'status': 'PASS', 'fixtures': witnesses})
    atomic_json(evidence / 'rtl_ports.json', {'source': str(rtl[0]), 'sha256': digest(rtl[0]), 'ports': ports})
    atomic_json(evidence / 'hls_resource_fit.json', capacity)
    return {'csim': result, 'csim_sha256_parity': 'PASS', 'csynth_xml': leaves,
            'resource_fit': capacity,
            'exports': [{'path': str(path.relative_to(case)), 'sha256': digest(path)} for path in exports],
            'rtl_port_count': len(ports)}


def route_gate(case, evidence):
    markers = list(case.glob('**/ROUTE_TIMING_DRC_PASS')) + list(case.glob('**/ROUTE_TIMING_PASS'))
    if len(markers) != 1:
        raise ValueError('exactly one route timing acceptance marker required')
    marker = markers[0]
    bits = sorted(case.glob('**/*.bit'))
    if not bits:
        raise ValueError('routed bitstream missing')
    metadata_path = case / 'vivado_shell' / 'shell_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if (metadata.get('schema_version') != 3 or
            metadata.get('scope') != 'TEMPLATE_COVERAGE_ONLY' or
            metadata.get('phase_counter_abi') != 'P3T1' or
            metadata.get('timer_magic') != TIMER_MAGIC or
            metadata.get('phase_contract') != PHASE_CONTRACT or
            metadata.get('route_timing_drc') != 'PASS'):
        raise ValueError('route artifact does not bind the new tiled timer ABI')
    reports = []
    for pattern in ('timing*.rpt', 'utilization*.rpt', 'route_status*.rpt', '*clock*.rpt'):
        reports.extend(case.glob('**/' + pattern))
    if not any('timing' in path.name for path in reports):
        raise ValueError('route timing report missing')
    for path in reports:
        shutil.copy2(path, evidence / path.name)
    shutil.copy2(marker, evidence / marker.name)
    for path in case.glob('**/shell_metadata.json'):
        shutil.copy2(path, evidence / path.name)
    artifacts = []
    for path in bits + sorted(case.glob('**/*.hwh')):
        artifacts.append({'path': str(path.relative_to(case)), 'sha256': digest(path),
                          'bytes': path.stat().st_size})
    atomic_json(evidence / 'bitstream_manifest.json', {'files': artifacts})
    return {'timing_marker': 'PASS', 'artifacts': artifacts}


def run_stage(root, case_row, stage, timeout, source_sha):
    name = case_row['case_id']
    case = root / 'cases' / name
    generated = case / ('hls_project' if stage == 'hls' else 'vivado_shell')
    if generated.exists():
        raise FileExistsError(f'preserve generated evidence; prepare a new attempt: {generated}')
    parent = root / 'results' / name / stage
    parent.mkdir(parents=True, exist_ok=True)
    attempt = len(list(parent.glob('run[0-9][0-9][0-9]'))) + 1
    evidence = parent / f'run{attempt:03d}'
    evidence.mkdir(exist_ok=False)
    command = ([HLS, '-f', 'run_hls.tcl'] if stage == 'hls' else
               [VIVADO, '-mode', 'batch', '-source', 'build_shell.tcl', '-notrace', '-tclargs', str(case)])
    status = {'case_id': name, 'case_sha256': case_row['case_sha256'], 'stage': stage,
              'command': command, 'cwd': str(case), 'status': 'RUNNING',
              'source_manifest_sha256': source_sha, 'started_utc': utc(),
              'monotonic_start_ns': time.monotonic_ns(), 'timeout_seconds': timeout,
              'evidence_directory': str(evidence.relative_to(root))}
    atomic_json(parent / 'latest.json', status)
    event(root / 'results' / 'events.jsonl', status)
    process = None
    try:
        with (evidence / 'process.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status['pid'] = process.pid
            atomic_json(parent / 'latest.json', status)
            try:
                status['returncode'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_owned_group(process)
                status['returncode'] = process.returncode
                raise TimeoutError(f'{stage} exceeded {timeout} seconds')
        if status['returncode'] != 0:
            raise RuntimeError(f'{stage} process exited {status["returncode"]}')
        status['gate'] = hls_gate(case, evidence) if stage == 'hls' else route_gate(case, evidence)
        if stage == 'hls' and status['gate']['resource_fit']['status'] != 'FIT':
            raise ValueError('HLS synthesis completed but core resource estimate is OVERSIZED')
        status['status'] = 'PASS'
    except Exception as error:
        status.update(status='FAILED', error=f'{type(error).__name__}: {error}')
    except BaseException:
        status.update(status='INTERRUPTED', error='worker interrupted')
        raise
    finally:
        if process is not None and process.poll() is None:
            terminate_owned_group(process)
        status.update(finished_utc=utc(), monotonic_end_ns=time.monotonic_ns())
        status['wall_seconds'] = (status['monotonic_end_ns'] - status['monotonic_start_ns']) / 1e9
        log_path = evidence / 'process.log'
        if log_path.exists():
            status['substage_timing'] = substage_times(log_path.read_text(errors='replace'))
            status['log_sha256'] = digest(log_path)
        timing_rows = []
        for pattern in ('hls_stages.jsonl', '**/shell_stages.jsonl', '**/stage_elapsed.jsonl'):
            for path in case.glob(pattern):
                shutil.copy2(path, evidence / path.name)
                try:
                    timing_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
                except (ValueError, OSError) as error:
                    status.setdefault('timing_parse_errors', []).append(str(error))
        status['substage_records'] = timing_rows
        atomic_json(evidence / 'status.json', status)
        atomic_json(parent / 'latest.json', status)
        event(root / 'results' / 'events.jsonl', status)
    return status


def latest(root, name, stage):
    path = root / 'results' / name / stage / 'latest.json'
    return json.loads(path.read_text()) if path.exists() else {}


def route_case_ids(root, manifest):
    if manifest['files'].get('inventory.json') != digest(root / 'inventory.json'):
        raise ValueError('route whitelist must be bound to source manifest')
    data = json.loads((root / 'inventory.json').read_text(encoding='utf-8'))
    names = data.get('representative_case_ids', [])
    all_names = {row['case_id'] for row in manifest['cases']}
    if len(names) != 13 or len(set(names)) != 13 or not set(names) <= all_names:
        raise ValueError('exactly 13 unique representative route cases required')
    return set(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('hls', 'route', 'all'), required=True)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--hls-timeout', type=int, default=7200)
    parser.add_argument('--route-timeout', type=int, default=7200)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--case', dest='case_id')
    args = parser.parse_args()
    if args.retry_failed:
        raise ValueError('failed inputs and generated evidence are immutable; prepare a new attempt')
    root = args.root.resolve()
    if not re.fullmatch(r'hardware_template_coverage_20261013_attempt\d+', root.name):
        raise ValueError('unexpected experiment root')
    if root.parent != Path('/home/jiajia/experiments'):
        raise ValueError('root outside isolated experiment parent')
    if args.limit < 1 or not 1 <= args.hls_timeout <= 7200 or not 1 <= args.route_timeout <= 7200:
        raise ValueError('invalid bounded work limit or timeout')
    manifest = verify_inputs(root)
    authorized_names = ({row['case_id'] for row in manifest['cases']} if args.stage == 'hls'
                        else route_case_ids(root, manifest))
    if args.case_id and args.case_id not in authorized_names:
        raise ValueError('requested case is outside this stage whitelist')
    (root / 'results').mkdir(exist_ok=True)
    lock = root / 'worker.lock'
    with lock.open('x') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'started_utc': utc()}))
    source_sha = digest(root / 'source_manifest.json')
    queue = {'status': 'RUNNING', 'stage': args.stage, 'limit': args.limit, 'pid': os.getpid(),
             'started_utc': utc(), 'source_manifest_sha256': source_sha, 'cases': []}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        atomic_json(root / 'results' / 'queue_status.json', queue)
        for row in manifest['cases']:
            name = row['case_id']
            if name not in authorized_names:
                continue
            if args.case_id and name != args.case_id:
                continue
            final = latest(root, name, 'hls' if args.stage == 'hls' else 'route')
            if final.get('status') == 'PASS':
                continue
            if final.get('status') in ('FAILED', 'INTERRUPTED') and not args.retry_failed:
                continue
            if len(queue['cases']) >= args.limit:
                break
            if args.stage == 'route' and latest(root, name, 'hls').get('status') != 'PASS':
                continue
            outcome = {'case_id': name, 'stages': []}
            queue['cases'].append(outcome)
            for stage in (('hls', 'route') if args.stage == 'all' else (args.stage,)):
                old = latest(root, name, stage)
                if old.get('status') == 'PASS':
                    outcome['stages'].append({'stage': stage, 'status': 'PASS', 'reused': True})
                    continue
                if old.get('status') in ('FAILED', 'INTERRUPTED') and not args.retry_failed:
                    outcome['stages'].append({'stage': stage, 'status': 'SKIPPED_PRIOR_FAILURE'})
                    break
                verify_inputs(root)
                result = run_stage(root, row, stage, args.hls_timeout if stage == 'hls'
                                   else args.route_timeout, source_sha)
                outcome['stages'].append({key: result[key] for key in ('stage', 'status', 'wall_seconds')})
                atomic_json(root / 'results' / 'queue_status.json', queue)
                if result['status'] != 'PASS':
                    break
            atomic_json(root / 'results' / 'queue_status.json', queue)
        queue['status'] = ('COMPLETE_WITH_FAILURES' if any(stage['status'] not in ('PASS',)
                           for row in queue['cases'] for stage in row['stages']) else 'COMPLETE')
    except BaseException as error:
        queue.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                     error=f'{type(error).__name__}: {error}')
        raise
    finally:
        queue['finished_utc'] = utc()
        atomic_json(root / 'results' / 'queue_status.json', queue)
        if lock.exists() and json.loads(lock.read_text()).get('pid') == os.getpid():
            lock.unlink()
    print(json.dumps(queue), flush=True)


if __name__ == '__main__':
    main()
