"""Finite isolated stem C-simulation/CSYNTH queue. No training/route/board work."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

TOOL = '/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls'
PART = 'xczu7ev-ffvc1156-2-e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_case(case):
    key = json.loads((case / 'measurement_key.json').read_text())
    if key['LEGAL_MAPPING_CHECK'] != 'PASS' or key['part'] != PART or key['tool_version'] != '2022.2':
        raise ValueError('Unregistered or illegal hardware case')
    if key['target_clock_ns'] != 5 or key['PE'] not in (8, 16, 32) or key['SIMD'] != 1:
        raise ValueError('The bounded pilot permits only the registered stem mappings')
    for field in ('source_files', 'witness_files'):
        for name, digest in key[field].items():
            target = (case / name).resolve()
            if not target.is_relative_to(case.resolve()) or sha(target) != digest:
                raise ValueError('Case source or numeric witness drift: ' + name)
    return key


def parse_report(path):
    xml = ET.parse(path).getroot()
    if xml.findtext('.//UserAssignments/Part') != PART or xml.findtext('.//ReportVersion/Version') != '2022.2':
        raise ValueError('Report platform or version mismatch')
    if float(xml.findtext('.//UserAssignments/TargetClockPeriod')) != 5:
        raise ValueError('Report target clock mismatch')
    def number(tag):
        value = float(xml.findtext(tag))
        if not math.isfinite(value) or value < 0:
            raise ValueError('Unknown or invalid report metric')
        return value
    metrics = dict(cycles=number('.//SummaryOfOverallLatency/Worst-caseLatency'),
                   estimated_clock_ns=number('.//SummaryOfTimingAnalysis/EstimatedClockPeriod'),
                   LUT=number('.//AreaEstimates/Resources/LUT'),
                   DSP=number('.//AreaEstimates/Resources/DSP'),
                   BRAM18=number('.//AreaEstimates/Resources/BRAM_18K'))
    metrics['latency_us'] = metrics['cycles'] * 5 / 1000
    metrics['timing_eligible'] = metrics['estimated_clock_ns'] <= 5
    metrics['resource_screen_eligible'] = (metrics['LUT'] <= 161280 and metrics['DSP'] <= 1209
                                           and metrics['BRAM18'] <= 436)
    return metrics


def execute_case(case, timeout_seconds):
    key = verify_case(case)
    started = time.monotonic()
    with (case / 'hls.log').open('w') as log:
        process = subprocess.Popen([TOOL, '-f', 'run.tcl'], cwd=case, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 'TIMEOUT'
    output = case / 'project/solution1/csim/build/hardware_output.bin'
    report = case / 'project/solution1/syn/report/closure_kernel_csynth.xml'
    parity = output.is_file() and output.stat().st_size > 0 and sha(output) == sha(case / 'reference.bin')
    metrics, error = None, None
    if report.is_file():
        try:
            metrics = parse_report(report)
        except (ValueError, TypeError, ET.ParseError) as exception:
            error = str(exception)
    passed = code == 0 and parity and metrics is not None
    result = dict(case_name=case.name, status='PASS_CSIM_AND_CSYNTH' if passed else 'FAILED_OR_INCOMPLETE',
                  returncode=code, wall_seconds=time.monotonic() - started, PE=key['PE'], SIMD=key['SIMD'],
                  source_hash=key['source_hash'], numeric_contract_hash=key['numeric_contract_hash'],
                  interface_contract_hash=key['interface_contract_hash'], csim_binary_exact=parity,
                  csim_reference_sha256=sha(case / 'reference.bin'),
                  csim_output_sha256=sha(output) if output.is_file() else None,
                  source_report_sha256=sha(report) if report.is_file() else None,
                  metrics=metrics, report_parse_error=error, measurement_scope='OP_LEVEL_HLS_CSYNTH',
                  board_measured=False, full_network_measured=False, test_images_accessed=False,
                  note='PASS verifies arithmetic and synthesis; timing/resource admission is separate')
    save(case / 'result.json', result)
    return result


def main():
    root = Path(__file__).resolve().parent
    if (root / 'status.json').exists():
        raise FileExistsError('Existing queue status; do not overwrite prior probe')
    staging = json.loads((root / 'staging_manifest.json').read_text())
    for name, expected in staging['files'].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target) != expected:
            raise ValueError('Staged source mismatch: ' + name)
    cases = sorted((root / 'cases').iterdir(), key=lambda p: json.loads((p / 'measurement_key.json').read_text())['PE'])
    if not 1 <= len(cases) <= 3:
        raise ValueError('A finite batch of one to three registered stem probes is required')
    if len({verify_case(case)['PE'] for case in cases}) != len(cases):
        raise ValueError('Duplicate parallelism settings in one probe batch')
    for case in cases:
        verify_case(case)
    status = dict(status='RUNNING', active=None, completed=[], started_utc=datetime.now(timezone.utc).isoformat(),
                  timeout_per_case_seconds=900, concurrency=1, case_count=len(cases), training_started=False,
                  route_started=False, board_started=False, test_accessed=False)
    for case in cases:
        status['active'] = case.name
        save(root / 'status.json', status)
        result = execute_case(case, 900)
        status['completed'].append(result)
        print(json.dumps(result), flush=True)
        save(root / 'status.json', status)
    status.update(status='FINITE_PROBE_COMPLETE', active=None, ended_utc=datetime.now(timezone.utc).isoformat())
    save(root / 'status.json', status)


if __name__ == '__main__':
    main()
