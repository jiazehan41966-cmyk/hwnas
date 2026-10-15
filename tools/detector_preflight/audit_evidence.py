"""Audit and snapshot existing PW/FPN board witnesses without accessing hardware.

This does not establish a trained detector, repeated latency, or power result.
The historical execution directory is read-only; output must be a new directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_COLUMNS = ('LUT', 'logic_LUT', 'LUTRAM', 'SRL', 'FF', 'RAMB36', 'RAMB18', 'URAM', 'DSP')
ABI_MACROS = dict(weights='WEIGHTS_DATA', biases='BIASES_DATA', numeric='NUMERIC_DATA',
                  low='LOW_R_DATA', mode='MODE_DATA', return_='AP_RETURN')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_child(root, name):
    root = Path(root).resolve()
    child = (root / name).resolve()
    if not child.is_relative_to(root) or child == root:
        raise ValueError('manifest path escapes its evidence directory: ' + name)
    return child


def parse_timing(text):
    section = text.split('| Design Timing Summary', 1)[1]
    section = section.split('WNS(ns)', 1)[1]
    row = next(line.split() for line in section.splitlines()
               if re.match(r'^\s*-?\d+\.\d+\s+-?\d+\.\d+', line))
    if len(row) != 12:
        raise ValueError('unexpected timing summary columns')
    values = [float(value) for value in row]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('nonfinite timing summary')
    result = dict(zip(('WNS_ns', 'TNS_ns', 'setup_failing', 'setup_total', 'WHS_ns',
                       'THS_ns', 'hold_failing', 'hold_total', 'WPWS_ns', 'TPWS_ns',
                       'pulse_failing', 'pulse_total'), values))
    if ('All user specified timing constraints are met.' not in text or
            any(result[key] < 0 for key in ('WNS_ns', 'WHS_ns', 'WPWS_ns')) or
            any(result[key] != 0 for key in ('TNS_ns', 'THS_ns', 'TPWS_ns',
                                             'setup_failing', 'hold_failing', 'pulse_failing')) or
            result['setup_total'] <= 0):
        raise ValueError('timing did not pass')
    clocks = re.findall(r'^\s*(\S+)\s+\{[^}]+\}\s+([\d.]+)\s+([\d.]+)\s*$', text, re.MULTILINE)
    if not clocks:
        raise ValueError('missing routed clock summary')
    result['clocks'] = [dict(name=name, period_ns=float(period), frequency_MHz=float(frequency))
                        for name, period, frequency in clocks]
    if any(clock['period_ns'] <= 0 or clock['frequency_MHz'] <= 0 for clock in result['clocks']):
        raise ValueError('invalid clock summary')
    return result


def parse_route(text):
    values = {}
    for key, label in (('routable_nets', 'routable nets'), ('fully_routed_nets', 'fully routed nets'),
                       ('routing_errors', 'nets with routing errors')):
        match = re.search(r'# of ' + label + r'\.+\s*:\s*(\d+)', text)
        if not match:
            raise ValueError('missing route count: ' + label)
        values[key] = int(match.group(1))
    if values['routing_errors'] or values['routable_nets'] <= 0 or values['routable_nets'] != values['fully_routed_nets']:
        raise ValueError('routing incomplete or erroneous')
    return values


def parse_resources(text):
    if '| Design State : Routed' not in text or '| Device       : xczu7ev-ffvc1156-2-e' not in text:
        raise ValueError('not a routed ZCU104 witness resource report')
    rows = {}
    for line in text.splitlines():
        parts = [part.strip() for part in line.split('|')[1:-1]]
        if len(parts) != 11 or parts[0] not in ('h2_dma_bd_wrapper', 'core'):
            continue
        if parts[0] in rows:
            raise ValueError('ambiguous resource hierarchy')
        rows[parts[0]] = dict(zip(RESOURCE_COLUMNS, map(int, parts[2:])))
    top, core = rows['h2_dma_bd_wrapper'], rows['core']
    shell = {key: top[key] - core[key] for key in RESOURCE_COLUMNS}
    if any(value < 0 for row in (top, core, shell) for value in row.values()):
        raise ValueError('inconsistent hierarchy resources')
    for row in (top, core, shell):
        row['BRAM36_equivalent'] = row['RAMB36'] + row['RAMB18'] / 2
    return dict(top=top, core=core, shell_difference=shell,
                scope='this routed PW/FPN witness only; not a full-detector reservation')


def audit(execution, output, driver=None):
    execution, output = Path(execution).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    board, build = execution / 'board_evidence', execution / 'dma_attempt02'
    driver = Path(driver or ROOT / 'tools/detector_preflight/pynq_dma_runner.py').resolve()
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  execution=str(execution), status='FAIL', scope='H2 PW/FPN DMA witnesses only',
                  full_detector=False, trained_checkpoint_bound=False, measured_power=False,
                  repeated_latency=False, new_hardware_execution=False,
                  checks=[], snapshots=[], findings=[])

    def check(name, action):
        try:
            value = action()
            result['checks'].append(dict(name=name, status='PASS'))
            return value
        except (OSError, ValueError, KeyError, IndexError, TypeError, StopIteration) as exc:
            result['checks'].append(dict(name=name, status='FAIL', error=str(exc)))
            return None

    def snapshot(path, destination, expected=None):
        path = Path(path)
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError('SHA256 mismatch: ' + str(path))
        target = safe_child(output / 'evidence', destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(target) != actual:
            raise ValueError('source changed while copying: ' + str(path))
        result['snapshots'].append(dict(source=str(path), snapshot=target.relative_to(output).as_posix(),
                                        sha256=actual, bytes=target.stat().st_size))
        return target

    def load(path, destination, expected=None):
        return json.loads(snapshot(path, destination, expected).read_text(encoding='utf-8'))

    manifest = check('board_manifest', lambda: load(board / 'board_manifest.json', 'board_manifest.json'))
    resolved = {}
    if manifest is not None:
        def resolve_payload():
            required = {'source_manifest.json', 'pynq_dma_runner.py', 'timing.rpt', 'route_status.rpt',
                        'utilization_hier.rpt', 'h2_dma.bit', 'h2_dma.hwh'}
            required.update(f'{case}/{name}.bin' for case in ('pw', 'fpn')
                            for name in ('input', 'reference', 'low', 'numeric', 'weights', 'biases'))
            if not required.issubset(manifest['files']):
                raise ValueError('board manifest lacks required payload members')
            if manifest['full_detector'] is not False or manifest['timing_status'] != 'PASS':
                raise ValueError('witness scope/timing mismatch')
            for name, expected in manifest['files'].items():
                candidates = [safe_child(board, name), safe_child(build, name)]
                if name == 'source_manifest.json':
                    candidates.append(build / 'manifest.json')
                if name == 'pynq_dma_runner.py':
                    candidates.append(driver)
                existing = [path for path in candidates if path.is_file()]
                if not existing:
                    raise ValueError('missing payload file: ' + name)
                # Never mask corruption of the authoritative board copy with a fallback.
                resolved[name] = snapshot(existing[0], 'payload/' + name, expected)
            return dict(files=len(resolved), all_manifest_payload_files_verified=True)
        result['payload'] = check('payload_hashes', resolve_payload)

        def source_binding():
            source = json.loads(resolved['source_manifest.json'].read_text())
            for name, expected in source['sources'].items():
                snapshot(safe_child(build, name), 'build_sources/' + name, expected)
            if not {'h2_dma.cpp', 'build_dma_bd.tcl', 'numeric_v1.h', 'run_dma.tcl', 'tb_dma.cpp'}.issubset(source['sources']):
                raise ValueError('missing build source binding')
            for label, mode in (('pw', 0), ('fpn', 1)):
                if source[label]['mode'] != mode:
                    raise ValueError('source case mode mismatch')
                for name, expected in source[label]['files'].items():
                    if sha256(resolved[label + '/' + name]) != expected:
                        raise ValueError('source/board manifest disagreement')
                for contract, filename in (('numeric_contract_hash', 'numeric_contract_v1.json'),
                                            ('interface_contract_hash', 'interface_contract_v1.json')):
                    contract_path = ROOT / 'reports/fpga_h2_contract' / filename
                    contract_object = json.loads(contract_path.read_text())
                    canonical = json.dumps(contract_object, sort_keys=True, separators=(',', ':')).encode()
                    if hashlib.sha256(canonical).hexdigest() != source[label][contract]:
                        raise ValueError('H2 canonical contract mismatch')
                    if label == 'pw':
                        snapshot(contract_path, 'contracts/' + filename)
            return source
        source = check('build_source_and_h2_binding', source_binding)

        def abi_binding():
            header = snapshot(build / 'evidence/core_abi.h', 'core_abi.h', manifest['header_sha256'])
            macros = dict(re.findall(r'#define\s+XH2_DMA_CONTROL_ADDR_(\w+)\s+(0x[0-9a-fA-F]+)', header.read_text()))
            expected = {name.rstrip('_'): int(macros[macro], 16) for name, macro in ABI_MACROS.items()}
            if expected != manifest['abi']:
                raise ValueError('board ABI differs from generated core header')
            return expected
        result['abi'] = check('generated_abi_binding', abi_binding)
        result['timing'] = check('routed_timing', lambda: parse_timing(resolved['timing.rpt'].read_text()))
        result['route'] = check('route_completion', lambda: parse_route(resolved['route_status.rpt'].read_text()))
        result['resources'] = check('routed_resource_scope', lambda: parse_resources(resolved['utilization_hier.rpt'].read_text()))

        def board_parity():
            if source is None:
                raise ValueError('source binding required before board parity admission')
            rows = load(board / 'board_results.json', 'board_results.json')
            if len(rows) != 2 or {row['case'] for row in rows} != {'pw', 'fpn'}:
                raise ValueError('expected exactly one PW and one FPN observation')
            verified = []
            for row in rows:
                label = row['case']
                actual_path = snapshot(board / (label + '_board_output.bin'), label + '_board_output.bin', row['output_sha256'])
                actual, reference = actual_path.read_bytes(), resolved[label + '/reference.bin'].read_bytes()
                count = len(actual)
                if count != len(reference) or count != source[label]['output_bytes'] or count != row['bytes']:
                    raise ValueError('output byte counts disagree: ' + label)
                mismatches = sum(a != b for a, b in zip(actual, reference))
                if mismatches or row['mismatch_count'] != mismatches or row['mode'] != source[label]['mode']:
                    raise ValueError('board parity/mode mismatch: ' + label)
                latency = row['end_to_end_submit_wait_ns']
                if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
                    raise ValueError('invalid single-transaction observation')
                if row['latency_scope'] != 'single_transaction_no_cache_or_allocation':
                    raise ValueError('unexpected latency measurement scope')
                verified.append(dict(case=label, bytes=count, mismatch_count=mismatches,
                                     output_sha256=sha256(actual_path), single_transaction_ms=latency / 1e6,
                                     latency_scope=row['latency_scope']))
            return verified
        result['board_outputs'] = check('board_output_bytes_and_parity', board_parity)

    def timeout_scope():
        probe = load(board / 'timeout_probe.json', 'timeout_probe.json')
        if (probe['status'] != 'PASS' or probe['scope'] != 'S2MM_without_producer; idle_core' or
                probe['dma_channels_halted'] is not True or probe['core_idle'] is not True or
                not 0 < probe['timeout_seconds'] <= 1):
            raise ValueError('timeout witness fields/scope invalid')
        return dict(**probe, proves_stuck_axi_master_recovery=False)
    result['timeout_witness'] = check('recorded_idle_core_timeout_witness', timeout_scope)

    old = check('historical_status_snapshot', lambda: load(execution / 'status.json', 'historical_status.json'))
    if old and (old.get('route') == 'RUNNING' or old.get('board_dma') == 'PENDING'):
        result['findings'].append('Historical status predates the retained route/board results; original file left unchanged.')
    result['findings'].extend([
        'PW/FPN use H2 synthetic reference vectors; no trained checkpoint or selected architecture is bound.',
        'Top-minus-core shell resources belong only to this routed witness and must not enter the detector cost LUT as a general reservation.',
        'Single transaction timings exclude cache maintenance and allocation; no repeated end-to-end detector latency or power measurement.',
        'Host graph schedule replay shares H2 arithmetic and synthetic weights; it is not independent trained full-network integer parity.',
        'Timing applies to the actual clock summary in the routed report; requested source clock frequency is not a measured or routed frequency.',
        'This audit verifies retained payload/source identities and results; it does not independently reproduce synthesis or prove bitstream derivation from the copied build script.',
    ])
    if all(row['status'] == 'PASS' for row in result['checks']):
        result['status'] = 'PASS_ISOLATED_DMA_WITNESS'
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    lines = ['# Existing detector DMA evidence audit', '', f"Status: {result['status']}", '',
             'Local read-only audit; copied and hash-bound evidence is under `evidence/`.', '',
             '## Verified checks', '']
    lines += [f"- {row['name']}: {row['status']}" + (': ' + row['error'] if 'error' in row else '') for row in result['checks']]
    if result.get('board_outputs'):
        lines += ['', '## Existing board observations', '']
        lines += [f"- {row['case']}: {row['bytes']} bytes; {row['mismatch_count']} mismatches; single transaction {row['single_transaction_ms']:.6f} ms." for row in result['board_outputs']]
    if result.get('resources'):
        shell = result['resources']['shell_difference']
        lines += ['', f"Witness shell difference: LUT {shell['LUT']}, FF {shell['FF']}, BRAM36 equivalents {shell['BRAM36_equivalent']}, DSP {shell['DSP']}."]
    if result.get('timing'):
        lines += ['', f"Routed WNS: {result['timing']['WNS_ns']} ns."]
        lines += [f"- Routed clock {clock['name']}: {clock['period_ns']} ns, {clock['frequency_MHz']} MHz." for clock in result['timing']['clocks']]
    lines += ['', '## Limits and remaining work', ''] + ['- ' + item for item in result['findings']]
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, default=ROOT / 'results/detector_execution_20260922')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--driver', type=Path)
    args = parser.parse_args()
    result = audit(args.execution, args.output, args.driver)
    print(json.dumps(dict(status=result['status'], checks=result['checks'], output=str(args.output)), indent=2))
    return 0 if result['status'] == 'PASS_ISOLATED_DMA_WITNESS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
