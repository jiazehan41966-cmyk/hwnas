"""One authenticated train witness; no valid, repeated benchmark, or test."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(directory):
    root = Path(directory)
    binding = json.loads((root/'SMOKE_BINDING.json').read_text())
    if binding['test_evaluated'] is not False or binding['scope'] != 'single_fixed_train_image':
        raise ValueError('Only sealed single-train-image smoke is permitted')
    for name, digest in binding['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Smoke artifact differs: '+name)
    driver = load_module('smoke_bound_driver', root/'pynq_graph_runner.py')
    power = load_module('smoke_bound_power', root/'smoke_power.py')
    output = root/'smoke_results'
    output.mkdir(exist_ok=False)
    session = driver.GraphSession(root, timeout=900)
    # Keep references in the module retained by the Jupyter cell on failure.
    global RETAINED_SESSION
    RETAINED_SESSION = session
    monitor = power.IioPowerMonitor(output/'power', {
        'deployment_binding_sha256': binding['deployment_binding_sha256'],
        'package_manifest_sha256': binding['package_manifest_sha256'],
        'board_manifest_sha256': hashlib.sha256((root/'board_manifest.json').read_bytes()).hexdigest(),
        'graph_sha256': binding['graph_sha256'],
        'benchmark_manifest_sha256': hashlib.sha256((root/'SMOKE_BINDING.json').read_bytes()).hexdigest(),
    })
    power_status = monitor.start()
    rows = []
    try:
        expected = (root/'expected_layers.bin').read_bytes()
        start = time.monotonic()
        for index, (words, layer) in enumerate(zip(session.descriptors, session.source['layers'])):
            print('SMOKE_NODE_START', binding['network'], index, flush=True)
            elapsed = driver.execute_node(session.core, session.manifest['abi'], session.arena,
                                          session.descriptor, words, session.timeout)
            offset, length, ref = int(words[13]), int(words[25]), int(words[29])
            actual = session.arena[offset:offset+length].tobytes()
            reference = expected[ref:ref+length]
            if len(reference) != length:
                raise ValueError('Truncated integer reference')
            count = sum(a != b for a, b in zip(actual, reference))
            digest = hashlib.sha256(actual).hexdigest()
            rows.append(dict(node=layer['node'], mismatch_count=count,
                             output_sha256=digest, submit_wait_seconds=elapsed))
            (output/'layers.json').write_text(json.dumps(rows, indent=2))
            if count or digest != layer['output_sha256']:
                raise RuntimeError('Layer mismatch: '+layer['node'])
        end = time.monotonic()
        heads = []
        for head in session.source['plan']['outputs']:
            raw = session.arena[head['arena_offset']:head['arena_offset']+head['real_bytes']].tobytes()
            expected_sha = next(v['output_sha256'] for v in session.source['layers'] if v['node'] == head['edge'])
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected_sha:
                raise RuntimeError('Final head overwritten: '+head['level'])
            (output/(head['level']+'.bin')).write_bytes(raw)
            heads.append(dict(level=head['level'], bytes=len(raw), sha256=digest))
        if not session.core.read(0) & 4:
            raise RuntimeError('Core not idle after inference')
        deadline = time.monotonic()+2
        while not monitor.error and (not monitor.samples or monitor.samples[-1]['monotonic_s'] <= end) and time.monotonic() < deadline:
            time.sleep(.05)
        monitor.close()
        power_result = dict(status='NOT_MEASURED', reason=monitor.error or power_status.get('reason'),
                            scope='whole_board_12v; single graph with layer-check overhead; not production benchmark')
        if not monitor.error:
            try:
                if power.validate_iio_sensor(monitor.sensor_path) != monitor.sensor:
                    raise ValueError('Power sensor changed')
                energy = power.paired_energy(monitor.samples, monitor.samples,
                    idle_window=monitor.idle_window, active_window=(start,end), completed_inferences=1)
                power_result.update(status='MEASURED_SINGLE_SMOKE', measurements=energy,
                    sensor=monitor.sensor, idle_window=monitor.idle_window, active_window=[start,end],
                    active_min_w=min(v['power_w'] for v in monitor.samples if start <= v['monotonic_s'] <= end),
                    active_max_w=max(v['power_w'] for v in monitor.samples if start <= v['monotonic_s'] <= end))
            except Exception as exc:
                power_result['reason'] = str(exc)
        (output/'power.json').write_text(json.dumps(power_result, indent=2))
        result = dict(status='SINGLE_TRAIN_SMOKE_PASS', network=binding['network'], seed=42,
            nodes=len(rows), heads=heads, mismatch_count=0, full_board_acceptance='PENDING',
            graph_submit_wait_seconds=sum(v['submit_wait_seconds'] for v in rows),
            graph_with_layer_checks_seconds=end-start, board_clock_mhz=session.clock_mhz,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            input_sha256=session.source['input_sha256'], binding=binding,
            valid_evaluated=False, test_evaluated=False, warmup=0, repetitions=1,
            power=power_result)
        session.close()
        (output/'completion.json').write_text(json.dumps(result, indent=2))
        print('SINGLE_TRAIN_SMOKE_PASS', binding['network'], flush=True)
        return result
    except BaseException as exc:
        monitor.close()
        session.failed = True
        driver.QUARANTINE.append((session.overlay, session.owned))
        (output/'failure.json').write_text(json.dumps(dict(status='FAILED_BUFFERS_RETAINED',
            error=str(exc), network=binding['network'], test_evaluated=False)))
        raise
