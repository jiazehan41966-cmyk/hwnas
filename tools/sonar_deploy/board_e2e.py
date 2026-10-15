"""Bounded host-to-ZCU104 end-to-end benchmark; no training or test access.

The host times one continuous transaction: frozen dataset __getitem__, input
quantization, HTTP transfer, resident-weight graph execution, raw-head return,
and frozen CPU decode/NMS.  The board serves exactly eleven calls in one main
thread.  Board failures retain the PYNQ session in a persistent Jupyter kernel;
the service is stopped, but potentially device-owned buffers are not freed.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import ipaddress
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'real-int8-e2e-v1'
ALLOWED_HOST = '192.168.31.1'
CALLS = 11
QUARANTINE = []


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_manifest(manifest):
    if manifest.get('schema_version') != SCHEMA or manifest.get('test_evaluated') is not False:
        raise ValueError('Invalid benchmark schema or test seal')
    if manifest.get('warmup') != 1 or manifest.get('repetitions') != 10 or manifest.get('batch_size') != 1:
        raise ValueError('Benchmark is fixed to batch1, warmup1, repetitions10')
    for key in ('deployment_binding_sha256', 'package_manifest_sha256', 'graph_sha256',
                'board_manifest_sha256', 'calibration_sha256', 'input_sha256', 'rpc_source_sha256'):
        if not _digest(manifest.get(key)):
            raise ValueError('Invalid benchmark SHA: '+key)
    if manifest.get('network') not in ('D', 'C', 'B1') or manifest.get('seed') != 42:
        raise ValueError('Only seed42 frozen D/C/B1 is benchmarked')
    if manifest.get('input_shape') != [1, 640, 640] or manifest.get('sample', {}).get('split') != 'train':
        raise ValueError('Benchmark must use frozen first train image, CHW grayscale640')
    expected_levels = ['P2', 'P3', 'P4', 'P5'] if manifest['network'] == 'B1' else ['P3', 'P4', 'P5']
    if [head.get('level') for head in manifest.get('heads', [])] != expected_levels:
        raise ValueError('Head level order must match the frozen architecture')
    for head in manifest['heads']:
        side = 640 // (2 ** int(head['level'][1:]))
        if head.get('shape') != [21, side, side] or not _digest(head.get('sha256')):
            raise ValueError('Raw head shape/hash mismatch')
    power = manifest.get('power_monitor')
    if power is not None:
        if (not isinstance(power, dict) or power.get('enabled') is not True or not _digest(power.get('source_sha256')) or
                power.get('sample_rate_hz') != 10 or power.get('idle_seconds') != 5 or
                power.get('sensor_path') != '/sys/bus/iio/devices/iio:device1' or
                power.get('scope') != 'whole_board_12v_input'):
            raise ValueError('Power monitoring must use the frozen read-only whole-board sensor protocol')
    return manifest


def _binding_headers(manifest):
    return {
        'X-Deployment-Sha256': manifest['deployment_binding_sha256'],
        'X-Package-Sha256': manifest['package_manifest_sha256'],
        'X-Graph-Sha256': manifest['graph_sha256'],
        'X-Board-Manifest-Sha256': manifest['board_manifest_sha256'],
        'X-Input-Sha256': manifest['input_sha256'],
        'X-Rpc-Source-Sha256': manifest['rpc_source_sha256'],
    }


class BoundedGraphServer(HTTPServer):
    """Single-threaded HTTPServer: PYNQ infer never runs in a request worker."""
    allow_reuse_address = False

    def __init__(self, address, session, manifest, *, allowed_host=ALLOWED_HOST, read_timeout=30, power_monitor=None):
        self.session, self.manifest = session, validate_manifest(manifest)
        self.allowed_host = str(ipaddress.ip_address(allowed_host))
        self.read_timeout = read_timeout
        self.calls, self.failure, self.rows = 0, None, []
        self.power_monitor, self.power_errors, self.power_finished = power_monitor, [], False
        super().__init__(address, GraphHandler)

    def verify_request(self, request, address):
        return str(ipaddress.ip_address(address[0])) == self.allowed_host

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.read_timeout)
        return connection, address


class GraphHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *_):
        pass

    def do_POST(self):
        server = self.server
        try:
            import numpy as np
            if self.path != '/infer' or server.calls >= CALLS:
                raise ValueError('Unregistered endpoint or benchmark exhausted')
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('Chunked requests are not accepted')
            for key, value in _binding_headers(server.manifest).items():
                if self.headers.get(key) != value:
                    raise ValueError('RPC identity mismatch: '+key)
            if self.headers.get('X-Call-Index') != str(server.calls):
                raise ValueError('Call index replay or reordering')
            size = math.prod(server.manifest['input_shape'])
            if self.headers.get('Content-Length') != str(size):
                raise ValueError('Input length differs from bound INT8 tensor')
            raw = self.rfile.read(size)
            if len(raw) != size or sha_bytes(raw) != server.manifest['input_sha256']:
                raise ValueError('Input payload differs from frozen train witness')
            image = np.frombuffer(raw, dtype=np.int8).reshape(server.manifest['input_shape'])
            _power_method(server, 'begin_inference', server.calls)
            try:
                heads, timing = server.session.infer(image)
            finally:
                _power_method(server, 'end_inference', server.calls)
            if set(heads) != {h['level'] for h in server.manifest['heads']}:
                raise ValueError('Graph returned wrong head levels')
            payload = []
            for head in server.manifest['heads']:
                value = heads[head['level']]
                if len(value) != math.prod(head['shape']) or sha_bytes(value) != head['sha256']:
                    raise ValueError('Raw head differs from integer witness: '+head['level'])
                payload.append(value)
            for key in ('graph_seconds', 'input_graph_capture_seconds', 'board_clock_mhz'):
                value = timing.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Invalid actual board timing: '+key)
            body = b''.join(payload)
            metadata = dict(call_index=server.calls, **timing, test_evaluated=False)
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Measurement', json.dumps(metadata, separators=(',', ':')))
            for key, value in _binding_headers(server.manifest).items():
                self.send_header(key, value)
            self.end_headers(); self.wfile.write(body); self.wfile.flush()
            server.rows.append(metadata); server.calls += 1
        except BaseException as exc:
            server.failure = exc
            try:
                self.send_error(500, 'Bound benchmark failed; retain device-owned buffers')
            except BaseException:
                pass


def _power_method(server, name, *args):
    if server.power_monitor is None:
        return None
    try:
        return getattr(server.power_monitor, name)(*args)
    except Exception as exc:
        # Optional sensor failure is never substituted by an estimated value.
        server.power_errors.append(name+': '+type(exc).__name__+': '+str(exc))
        return None


def _finish_power(server, output):
    if server.power_monitor is None or server.power_finished:
        return
    _power_method(server, 'finish')
    _power_method(server, 'close')
    server.power_finished = True
    if server.power_errors and output:
        path=Path(output).parent/'power/power.json'
        value=read(path) if path.exists() else {}
        value.update(status='NOT_MEASURED',integration_errors=server.power_errors,
                     test_evaluated=False)
        write(path,value)


def serve_session(server, *, idle_timeout=120, max_session_seconds=7200, clock=time.monotonic, output=None):
    """Use in a retained interpreter. Any failure closes RPC and retains session."""
    started = last_call = clock()
    server.timeout = min(1., idle_timeout)
    try:
        while server.calls < CALLS:
            if clock() - last_call >= idle_timeout or clock() - started >= max_session_seconds:
                raise TimeoutError('Bounded board session deadline exceeded')
            previous = server.calls
            server.handle_request()
            if server.failure is not None:
                raise server.failure
            if server.calls != previous:
                last_call = clock()
        server.session.close()
        _finish_power(server,output)
        evidence = dict(status='PASS', successful_calls=server.calls, rows=server.rows,
                        weights_resident=True, test_evaluated=False,
                        graph_sha256=server.manifest['graph_sha256'],
                        package_manifest_sha256=server.manifest['package_manifest_sha256'],
                        board_manifest_sha256=server.manifest['board_manifest_sha256'])
        if output:
            write(output, evidence)
        return evidence
    except BaseException as exc:
        QUARANTINE.append(server.session)
        if output:
            write(output, dict(status='QUARANTINED', successful_calls=server.calls,
                               error=type(exc).__name__+': '+str(exc), test_evaluated=False))
        raise
    finally:
        _finish_power(server,output)
        server.server_close()


def serve_board(board_root, benchmark_manifest, *, bind='0.0.0.0', port=18774,
                node_timeout=180, idle_timeout=120, max_session_seconds=7200):
    """Board entry: call from the retained main-thread Jupyter execution cell."""
    import threading
    try:
        from IPython import get_ipython
    except ImportError as exc:
        raise RuntimeError('Board RPC requires retained main-thread IPython kernel') from exc
    shell = get_ipython()
    if shell is None or not hasattr(shell, 'kernel') or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Board RPC requires retained main-thread IPython kernel; never launch as disposable process')
    board_root = Path(board_root)
    manifest = validate_manifest(read(benchmark_manifest))
    if sha_file(__file__) != manifest['rpc_source_sha256']:
        raise ValueError('RPC adapter source changed')
    if sha_file(board_root/'board_manifest.json') != manifest['board_manifest_sha256']:
        raise ValueError('Board payload binding changed')
    board_binding = read(board_root/'board_manifest.json')
    for name, digest in board_binding['files'].items():
        path = (board_root/name).resolve()
        if not path.is_relative_to(board_root.resolve()) or sha_file(path) != digest:
            raise ValueError('Board payload file changed before driver import: '+name)
    source, graph = read(board_root/'source_manifest.json'), read(board_root/'graph.json')
    if not source.get('real_weights_bound') or sha_file(board_root/'graph.json') != manifest['graph_sha256']:
        raise ValueError('A synthetic or different graph is not admissible')
    wrapper_raw = manifest['package_manifest_text'].encode('utf-8')
    if sha_bytes(wrapper_raw) != manifest['package_manifest_sha256']:
        raise ValueError('Parameter manifest envelope hash mismatch')
    package = json.loads(wrapper_raw)
    for key in ('network', 'seed', 'architecture_id', 'checkpoint_sha256', 'profile_sha256', 'calibration_sha256', 'numeric_contract_hash'):
        if graph['provenance'].get(key) != package.get(key) or package.get(key) != manifest.get(key):
            raise ValueError('Real parameter package identity mismatch: '+key)
    if package['files']['graph.json'] != manifest['graph_sha256']:
        raise ValueError('Graph not bound to parameter wrapper')
    for name, digest in graph['binary_hashes'].items():
        if package['files'].get(name) != digest:
            raise ValueError('Parameter binary digest mismatch: '+name)
    spec = importlib.util.spec_from_file_location('real_int8_graph_driver', board_root/'pynq_graph_runner.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    # Keep both driver and session reachable on failure. Driver owns its own
    # low-level quarantine if a node timed out while the FPGA owns the arena.
    try:
        session = driver.GraphSession(board_root, timeout=node_timeout)
    except BaseException:
        QUARANTINE.append(driver); raise
    monitor=None
    try:
        protocol=manifest.get('power_monitor')
        if protocol:
            power_path=Path(benchmark_manifest).parent/'power.py'
            if sha_file(power_path)!=protocol['source_sha256']:
                raise ValueError('Bound power monitor implementation changed')
            module_spec=importlib.util.spec_from_file_location('bound_real_int8_power',power_path)
            module=importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
            power_binding={
                **{k:manifest[k] for k in ('deployment_binding_sha256','package_manifest_sha256','board_manifest_sha256','graph_sha256')},
                'benchmark_manifest_sha256':sha_file(benchmark_manifest)}
            try:
                monitor=module.IioPowerMonitor(output=board_root/'results/power',binding=power_binding,
                    sensor_path=protocol['sensor_path'],sample_rate_hz=protocol['sample_rate_hz'])
                monitor.start(idle_seconds=protocol['idle_seconds'])
            except Exception as exc:
                if monitor is not None:
                    try:monitor.close()
                    except Exception:pass
                monitor=None
                write(board_root/'results/power/power.json',dict(status='NOT_MEASURED',
                    reason='Optional sensor unavailable: '+type(exc).__name__+': '+str(exc),
                    binding=power_binding,source_sha256=protocol['source_sha256'],test_evaluated=False))
        server = BoundedGraphServer((bind, port), session, manifest,power_monitor=monitor)
    except BaseException:
        if monitor is not None:
            try:monitor.close()
            except Exception:pass
        QUARANTINE.append((driver, session)); raise
    print('E2E_RPC_READY',json.dumps(dict(port=server.server_port,network=manifest['network'],
        calls=CALLS,graph_sha256=manifest['graph_sha256'],allowed_host=ALLOWED_HOST)),flush=True)
    return serve_session(server, idle_timeout=idle_timeout, max_session_seconds=max_session_seconds,
                         output=board_root/'results/e2e_rpc_session.json')


class GraphClient:
    def __init__(self, url, manifest, timeout=3600):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'http' or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
            raise ValueError('Expected credential-free direct board HTTP endpoint')
        address = ipaddress.ip_address(parsed.hostname)
        if not address.is_private:
            raise ValueError('Board endpoint must be a private IP, not an external service')
        self.url, self.manifest, self.timeout = url.rstrip('/')+'/infer', validate_manifest(manifest), timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def infer(self, image, index):
        import numpy as np
        if not isinstance(image, np.ndarray) or image.dtype != np.int8 or list(image.shape) != self.manifest['input_shape']:
            raise ValueError('Host must submit bound CHW INT8')
        raw = image.copy(order='C').tobytes()
        if sha_bytes(raw) != self.manifest['input_sha256']:
            raise ValueError('Frozen host preprocessing differs from witness')
        headers = {**_binding_headers(self.manifest), 'X-Call-Index': str(index),
                   'Content-Type': 'application/octet-stream'}
        request = urllib.request.Request(self.url, data=raw, headers=headers, method='POST')
        length = sum(math.prod(h['shape']) for h in self.manifest['heads'])
        with self.opener.open(request, timeout=self.timeout) as response:
            for key, value in _binding_headers(self.manifest).items():
                if response.headers.get(key) != value:
                    raise ValueError('Response package binding mismatch')
            if response.headers.get('Content-Length') != str(length):
                raise ValueError('Raw head transport length mismatch')
            metadata = json.loads(response.headers['X-Measurement'])
            raw = response.read(length+1)
        if len(raw) != length or metadata.get('call_index') != index or metadata.get('test_evaluated') is not False:
            raise ValueError('Incomplete or reordered graph response')
        heads, offset = {}, 0
        for head in self.manifest['heads']:
            size = math.prod(head['shape']); value = raw[offset:offset+size]; offset += size
            if sha_bytes(value) != head['sha256']:
                raise ValueError('Returned raw head does not match integer witness')
            heads[head['level']] = value
        return heads, metadata


def _summary(values):
    ordered = sorted(values)
    return dict(median_ms=statistics.median(values)*1000,
                p95_ms=(ordered[8]*.45+ordered[9]*.55)*1000)


def benchmark(preprocess, decode, infer, manifest, *, clock=time.perf_counter, progress=None):
    """preprocess and decode are real synchronous CPU callbacks, timed together."""
    validate_manifest(manifest)
    rows = []
    measured_begin = None
    for index in range(CALLS):
        begin = clock()                 # before disk read / preprocessing
        if index == 1:
            measured_begin = begin
        image = preprocess()
        heads, board_timing = infer(image, index)
        predictions = decode(heads)     # frozen CPU decode + NMS finishes here
        end = clock()
        elapsed = end - begin           # continuous host wall clock, never sum
        graph = board_timing['graph_seconds']
        if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(graph) or graph <= 0 or elapsed < graph:
            raise ValueError('Invalid continuous timing or incomparable board duration')
        rows.append(dict(call_index=index, warmup=index == 0, end_to_end_seconds=elapsed,
                         graph_seconds=graph, input_graph_capture_seconds=board_timing['input_graph_capture_seconds'],
                         board_clock_mhz=board_timing['board_clock_mhz'],
                         detection_count=len(predictions[0]['scores'])))
        if progress:
            progress(rows)              # evidence I/O is outside the timed transaction
    measured = rows[1:]
    graph = [r['graph_seconds'] for r in measured]
    end_values = [r['end_to_end_seconds'] for r in measured]
    timing = dict(graph_seconds=graph, end_to_end_seconds=end_values,
        graph_median_ms=_summary(graph)['median_ms'], graph_p95_ms=_summary(graph)['p95_ms'],
        end_to_end_median_ms=_summary(end_values)['median_ms'], end_to_end_p95_ms=_summary(end_values)['p95_ms'])
    if len({r['board_clock_mhz'] for r in rows}) != 1:
        raise ValueError('Board clock changed during fixed benchmark')
    return dict(schema_version=SCHEMA, status='PASS', scope='host_CPU_preprocess_RPC_board_CPU_decode_NMS',
        host_role='server CPU; not ARM-only local deployment', warmup=1, repetitions=10, batch_size=1,
        timing_scope='host_preprocess_rpc_board_graph_receive_cpu_decode',
        input_size=640, precision='INT8', quantization_policy='W8A8',
        weights_resident=True, rows=rows, samples=measured, timing=timing,
        board_latency_ms=timing['end_to_end_median_ms'], throughput_fps=1000/timing['end_to_end_median_ms'],
        actual_clock_mhz=rows[0]['board_clock_mhz'], test_evaluated=False,
        mismatch_count=0, integer_witness_sample_count=1,
        measured_window_seconds=end-measured_begin,
        completed_window_fps=10/(end-measured_begin),
        completed_window_scope='ten measured host transactions including inter-call progress persistence',
        expected_head_sha256={head['level']:head['sha256'] for head in manifest['heads']},
        deployment_binding_sha256=manifest['deployment_binding_sha256'],
        package_manifest_sha256=manifest['package_manifest_sha256'], graph_sha256=manifest['graph_sha256'],
        board_manifest_sha256=manifest['board_manifest_sha256'], sample=manifest['sample'])


def _host_components(root, network, smoke_evidence=None):
    sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'tools/sonar_analysis'), str(ROOT/'scripts')]
    import numpy as np
    import torch
    import baseline as frozen
    from phase5a_quantization.common import baseline, build_network
    from hwnas_fpga.training.detection_trainer import decode_detection_predictions
    from tools.sonar_deploy import admission
    torch.set_num_threads(2)
    if smoke_evidence is None:
        decision = admission.action_admission(root, network, 'board', 42)
    else:
        from tools.sonar_deploy.performance_admission import check
        decision = check(root, network, smoke_evidence)
    if not decision['allowed']:
        raise RuntimeError('Board benchmark admission failed: '+json.dumps(decision))
    binding = admission.load_binding(root)
    expected = next(r for r in binding['runs'] if r['network'] == network and r['seed'] == 42)
    result = admission.load_result(root, expected)
    package = Path(result['package_manifest_path']).parent
    graph = read(package/'graph.json')
    calibration = read(binding['source_files']['calibration_manifest']['path'])
    if calibration.get('split') != 'train' or calibration.get('test_evaluated') is not False or len(calibration['entries']) != 64:
        raise ValueError('Original64 train calibration entries required')
    config = baseline(); data_root = frozen.resolve(config['dataset']['data_dir'])
    dataset = frozen.make_loader(config, data_root, 'train', 42, 0, train=False).dataset
    actual = [p.relative_to(dataset.root).as_posix() for p in dataset.samples[:64]]
    if actual != [r['image'] for r in calibration['entries']]:
        raise ValueError('Calibration ordering changed')
    entry = calibration['entries'][0]
    for key in ('image', 'annotation'):
        path = Path(entry[key])
        if path.is_absolute() or path.parts[0] != 'train' or '..' in path.parts or sha_file(data_root/path) != entry[key+'_sha256']:
            raise ValueError('Benchmark sample is not bound train-only data')
    model, architecture_id = build_network(network)
    if architecture_id != expected['architecture_id']:
        raise ValueError('Candidate metadata changed')
    loss_config = model.loss_config(config)
    proto = config['evaluation_protocol']
    def preprocess():
        image, _ = dataset[0]  # frozen image read/resize/normalization, no augmentation
        if image.device.type != 'cpu' or list(image.shape) != [1,640,640]:
            raise ValueError('Frozen host preprocessing must return CPU CHW640')
        return torch.round(image.to(torch.float64)/(2.**graph['inputs']['image']['exp'])).clamp(-128,127).to(torch.int8).numpy()
    def decode(raw_heads):
        heads = []
        for level, edge in graph['outputs'].items():
            c,h,w = graph['edges'][edge]['shape']
            integer = np.frombuffer(raw_heads[level], dtype=np.int8).reshape(h,w,c).transpose(2,0,1).copy()
            value = torch.from_numpy(integer).unsqueeze(0).to(torch.float32) * (2.**graph['edges'][edge]['exp'])
            heads.append(value)
        with torch.inference_mode():
            return decode_detection_predictions({'predictions': tuple(heads)}, config=loss_config,
                score_threshold=proto['ap_decode_score_threshold'], nms_iou_threshold=proto['nms_iou_threshold'],
                pre_nms_topk=proto['pre_nms_topk'], max_detections=proto['post_nms_max_detections_per_image'])
    return binding, result, graph, entry, preprocess, decode


def prepare_manifest(root, network, board_manifest, output, *, power_monitor=False, smoke_evidence=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    binding, result, graph, entry, preprocess, _ = (_host_components(root, network) if smoke_evidence is None
                                                  else _host_components(root, network, smoke_evidence))
    board = read(board_manifest)
    package = Path(result['package_manifest_path']).parent
    if not board.get('real_weights_bound') or board['graph_sha256'] != sha_file(package/'graph.json'):
        raise ValueError('Timing target must be the matching real trained board package')
    trace_root = root/network/'seed_42/parity/train_traces'
    trace = read(trace_root/'TRACE_MANIFEST.json')
    if trace.get('graph_sha256') != board['graph_sha256'] or trace.get('calibration_manifest_sha256') != result['calibration_sha256']:
        raise ValueError('Fixed train witness belongs to another graph/calibration')
    input_raw = preprocess().copy(order='C').tobytes()
    first = trace['frames'][0]
    import numpy as np
    expected_input = np.load(trace_root/first['input_path'], allow_pickle=False).copy(order='C').tobytes()
    if input_raw != expected_input:
        raise ValueError('Timed preprocessing is not equivalent to integer witness input')
    heads = []
    for level, edge in graph['outputs'].items():
        path = trace_root/first['folder']/(level+'_hwc.bin')
        relative = path.relative_to(trace_root).as_posix()
        digest = sha_file(path)
        if trace['files'].get(relative) != digest:
            raise ValueError('Expected head witness hash changed')
        heads.append(dict(level=level,shape=graph['edges'][edge]['shape'],sha256=digest))
    manifest = dict(schema_version=SCHEMA, **{k: result[k] for k in (
        'network','seed','architecture_id','checkpoint_sha256','profile_sha256','calibration_sha256','numeric_contract_hash')},
        deployment_binding_sha256=sha_file(root/'DEPLOYMENT_BINDING.json'),
        package_manifest_sha256=result['package_manifest_sha256'],
        package_manifest_text=Path(result['package_manifest_path']).read_bytes().decode('utf-8'),
        graph_sha256=board['graph_sha256'],board_manifest_sha256=sha_file(board_manifest),
        rpc_source_sha256=sha_file(__file__),
        input_shape=[1,640,640],input_sha256=sha_bytes(input_raw),heads=heads,
        sample=dict(split='train',calibration_index=0,**entry),warmup=1,repetitions=10,batch_size=1,test_evaluated=False)
    if power_monitor:
        manifest['power_monitor']=dict(enabled=True,source_sha256=sha_file(Path(__file__).with_name('power.py')),
            sample_rate_hz=10,idle_seconds=5,sensor_path='/sys/bus/iio/devices/iio:device1',scope='whole_board_12v_input')
    validate_manifest(manifest)
    if output.exists() and read(output) != manifest:
        raise ValueError('Frozen benchmark manifest changed; use an explicit new attempt directory')
    write(output,manifest)
    return manifest


def run_host(root, network, manifest_path, url, output, timeout=3600, smoke_evidence=None):
    root = Path(root).resolve(); manifest = validate_manifest(read(manifest_path))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Benchmark output already exists; never silently replace measured timing')
    binding,result,graph,entry,preprocess,decode = (_host_components(root, network) if smoke_evidence is None
                                                  else _host_components(root, network, smoke_evidence))
    if manifest['network'] != network or manifest['deployment_binding_sha256'] != sha_file(root/'DEPLOYMENT_BINDING.json') or manifest['package_manifest_sha256'] != result['package_manifest_sha256']:
        raise ValueError('Host benchmark manifest changed network/package binding')
    canonical = output.parent/'benchmark_manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_manifest = Path(manifest_path).read_bytes()
    if canonical.exists() and canonical.read_bytes() != raw_manifest:
        raise ValueError('Another frozen benchmark manifest already occupies the output directory')
    if not canonical.exists():
        canonical.write_bytes(raw_manifest)
    client = GraphClient(url, manifest, timeout)
    def progress(rows):
        write(output.parent/'e2e_progress.json',dict(status='IN_PROGRESS',calls=len(rows),rows=rows,
            benchmark_manifest_sha256=sha_file(canonical),test_evaluated=False))
    try:
        report = benchmark(preprocess,decode,client.infer,manifest,progress=progress)
    except BaseException as exc:
        write(output.parent/'e2e_failure.json',dict(status='FAILED',error=type(exc).__name__+': '+str(exc),
            benchmark_manifest_sha256=sha_file(canonical),test_evaluated=False))
        raise
    report.update(benchmark_manifest_sha256=sha_file(manifest_path), source_sha256=sha_file(__file__))
    write(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','host'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--network',choices=['D','C','B1'],required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--board-manifest',type=Path)
    parser.add_argument('--url')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--timeout',type=float,default=3600)
    parser.add_argument('--power-monitor',action='store_true')
    parser.add_argument('--smoke-evidence',type=Path,help='Explicit performance-only gate; never admits full validation')
    args=parser.parse_args()
    if args.action=='prepare':
        if not args.board_manifest:parser.error('--board-manifest required')
        value=prepare_manifest(args.root,args.network,args.board_manifest,args.manifest,power_monitor=args.power_monitor,smoke_evidence=args.smoke_evidence)
    else:
        if not args.url or not args.output:parser.error('--url and --output required')
        value=run_host(args.root,args.network,args.manifest,args.url,args.output,args.timeout,smoke_evidence=args.smoke_evidence)
    print(json.dumps({k:v for k,v in value.items() if k not in ('package_manifest_text','rows','heads')},indent=2))


if __name__=='__main__':main()
