"""Execute a manifest-bound H2 graph on the serial DDR FPGA core.

No software arithmetic fallback. Retain the Jupyter kernel on failure because
the core can still own the arena. The mapping is for functional evidence only.
"""
import hashlib
import json
from pathlib import Path
import time

QUARANTINE = []
REQUANT_EXECUTORS = ('serial_ddr_graph_v2_requant', 'serial_ddr_graph_v2_requant_timing02')


def validate_real_executor(manifest, source):
    executor = manifest['executor']
    if executor.get('executor_version') not in REQUANT_EXECUTORS:
        raise ValueError('explicit supported requant capable image required')
    if executor != source.get('implementation'):
        raise ValueError('board executor differs from bound source implementation')
    if executor['executor_version'].endswith('_timing02') and executor.get('requant_pipeline') != 'off':
        raise ValueError('timing02 requires the bound requant pipeline-off implementation')
    return executor


class GraphSession:
    """Persistent authenticated weights for real host-to-board RPC timing.

    infer receives already preprocessed/quantized CHW and returns raw HWC heads;
    the caller owns preprocessing/decode and measures actual end-to-end time.
    """
    def __init__(self, directory, timeout=180):
        import numpy as np
        from pynq import Overlay, allocate, MMIO, Clocks
        self.np=np;self.root=Path(directory);self.timeout=timeout
        self.owned=[];self.failed=False;self.closed=False;self.overlay=None
        self.manifest=json.loads((self.root/'board_manifest.json').read_text())
        for name,digest in self.manifest['files'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('payload hash mismatch: '+name)
        self.source=json.loads((self.root/'source_manifest.json').read_text())
        if not self.source.get('real_weights_bound'):
            raise ValueError('RPC requires real parameter binding')
        executor=validate_real_executor(self.manifest,self.source)
        self.descriptors=np.fromfile(self.root/'descriptors.bin',dtype='<i4').reshape(-1,32)
        initial=np.fromfile(self.root/'arena.bin',dtype=np.int8)
        if len(initial)>executor['arena_capacity_bytes'] or len(self.descriptors)!=len(self.source['layers']):
            raise ValueError('payload arena/node capacity mismatch')
        if any(int(row[0]) not in executor['supported_opcodes'] or int(row[22])!=len(initial) or int(row[26])!=0x48324731 for row in self.descriptors):
            raise ValueError('descriptor ABI mismatch')
        try:
            self.overlay=Overlay(str(self.root/'h2_graph.bit'),download=False)
            info=self.overlay.ip_dict['core']
            if 'h2_graph' not in info['type']:raise ValueError('wrong core identity')
            self.overlay.download();self.clock_mhz=float(Clocks.fclk0_mhz)
            if abs(self.clock_mhz-self.manifest['implemented_clock_mhz'])>.1:
                raise RuntimeError('board clock differs from timing constraint')
            self.core=MMIO(info['phys_addr'],info['addr_range'])
            self.arena=allocate(shape=initial.shape,dtype=np.int8);self.owned.append(self.arena)
            self.descriptor=allocate(shape=(32,),dtype=np.int32);self.owned.append(self.descriptor)
            self.arena[:]=initial;self.arena.flush()
        except BaseException:
            self.failed=True;QUARANTINE.append((self.overlay,self.owned));raise

    def infer(self, image):
        if self.failed or self.closed:raise RuntimeError('session failed/closed')
        allocation=self.source['plan']['allocations']['image']
        image=self.np.asarray(image)
        if image.dtype!=self.np.int8 or list(image.shape)!=allocation['shape']:
            raise ValueError('image must be matched INT8 CHW')
        start=time.monotonic()
        try:
            data=image.transpose(1,2,0).copy().reshape(-1);offset=allocation['offset']
            self.arena[offset:offset+len(data)]=data;self.arena.flush()
            begin=time.monotonic()
            for words in self.descriptors:
                execute_node(self.core,self.manifest['abi'],self.arena,self.descriptor,words,self.timeout,invalidate=False)
            self.arena.invalidate()
            heads={output['level']:self.arena[output['arena_offset']:output['arena_offset']+output['real_bytes']].tobytes()
                   for output in self.source['plan']['outputs']}
            return heads,dict(graph_seconds=time.monotonic()-begin,
                              input_graph_capture_seconds=time.monotonic()-start,board_clock_mhz=self.clock_mhz)
        except BaseException:
            self.failed=True;QUARANTINE.append((self.overlay,self.owned));raise

    def close(self):
        if self.closed:return
        self.closed=True
        if self.failed:return
        if not self.core.read(0)&4:
            self.failed=True;QUARANTINE.append((self.overlay,self.owned))
            raise RuntimeError('core still owns buffers; retaining session')
        for buffer in reversed(self.owned):buffer.freebuffer()


def pointer(core, offset, address):
    core.write(offset, address & 0xffffffff)
    core.write(offset + 4, address >> 32)


def execute_node(core, abi, arena, descriptor, words, timeout, invalidate=True):
    if not core.read(0) & 4:
        raise RuntimeError('core not idle before node')
    descriptor[:] = words
    descriptor.flush()
    for name, address in {
        'arena': int(arena.physical_address),
        'weights': int(arena.physical_address) + int(words[27]),
        'biases': int(arena.physical_address) + int(words[28]),
        'descriptor': int(descriptor.physical_address),
    }.items():
        pointer(core, abi[name], address)
    start = time.monotonic()
    core.write(0, 1)
    done = False
    while True:
        control = core.read(0)
        done = done or bool(control & 2)
        if done and control & 4:
            break
        if time.monotonic() - start > timeout:
            raise TimeoutError('graph node; retain device-owned arena')
        time.sleep(0.002)
    elapsed = time.monotonic() - start
    status = core.read(abi['return'])
    if status:
        raise RuntimeError('graph core status ' + str(status))
    if invalidate:
        arena.invalidate()
    return elapsed


def validate_frames(root, source):
    """Full-detector batch is exactly fixed train8 plus valid99; never test."""
    path = root/'frames.json'
    if not path.is_file():
        return None
    spec = json.loads(path.read_text())
    if spec.get('test_evaluated') is not False or spec.get('graph_sha256') != source['graph_sha256']:
        raise ValueError('frame test seal or graph binding mismatch')
    if not source.get('real_weights_bound'):
        raise ValueError('full valid batch requires authenticated trained weights')
    provenance=source.get('export_binding',{}).get('provenance',{})
    if provenance and spec.get('calibration_manifest_sha256')!=provenance.get('calibration_sha256'):
        raise ValueError('frame calibration manifest differs from parameter provenance')
    rows = spec['frames']
    if [r['split'] for r in rows].count('train') != 8 or [r['split'] for r in rows].count('valid') != 99:
        raise ValueError('expected exactly train8 and valid99')
    if any(r['split'] not in ('train', 'valid') for r in rows):
        raise ValueError('only train/valid frames allowed')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate frame sample identity')
    train_ids = [r['sample_id'] for r in rows if r['split'] == 'train']
    if train_ids != spec['calibration_first8_sample_ids']:
        raise ValueError('train witness order must match frozen calibration prefix')
    for name, digest in spec['files'].items():
        target = (root/name).resolve()
        if not target.is_relative_to(root.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('frame file SHA mismatch: ' + name)
    levels = {o['level'] for o in source['plan']['outputs']}
    layer_names = {o['node'] for o in source['layers']}
    for row in rows:
        if row['input'] not in spec['files'] or set(row['expected_heads']) != levels:
            raise ValueError('frame input/head binding incomplete')
        if row['split'] == 'train' and set(row.get('expected_layers', {})) != layer_names:
            raise ValueError('train trace needs every node hash')
    if spec.get('warmup', 1) != 1 or spec.get('repetitions', 10) != 10:
        raise ValueError('frozen benchmark is one warmup and ten repetitions')
    return spec


def execute_frames(root, source, frames, core, abi, arena, descriptor, descriptors, timeout, results):
    import numpy as np
    image = source['plan']['allocations']['image']
    c, h, w = image['shape']
    records = []

    def put_image(row):
        value = np.load(root/row['input'], allow_pickle=False)
        if value.dtype != np.int8 or list(value.shape) != [c, h, w]:
            raise ValueError('frame must be quantized CHW INT8')
        raw = value.transpose(1, 2, 0).copy().reshape(-1)
        begin = image['offset']
        arena[begin:begin+len(raw)] = raw
        arena.flush()

    for index, row in enumerate(frames['frames']):
        start = time.monotonic(); put_image(row)
        layer_times = []; trace = []
        for words, layer in zip(descriptors, source['layers']):
            elapsed = execute_node(core, abi, arena, descriptor, words, timeout, invalidate=row['split'] == 'train')
            layer_times.append(elapsed)
            if row['split'] == 'train':
                offset, length = int(words[13]), int(words[25])
                digest = hashlib.sha256(arena[offset:offset+length].tobytes()).hexdigest()
                if digest != row['expected_layers'][layer['node']]:
                    raise RuntimeError('train trace parity mismatch: '+row['sample_id']+' '+layer['node'])
                trace.append(dict(node=layer['node'], sha256=digest, mismatch_count=0))
        arena.invalidate()
        folder = results/'frames'/str(index); folder.mkdir(parents=True, exist_ok=False)
        heads = []
        for output in source['plan']['outputs']:
            offset, length = output['arena_offset'], output['real_bytes']
            raw = arena[offset:offset+length].tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row['expected_heads'][output['level']]:
                raise RuntimeError('valid/train final head mismatch: '+row['sample_id']+' '+output['level'])
            (folder/(output['level']+'.bin')).write_bytes(raw)
            # Device graph writes real values only; transport padding is explicitly zeroed.
            (folder/(output['level']+'.padded.bin')).write_bytes(raw+bytes(output['tail_zero_bytes']))
            heads.append(dict(level=output['level'], bytes=length, sha256=digest, padding_zero=True))
        record = dict(index=index, sample_id=row['sample_id'], split=row['split'], heads=heads,
                      layer_trace=trace, graph_submit_wait_seconds=sum(layer_times),
                      frame_input_and_capture_seconds=time.monotonic()-start, mismatch_count=0)
        records.append(record)
        (results/'frames.json').write_text(json.dumps(records, indent=2))
        print('GRAPH_FRAME', row['split'], index, row['sample_id'], flush=True)
    benchmark_row = next(row for row in frames['frames'] if row['split'] == 'train')
    measured = []
    for index in range(11):
        start = time.monotonic(); put_image(benchmark_row)
        graph_start = time.monotonic()
        for words in descriptors:
            execute_node(core, abi, arena, descriptor, words, timeout, invalidate=False)
        arena.invalidate()
        # Include final head transfer, exclude decode/NMS which is performed on host.
        for output in source['plan']['outputs']:
            arena[output['arena_offset']:output['arena_offset']+output['real_bytes']].tobytes()
        if index:
            measured.append(dict(graph_seconds=time.monotonic()-graph_start,
                                 input_graph_capture_seconds=time.monotonic()-start))
    times = np.asarray([row['graph_seconds'] for row in measured])
    benchmark = dict(batch_size=1, input_size=640, precision='INT8', warmup=1, repetitions=10,
                     weights_resident=True, samples=measured, median_seconds=float(np.median(times)),
                     p95_seconds=float(np.percentile(times,95)), fps=float(1/np.median(times)),
                     scope='serial graph submit/wait and raw head capture; excludes preprocessing and decode/NMS',
                     end_to_end_status='PENDING_HOST_PREPROCESS_DECODE_TIMING')
    (results/'benchmark.json').write_text(json.dumps(benchmark, indent=2))
    return dict(train_frames=8, valid_frames=99, mismatch_count=0, test_evaluated=False)


def run(directory, timeout=180):
    import numpy as np
    from pynq import Overlay, allocate, MMIO, Clocks

    root = Path(directory)
    manifest = json.loads((root/'board_manifest.json').read_text())
    for name, sha in manifest['files'].items():
        target = (root/name).resolve()
        if not target.is_relative_to(root.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise ValueError('payload hash mismatch ' + name)
    source = json.loads((root/'source_manifest.json').read_text())
    frames = validate_frames(root, source)
    initial = np.fromfile(root/'arena.bin', dtype=np.int8)
    descriptors = np.fromfile(root/'descriptors.bin', dtype='<i4').reshape(-1,32)
    expected = (root/'expected_layers.bin').read_bytes()
    if len(descriptors) != len(source['layers']):
        raise ValueError('layer count mismatch')
    if source.get('real_weights_bound'):
        executor = validate_real_executor(manifest,source)
        if len(initial) > executor['arena_capacity_bytes']:
            raise ValueError('real graph exceeds synthesized arena capacity')
        if any(int(row[0]) not in executor['supported_opcodes'] for row in descriptors):
            raise ValueError('descriptor opcode absent from synthesized executor')
    if any(int(row[22]) != len(initial) or int(row[26]) != 0x48324731 for row in descriptors):
        raise ValueError('descriptor arena/magic binding mismatch')
    overlay = Overlay(str(root/'h2_graph.bit'), download=False)
    info = overlay.ip_dict['core']
    if 'h2_graph' not in info['type']:
        raise ValueError('wrong core identity')
    overlay.download()
    clock_mhz=float(Clocks.fclk0_mhz)
    if abs(clock_mhz-manifest['implemented_clock_mhz'])>0.1:
        raise RuntimeError('board clock differs from implemented timing constraint')
    core = MMIO(info['phys_addr'], info['addr_range'])
    owned = []
    try:
        arena = allocate(shape=initial.shape, dtype=np.int8)
        owned.append(arena)
        descriptor = allocate(shape=(32,), dtype=np.int32)
        owned.append(descriptor)
        arena[:] = initial
        arena.flush()
        report = []
        results = root/'results'
        results.mkdir(exist_ok=False)
        for index, (words, layer) in enumerate(zip(descriptors, source['layers'])):
            print('GRAPH_START',index,layer['node'],flush=True)
            elapsed = execute_node(core, manifest['abi'], arena, descriptor, words, timeout)
            offset, length, ref_offset = int(words[13]), int(words[25]), int(words[29])
            actual = arena[offset:offset+length].tobytes()
            reference = expected[ref_offset:ref_offset+length]
            count = int(np.count_nonzero(np.frombuffer(actual,dtype=np.uint8) != np.frombuffer(reference,dtype=np.uint8)))
            sha = hashlib.sha256(actual).hexdigest()
            row = dict(index=index, node=layer['node'], bytes=length,
                       mismatch_count=count, output_sha256=sha, submit_wait_seconds=elapsed)
            report.append(row)
            (results/'layers.json').write_text(json.dumps(report,indent=2))
            print('GRAPH_NODE', json.dumps(row), flush=True)
            if count or sha != layer['output_sha256']:
                raise RuntimeError('FPGA layer parity mismatch')
        heads = []
        for output in source['plan']['outputs']:
            offset, length = output['arena_offset'], output['real_bytes']
            data = arena[offset:offset+length].tobytes()
            expected_sha = next(layer['output_sha256'] for layer in source['layers']
                                if layer['node'] == output['edge'])
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != expected_sha:
                raise RuntimeError('final head overwritten: ' + output['level'])
            (results/(output['level']+'.bin')).write_bytes(data)
            heads.append(dict(level=output['level'], bytes=length, sha256=actual_sha,
                              preserved_until_graph_end=True))
        (results/'heads.json').write_text(json.dumps(heads,indent=2))
        frame_status = execute_frames(root, source, frames, core, manifest['abi'], arena, descriptor,
                                      descriptors, timeout, results) if frames else None
        (results/'completion.json').write_text(json.dumps(dict(
            status='FPGA_GRAPH_PARITY_PASS', nodes=len(report),
            board_clock_mhz=clock_mhz,
            scope=source['scope'],real_weights_bound=source.get('real_weights_bound',False),
            frame_status=frame_status,test_evaluated=False),indent=2))
        print('GRAPH_COMPLETE', len(report), flush=True)
    except BaseException:
        QUARANTINE.append((overlay, owned))
        raise
    else:
        for buffer in reversed(owned):
            buffer.freebuffer()
