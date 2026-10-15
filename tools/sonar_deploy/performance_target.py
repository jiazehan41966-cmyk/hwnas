"""Resident native graph benchmark: one warmup, ten timed real inferences."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def run(root, adapter):
    root,adapter=Path(root),Path(adapter)
    rpc=module('native_rpc_helpers',adapter/'board_e2e.py')
    manifest=rpc.validate_manifest(rpc.read(adapter/'benchmark_manifest.json'))
    if rpc.sha_file(root/'board_manifest.json')!=manifest['board_manifest_sha256']:
        raise ValueError('Board manifest changed')
    if rpc.sha_file(adapter/'power.py')!=manifest['power_monitor']['source_sha256']:
        raise ValueError('Power source changed')
    board=rpc.read(root/'board_manifest.json')
    for name,digest in board['files'].items():
        if rpc.sha_file(root/name)!=digest:raise ValueError('Changed payload '+name)
    driver=module('native_graph_driver',root/'pynq_graph_runner.py')
    power=module('native_power',adapter/'power.py')
    output=root/'native_results';output.mkdir(exist_ok=False)
    global RETAINED
    session=driver.GraphSession(root,timeout=900);RETAINED=session
    monitor=power.IioPowerMonitor(output/'power',{
        **{k:manifest[k] for k in ('deployment_binding_sha256','package_manifest_sha256','board_manifest_sha256','graph_sha256')},
        'benchmark_manifest_sha256':rpc.sha_file(adapter/'benchmark_manifest.json')})
    rows=[]
    try:
        import numpy as np
        allocation=session.source['plan']['allocations']['image']
        c,h,w=allocation['shape'];offset=allocation['offset']
        image=session.arena[offset:offset+c*h*w].copy().reshape(h,w,c).transpose(2,0,1).copy()
        if rpc.sha_bytes(image.tobytes())!=manifest['input_sha256']:raise ValueError('Input differs')
        monitor.start()
        measured_start=None
        for index in range(11):
            if index==1:measured_start=time.monotonic()
            monitor.begin_inference(index)
            try:heads,timing=session.infer(image)
            finally:monitor.end_inference(index)
            end=time.monotonic()
            for head in manifest['heads']:
                if rpc.sha_bytes(heads[head['level']])!=head['sha256']:raise ValueError('Output mismatch')
            rows.append(dict(call_index=index,warmup=index==0,**timing))
            rpc.write(output/'progress.json',dict(completed_calls=len(rows),rows=rows,test_evaluated=False))
            print('NATIVE_CALL_COMPLETE',manifest['network'],index,flush=True)
        measured_duration=end-measured_start
        power_result=monitor.finish()
        session.close()
        result=dict(status='PASS',network=manifest['network'],seed=42,warmup=1,repetitions=10,
            batch_size=1,input_size=640,precision='INT8',weights_resident=True,rows=rows,
            graph_latency=rpc._summary([r['graph_seconds'] for r in rows[1:]]),
            input_graph_capture_latency=rpc._summary([r['input_graph_capture_seconds'] for r in rows[1:]]),
            measured_window_seconds=measured_duration,throughput_fps=10/measured_duration,
            throughput_scope='ten native calls wall window; includes inter-call witness check and record I/O',
            power_scope='board input over infer calls; host energy excluded',power_status=power_result['status'],
            clock_mhz=session.clock_mhz,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            benchmark_manifest_sha256=rpc.sha_file(adapter/'benchmark_manifest.json'),test_evaluated=False)
        rpc.write(output/'completion.json',result)
    except BaseException as exc:
        monitor.close();session.failed=True;driver.QUARANTINE.append((session.overlay,session.owned))
        rpc.write(output/'failure.json',dict(error=str(exc),status='QUARANTINED',test_evaluated=False))
        raise
