"""Recover a disconnected, already claimed board transaction without reprogramming.

Waits for the original cell's immutable return, then applies the same full-frame
validator and normal E2E/ledger admission. Never repeats a failed FPGA run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from tools.sonar_deploy import board_campaign as c
from tools.sonar_deploy import admission as a


def extract_return(archive, destination, max_bytes):
    destination = Path(destination).resolve()
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or sum(m.size for m in members) > max_bytes:
            raise ValueError('Duplicate or oversized return')
        if not {'completion.json', 'frames.json', 'layers.json', 'benchmark.json'} <= set(names):
            raise ValueError('Original board transaction not complete')
        for m in members:
            target = (destination/m.name).resolve()
            if not m.isfile() or not target.is_relative_to(destination):
                raise ValueError('Unsafe returned artifact')
            if target.exists():
                raise ValueError('Never overwrite returned evidence')
        for m in members:
            target = destination/m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tf.extractfile(m).read())


def wait_return(manifest, row, output, timeout):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            value = c.remote_json(c.remote_prefix(manifest['remote_project'])+f'''
r=Path({row['build']!r});owner=json.loads((r/'http_recovery.json').read_text())
proc=Path('/proc')/str(owner['pid']);args=(proc/'cmdline').read_bytes().split(bytes([0])) if proc.exists() else []
owned=str(r).encode() in args and any(v.endswith(b'http_transfer.py') for v in args)
archive=r/'board_return.tar.gz'
print(json.dumps({{'return_exists':archive.is_file(),'bytes':archive.stat().st_size if archive.exists() else 0,
 'sha256':a.sha256(archive) if archive.exists() else None,'transport_alive':owned,
 'owner':dict(pid=owner['pid'],directory=str(r),start_ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19]) if owned else None}}))
'''.replace("r=Path", "from tools.sonar_deploy import admission as a\nr=Path", 1), timeout=25)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            value = dict(status='WAITING_CONNECTION', error=type(exc).__name__)
        a.atomic_json(output/'RECOVERY_STATUS.json', dict(**value, test_evaluated=False))
        if value.get('return_exists'):
            return value
        if value.get('transport_alive') is False:
            raise RuntimeError('Recovery receiver stopped; retain original board session')
        time.sleep(min(30, max(.01, deadline-time.monotonic())))
    raise TimeoutError('Bounded recovery wait expired; no rerun or buffer cleanup')


def verify_idle_receipt(folder, binding, evidence):
    """Read-only MMIO check after the original batch; never free unknown buffers."""
    expected=a.sha256(evidence/'completion.json')
    code=f'''import hashlib,json,time
from pathlib import Path
from pynq import Overlay,MMIO,Clocks
r=Path({binding['board_directory']!r})
assert hashlib.sha256((r/'results/completion.json').read_bytes()).hexdigest()=={expected!r}
overlay=Overlay(str(r/'h2_graph.bit'),download=False)
info=overlay.ip_dict['core'];core=MMIO(info['phys_addr'],info['addr_range'])
states=[core.read(0)];time.sleep(.1);states.append(core.read(0))
assert all(v&4 for v in states),'Core not idle; retain original session and buffers'
print('RECOVERY_IDLE_RECEIPT',json.dumps(dict(status='PASS',control_states=states,
 completion_sha256={expected!r},actual_clock_mhz=float(Clocks.fclk0_mhz),
 reprogrammed=False,buffers_freed=False,test_evaluated=False)),flush=True)
'''
    script=folder/'recovery_idle_readonly.py';script.write_text(code,encoding='utf-8')
    log=evidence/'recovery_idle_readonly.log'
    c._local_command([sys.executable,str(c.BRIDGE),str(script),str(log),'--keep-kernel'],folder/'recovery_idle_bridge.log')
    rows=[line.split('RECOVERY_IDLE_RECEIPT ',1)[1] for line in log.read_text().splitlines() if line.startswith('RECOVERY_IDLE_RECEIPT ')]
    if len(rows)!=1 or json.loads(rows[0]).get('status')!='PASS':
        raise ValueError('Missing board idle receipt; do not start next session')
    a.atomic_json(evidence/'recovery_idle_receipt.json',json.loads(rows[0]))


def recover(output, network, timeout, continue_queue):
    output = Path(output).resolve()
    manifest = a.read_json(output/'CAMPAIGN_MANIFEST.json')
    state = c.snapshot(manifest['remote_run'], manifest['remote_project'])
    row = c._verify_snapshot(manifest, state, network)
    folder = output/network/'graph_build'
    attempt = a.read_json(output/network/'ATTEMPT.json')
    if not row['attempt_exists'] or row['board_complete']:
        raise ValueError('Recovery requires an existing incomplete board attempt')
    if any(attempt[k] != row[k] for k in ('graph_sha256', 'package_manifest_sha256')):
        raise ValueError('Original attempt identity changed')
    binding = a.read_json(folder/'board_execution_binding.json')
    if binding['remote_directory'] != row['build']:
        raise ValueError('Original execution directory changed')
    marker = output/network/'RECOVERY.json'
    with marker.open('x', encoding='utf-8') as handle:
        json.dump(dict(started=time.time(),action='collect_original_transaction',
            original_status=a.read_json(output/network/'STATUS.json'),source_sha256=a.sha256(__file__),
            test_evaluated=False),handle,indent=2)
    result = wait_return(manifest, row, output/network, timeout)
    evidence = folder/'board_evidence'
    tx = c.transport()
    archive = evidence/'board_return.tar.gz'
    c._scp([tx.HOST+':'+row['build']+'/board_return.tar.gz'], archive)
    if a.sha256(archive) != result['sha256']:
        raise ValueError('Returned archive hash mismatch')
    extract_return(archive,evidence,binding['return_limit_bytes'])
    names=['timing.rpt','utilization_hier.rpt','route_status.rpt','rtl_equivalence.json','hls.log',
           'board_payload/board_manifest.json','board_payload/source_manifest.json',
           'board_payload/h2_graph.bit','board_payload/h2_graph.hwh','board_payload/graph.json',
           'board_payload/export_manifest.json','utilization.rpt']
    c._scp([tx.HOST+':'+row['build']+'/'+name for name in names],evidence)
    from tools.sonar_deploy.board_result import validate_actual_frames
    validate_actual_frames(evidence,a.read_json(folder/'graph.json'),a.read_json(folder/'frames.json'),a.read_json(folder/'manifest.json'))
    # Disconnected stdout cannot certify buffer release. Independently confirm
    # the core is idle, retain original kernel/allocations, and never force free.
    verify_idle_receipt(folder,binding,evidence)
    c.upload_evidence(manifest,row,evidence)
    if result.get('owner'):
        c._stop_transfer(manifest['remote_project'],result['owner'])
    a.atomic_json(output/network/'RECOVERY_STATUS.json',dict(status='ORIGINAL_FRAME_RETURN_VERIFIED_E2E_PENDING',archive_sha256=result['sha256'],test_evaluated=False))
    c.e2e_transaction(manifest,row,folder)
    c._remote_command(manifest['remote_project'],[tx.PYTHON,'tools/sonar_deploy/run.py','board',
        '--root',manifest['remote_run'],'--network',network,'--seed','42'],timeout=1200)
    final=c._verify_snapshot(manifest,c.snapshot(manifest['remote_run'],manifest['remote_project']),network)
    if not final['board_complete']:
        raise RuntimeError('Board evidence did not pass final admission')
    a.atomic_json(output/network/'STATUS.json',dict(status='COMPLETE_VERIFIED',recovered_original_transaction=True,test_evaluated=False))
    a.atomic_json(output/network/'RECOVERY_STATUS.json',dict(status='COMPLETE_VERIFIED',test_evaluated=False))
    if continue_queue:
        for next_network in manifest['networks'][manifest['networks'].index(network)+1:]:
            result=c.run_network(output,manifest,next_network)
            print(json.dumps(result),flush=True)
            if result['status'] not in ('COMPLETE_VERIFIED','ALREADY_VERIFIED','STOP_ACCURACY_BELOW_D'):
                raise RuntimeError('Downstream campaign stopped: '+str(result))
    c._remote_command(manifest['remote_project'],[tx.PYTHON,'tools/sonar_deploy/report.py','--root',manifest['remote_run']],timeout=120)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--network',choices=c.NETWORKS,default='D')
    parser.add_argument('--timeout',type=float,default=43200)
    parser.add_argument('--continue-queue',action='store_true')
    args=parser.parse_args()
    if not 0<args.timeout<=43200:
        parser.error('Recovery bounded to at most twelve hours')
    from hwnas_fpga.metrics_ledger import _ledger_lock
    with _ledger_lock(ROOT/'results/sonar_real_int8_deployment_v1/zcu104_board_campaign.lock',timeout=1):
        try:
            recover(args.output,args.network,args.timeout,args.continue_queue)
        except BaseException as exc:
            a.atomic_json(args.output/args.network/'RECOVERY_STATUS.json',dict(status='FAILED_RETAINED_NO_AUTORETRY',error=type(exc).__name__+': '+str(exc),test_evaluated=False))
            raise


if __name__=='__main__':main()
