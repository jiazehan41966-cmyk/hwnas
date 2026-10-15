"""A short-lived, strictly read-only board identity/environment/lock receipt.

No credential/configuration is displayed, copied or persisted. The existing
credential-backed bridge executes one read-only notebook cell, then closes its
own kernel. No overlay, reset, governor write, worker, lock or job is created.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from . import board_transfer as transfer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hardware_template_coverage_20261013/toolchain_preflight"
BRIDGE = ROOT / "tools/hardware_template_coverage_v1/board_bridge_stdin.py"

# Deliberately avoid importing the mutating CpuGovernor or any Overlay object.
CODE = r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, re

started=datetime.now(timezone.utc).isoformat()
def text(path):
    try:
        return Path(path).read_text(encoding='ascii').strip()
    except (OSError,UnicodeError):
        return None
def safe_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError,UnicodeError):
        return None
def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
identity=[]
machine_path=Path('/etc/machine-id')
machine=text(machine_path)
if machine and re.fullmatch(r'[0-9a-fA-F]{32}',machine):
    identity.append(dict(board_id='machine-id:'+machine,evidence_path=str(machine_path),
        evidence_sha256=file_hash(machine_path),scope='board_OS_installation_not_silicon'))
for name in ('/sys/firmware/devicetree/base/serial-number','/proc/device-tree/serial-number'):
    try:
        p=Path(name)
        value=p.read_bytes().rstrip(b'\0\r\n').decode('ascii').strip()
        if value:
            identity.append(dict(board_id='serial:'+value,evidence_path=name,evidence_sha256=file_hash(p),scope='device_tree_serial'))
    except (OSError,UnicodeError):
        pass
clock=dict(source='pynq.Clocks read-only PS register-derived rates',independent_physical_frequency_calibration='NOT_RUN')
try:
    from pynq import Clocks
    for name in ('fclk0_mhz','fclk1_mhz','fclk2_mhz','fclk3_mhz','cpu_mhz'):
        try:
            value=float(getattr(Clocks,name))
            clock[name]=value if math.isfinite(value) else None
        except (AttributeError,OSError,ValueError,RuntimeError):
            clock[name]=None
except Exception as error:
    clock['read_status']=type(error).__name__
policies=[]
for policy in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
    row=dict(path=str(policy),actual=text(policy/'scaling_governor'),
        available=(text(policy/'scaling_available_governors') or '').split())
    row['performance_supported']='performance' in row['available']
    row['frequency_raw_khz']={name:text(policy/name) for name in ('scaling_cur_freq','scaling_min_freq','scaling_max_freq','cpuinfo_cur_freq','scaling_available_frequencies')}
    policies.append(row)
cpu=dict(policies=policies,any_policy_present=bool(policies),
    performance_supported_for_all=bool(policies) and all(p['performance_supported'] for p in policies),
    actual_performance_for_all=bool(policies) and all(p['actual']=='performance' for p in policies),
    changes_performed=False)
sensors=[]
paths=list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))+list(Path('/sys/class/hwmon').glob('hwmon*/temp*_input'))
for path in sorted(paths):
    raw=text(path)
    label_path=path.parent/('type' if path.name=='temp' else path.name.replace('_input','_label'))
    row=dict(path=str(path),raw_millidegrees_c=raw,label=text(label_path) or path.name)
    try:
        value=float(raw)/1000.0
        row.update(temperature_c=value if math.isfinite(value) else None,valid=math.isfinite(value) and -40<value<150)
    except (TypeError,ValueError):
        row.update(temperature_c=None,valid=False)
    sensors.append(row)
for device in sorted(Path('/sys/bus/iio/devices').glob('iio:device*')):
    for path in sorted(device.glob('in_temp*_raw')):
        prefix=path.name[:-4]
        scale_path=device/(prefix+'_scale')
        offset_path=device/(prefix+'_offset')
        if not scale_path.is_file():scale_path=device/'in_temp_scale'
        if not offset_path.is_file():offset_path=device/'in_temp_offset'
        raw,scale,offset=text(path),text(scale_path),text(offset_path)
        row=dict(path=str(path),raw=raw,scale=scale,offset=offset,label=text(device/(prefix+'_label')) or text(device/'name') or device.name,
            formula_c='(raw + offset) * scale / 1000',offset_default_zero=offset is None)
        try:
            value=(float(raw)+(float(offset) if offset is not None else 0))*float(scale)/1000.0
            row.update(temperature_c=value if math.isfinite(value) else None,valid=math.isfinite(value) and -40<value<150)
        except (TypeError,ValueError):
            row.update(temperature_c=None,valid=False)
        sensors.append(row)
    for path in sorted(device.glob('in_temp*_input')):
        raw=text(path)
        row=dict(path=str(path),raw_millidegrees_c=raw,label=text(device/'name') or device.name,source='IIO_PROCESSED')
        try:
            value=float(raw)/1000.0
            row.update(temperature_c=value if math.isfinite(value) else None,valid=math.isfinite(value) and -40<value<150)
        except (TypeError,ValueError):
            row.update(temperature_c=None,valid=False)
        sensors.append(row)
valid=[s['temperature_c'] for s in sensors if s['valid']]
pl=[s['temperature_c'] for s in sensors if s['valid'] and s.get('label','').lower()=='pl_temp']
temperature=dict(sensors=sensors,temperature_c=max(pl) if pl else max(valid) if valid else None,
    selection='pl_temp_label' if pl else 'maximum_valid_local_sensor')
def process_identity(pid):
    try:
        raw=(Path('/proc')/str(pid)/'stat').read_text()
        fields=raw[raw.rfind(')')+2:].split()
        return dict(pid=int(pid),state=fields[0],start_ticks=fields[19],comm=text(Path('/proc')/str(pid)/'comm'))
    except (OSError,ValueError,IndexError):
        return None
roots=[]
for root_name in ('/home/xilinx/hardware_measurement_pilot_20261012','/home/xilinx/hardware_template_coverage_20261013'):
    root=Path(root_name)
    lock=root/'board.lock'
    lock_value=safe_json(lock)
    lock_safe={k:lock_value.get(k) for k in ('job_id','created_utc','archive_sha256')} if isinstance(lock_value,dict) else None
    jobs=[]
    for job in sorted((root/'jobs').glob('*')):
        if not job.is_dir():continue
        status=safe_json(job/'status.json') or {}
        launch=safe_json(job/'launch.json') or {}
        row=dict(job_id=job.name,state=status.get('state'),mode=status.get('mode') or launch.get('mode'),
            board_lock_retained=status.get('board_lock_retained'),finished_utc=status.get('finished_utc'),
            worker_pid=launch.get('worker_pid'))
        row['worker_process']=process_identity(row['worker_pid']) if isinstance(row['worker_pid'],int) else None
        original=launch.get('worker_process') or {}
        row['recorded_process_identity_matches']=bool(row['worker_process']) and str(row['worker_process']['start_ticks'])==str(original.get('start_ticks'))
        row['status_sha256']=file_hash(job/'status.json')
        jobs.append(row)
    roots.append(dict(root=str(root),exists=root.exists(),board_lock_present=lock.is_file(),
        board_lock=lock_safe,board_lock_sha256=file_hash(lock),jobs=jobs))
active=[]
keywords=('board_runner.py','board_transfer.py','hardware_measurement_pilot_20261012','hardware_template_coverage_20261013')
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        argv=(path/'cmdline').read_bytes().decode('utf-8',errors='replace').split('\0')
        categories=[name for name in keywords if any(name in arg for arg in argv)]
        if categories:
            row=process_identity(int(path.name))
            if row:row['task_categories']=categories;active.append(row)
        # Raw argv values are neither returned nor persisted.
        argv=None
    except OSError:
        continue
result=dict(ok=True,schema='t6_board_readonly_preflight_v1',started_utc=started,
    finished_utc=datetime.now(timezone.utc).isoformat(),board_identity_options=identity,
    historical_machine_id_matches=machine=='bd081b0be1fd457093e07237b0dee576',
    machine_id_observed=machine,boot_id=text('/proc/sys/kernel/random/boot_id'),
    clock=clock,cpu=cpu,temperature=temperature,roots=roots,active_task_processes=active,
    preflight_kernel_pid=os.getpid(),hardware_actions='NONE',board_file_writes=False,
    credentials_or_helper_configuration_recorded=False,
    limits=['machine-id identifies OS installation, not immutable silicon',
        'Clock is a PS register-derived reading; no independent physical frequency calibration',
        'Process observations are a point-in-time read; no PL idle or quarantine-release claim',
        'No other Jupyter kernel namespace was inspected; this kernel is closed by its caller'])
print('PILOT_TRANSPORT_JSON='+json.dumps(result,allow_nan=False))
'''


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = OUT / ("board_readonly_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    result = transfer.bridge_call(folder, {"bridge": str(BRIDGE)}, CODE, "preflight")
    safe = dict(status="READ_ONLY_RECEIPT_RECORDED", receipt=str(folder / "receipt_preflight.json"),
                receipt_sha256=transfer.sha(folder / "receipt_preflight.json"),
                request_sha256=transfer.sha(folder / "request_preflight.py"),
                bridge_adapter_sha256=transfer.sha(BRIDGE), caller_sha256=transfer.sha(__file__),
                machine_id_observed=result.get("machine_id_observed"),
                historical_machine_id_matches=result.get("historical_machine_id_matches"),
                hardware_actions="NONE", credentials_or_helper_configuration_recorded=False)
    transfer.save(folder / "local_receipt_binding.json", safe, exclusive=True)
    print(json.dumps(safe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
