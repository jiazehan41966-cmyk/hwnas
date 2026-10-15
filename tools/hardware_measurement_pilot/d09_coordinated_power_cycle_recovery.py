"""Archive three pinned D09/T6 locks and both jobs after a user-confirmed full power cycle.

Import is inert. No bridge, FPGA/PYNQ/MMIO, service mutation, reset, kill or free.
The caller must obtain a fresh new boot ID and explicit completed-power-cycle
confirmation. A different boot ID alone is not evidence of physical power loss.
"""
from __future__ import annotations
import base64
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import uuid

PILOT_ROOT=Path('/home/xilinx/hardware_measurement_pilot_20261012')
CAMPAIGN_ROOT=Path('/home/xilinx/d09_cpu_freeze_20261013')
T6_ROOT=Path('/home/xilinx/hardware_template_coverage_20261013')
T6_JOB_ID='t6_boundary_20261013T0406_timing'
T6_OLD_PROCESSES={5292:'751633',5293:'751700'}
T6_BOARD_LOCK_SHA='301572adefbded77f2501ab2c1bc6d444be3557f75ab063660ffbdf420f5e2d6'
T6_REQUEST_SHA='e8f3fb68c9ce9f3c7d4273331ed9a7a38cf526fee8698b32fb41ba01a9028660'
JOB_ID='q2_body_high_cal01'
OLD_BOOT='1d339ada-c002-4361-a65b-6bd7cd49552d'
MACHINE='bd081b0be1fd457093e07237b0dee576'
OLD_PROCESSES={5547:'768501',5548:'768566'}
PACKAGE_SHA='2510296772b9e5ce4d5cd7a043662614d2a0b9581dd0d8a1738ab67051d14993'
BOARD_LOCK_SHA='d581b9c272bad06a34f3becf496deae9ba221f121a77044bc753755e0aedd1b5'
SYSTEM_LOCK_SHA='69e1085770788ab2633fe14100cc6f0b0de64353306905ea7b7470aa5020db8e'
ORIGINAL_SHA='6bd2f34e17c04766495cf99b16d657d65ff7da4ddc3876a94eded4ae8efb9536'
CONFIG_SHA='2f614f8eba66a82684f4e5fc4e5395a32907948c2041ea446e7b9d8c2b90d4d3'
REQUEST_SHA='d041c7a1ae372e3639732f0d31395bedb52b89e5ecdbbe75ec881d7685a76e1e'
QUARANTINE_SHA='03c2f12db915a7340ce110d66d365dcc9dc1fc06ad0aede22b1e6cf893079730'
CPU_DEFERRED_SHA='b14d019a97c6d7fb8089b7ae70ba1bba37c224f242812aeb4a6c1fd6fb572d7e'
SYSTEM_OWNER='61f0552bcc494db59192b6b565458f4e'
UNITS=('apt-daily.timer','apt-daily-upgrade.timer','cron.service','unattended-upgrades.service','pynq-x11.service','nmbd.service','smbd.service')
IRQ_IDS=('39','47','48','49','57')
SERVICE_FIELDS=('Id','LoadState','ActiveState','SubState','UnitFileState')
MAX_FILES=1024
MAX_BYTES=256*1024*1024

def utc():return datetime.now(timezone.utc).isoformat()
def digest(data):return hashlib.sha256(data).hexdigest()
def decode(data):return json.loads(data.decode('utf-8'))
def exists_any(path):return os.path.lexists(path)
def file_identity(path):
    value=Path(path).lstat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)
def require(test,message):
    if not test:raise ValueError(message)
def save_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
def safe_path(path,root):
    path,root=Path(path),Path(root)
    require(path.is_absolute() and root.is_absolute(),'ABSOLUTE_PATH_REQUIRED')
    require(path==path.resolve() and root==root.resolve() and path.is_relative_to(root),'PATH_ESCAPE_OR_ALIAS')
    for item in (path,*path.parents):
        require(not item.is_symlink(),'SYMLINK_FORBIDDEN')
    return path
def read_regular(path,root):
    path=safe_path(path,root)
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1,'REGULAR_UNLINKED_FILE_REQUIRED')
    require(before.st_size<=MAX_BYTES,'FILE_SIZE_LIMIT')
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    try:
        opened=os.fstat(fd)
        require((opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino),'FILE_REPLACED_BEFORE_OPEN')
        with os.fdopen(fd,'rb',closefd=False) as stream:data=stream.read(MAX_BYTES+1)
        require(len(data)<=MAX_BYTES,'FILE_SIZE_LIMIT')
    finally:os.close(fd)
    after=path.lstat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'FILE_CHANGED_DURING_READ')
    return data

class LinuxBackend:
    def __init__(self):
        self.pilot_root=PILOT_ROOT;self.campaign_root=CAMPAIGN_ROOT;self.t6_root=T6_ROOT
    def identity(self):
        return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),machine_id=Path('/etc/machine-id').read_text().strip())
    def process_inventory(self):
        old_pids={};active=[]
        roots=(str(self.pilot_root),str(self.campaign_root),str(self.t6_root))
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():continue
            pid=int(folder.name)
            try:
                raw=(folder/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
                item=dict(pid=pid,state=parts[0],start_ticks=parts[19])
                argv=(folder/'cmdline').read_bytes().split(b'\0')
                again=(folder/'stat').read_text();current=again[again.rfind(')')+2:].split()
                require(current[19]==item['start_ticks'],'PROCESS_IDENTITY_CHANGED_DURING_SCAN')
            except (FileNotFoundError,ProcessLookupError):continue
            if pid in OLD_PROCESSES or pid in T6_OLD_PROCESSES:old_pids[str(pid)]=item
            if pid==os.getpid() or item['state']=='Z':continue
            # Inspect inline code only in RAM. No argv/environment is persisted.
            tokens=[x.decode('utf-8',errors='replace') for x in argv if x]
            scope=any(root+'/' in token for root in roots for token in tokens)
            role=any(any(mark in token for mark in ('board_runner.py','d09_cpu_runner.py','_worker','system_conditions')) for token in tokens)
            if scope and role:active.append(item)
        return dict(old_pid_observations=old_pids,active_campaign_processes=active,argv_persisted=False,
                    scope='KNOWN_D09_AND_T6_ROOTS_INCLUDING_INLINE_RUNPY_NOT_UNKNOWN_EXTERNAL_PROGRAMS')
    def conditions(self):
        services={}
        for unit in UNITS:
            result=subprocess.run(['systemctl','show',unit,'--no-pager',*[arg for name in SERVICE_FIELDS for arg in ('-p',name)]],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
            require(result.returncode==0,'SERVICE_READ_FAILED:'+unit)
            row=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
            require(all(k in row for k in SERVICE_FIELDS),'SERVICE_FIELDS_MISSING:'+unit)
            services[unit]={k:row[k] for k in SERVICE_FIELDS}
        lines=Path('/proc/interrupts').read_text().splitlines();count=len(lines[0].split());irqs={}
        for irq in IRQ_IDS:
            matches=[line.split(':',1)[1].split() for line in lines[1:] if ':' in line and line.split(':',1)[0].strip()==irq]
            require(len(matches)==1 and len(matches[0])>=count,'IRQ_IDENTITY_MISSING:'+irq)
            root=Path('/proc/irq')/irq
            irqs[irq]=dict(description=' '.join(matches[0][count:]),requested_b64=base64.b64encode((root/'smp_affinity_list').read_bytes()).decode(),effective_b64=base64.b64encode((root/'effective_affinity_list').read_bytes()).decode())
        return dict(services=services,irqs=irqs)
    def fsync_directory(self,path):
        fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def rename_noreplace(self,source,target):
        # Linux renameat2 is atomic and refuses any existing target. No unsafe fallback.
        libc=ctypes.CDLL(None,use_errno=True)
        require(hasattr(libc,'renameat2'),'RENAME_NOREPLACE_UNAVAILABLE')
        call=libc.renameat2;call.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint);call.restype=ctypes.c_int
        if call(-100,os.fsencode(source),-100,os.fsencode(target),1):
            code=ctypes.get_errno();raise OSError(code,'SCOPED_LOCK_ARCHIVE_FAILED')

def validate_confirmation(config):
    require(isinstance(config,dict) and config.get('schema')=='d09_coordinated_power_cycle_recovery_v1','INVALID_RECOVERY_CONFIG')
    require(config.get('complete_power_cycle') is True,'COMPLETED_FULL_POWER_CYCLE_CONFIRMATION_REQUIRED')
    require(config.get('confirmation_kind')=='USER_CONFIRMED_FULL_ZCU104_POWER_CYCLE_COMPLETED','EXPLICIT_PHYSICAL_POWER_CYCLE_REQUIRED')
    require(isinstance(config.get('confirmation_source'),str) and bool(config['confirmation_source'].strip()),'USER_CONFIRMATION_SOURCE_REQUIRED')
    confirmed=datetime.fromisoformat(config.get('confirmation_received_utc',''))
    require(confirmed.tzinfo is not None,'CONFIRMATION_TIMEZONE_REQUIRED')
    new=config.get('expected_new_boot_id')
    require(isinstance(new,str) and str(uuid.UUID(new))==new and new!=OLD_BOOT,'DISTINCT_CANONICAL_NEW_BOOT_REQUIRED')
    require(config.get('helper_sha256')==digest(Path(__file__).read_bytes()),'RECOVERY_HELPER_HASH_CHANGED')
    pins=config.get('post_boot_evidence_sha256')
    require(isinstance(pins,dict) and set(pins)=={'d09_status','t6_status','system_state'},'EXACT_DYNAMIC_EVIDENCE_PINS_REQUIRED')
    require(all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in pins.values()),'INVALID_EVIDENCE_HASH')
    return new

def difference(original,current):
    differences=[]
    for unit in UNITS:
        for key in ('Id','LoadState','ActiveState','UnitFileState'):
            if current.get('services',{}).get(unit,{}).get(key)!=original['services'][unit][key]:differences.append(dict(kind='service',id=unit,field=key,expected=original['services'][unit][key],actual=current.get('services',{}).get(unit,{}).get(key)))
    for irq in IRQ_IDS:
        for key in ('description','requested_b64','effective_b64'):
            if current.get('irqs',{}).get(irq,{}).get(key)!=original['irqs'][irq][key]:differences.append(dict(kind='irq',id=irq,field=key,expected=original['irqs'][irq][key],actual=current.get('irqs',{}).get(irq,{}).get(key)))
    return differences

def recover(config,backend=None):
    new_boot=validate_confirmation(config)
    backend=backend or LinuxBackend()
    pilot=safe_path(backend.pilot_root,backend.pilot_root)
    campaign=safe_path(backend.campaign_root,backend.campaign_root)
    t6=safe_path(backend.t6_root,backend.t6_root)
    job=safe_path(pilot/'jobs'/JOB_ID,pilot)
    t6_job=safe_path(t6/'jobs'/T6_JOB_ID,t6)
    system=safe_path(campaign/'system_conditions/cpu_freeze01',campaign)
    paths={'d09_system_conditions.lock':pilot/'system_conditions.lock',
           't6_board.lock':t6/'board.lock','d09_board.lock':pilot/'board.lock'}
    roots={'d09_system_conditions.lock':pilot,'t6_board.lock':t6,'d09_board.lock':pilot}
    expected={'d09_system_conditions.lock':SYSTEM_LOCK_SHA,'t6_board.lock':T6_BOARD_LOCK_SHA,'d09_board.lock':BOARD_LOCK_SHA}
    foreign_system=t6/'system_conditions.lock'
    def lifecycle():
        identity=backend.identity()
        require(identity==dict(boot_id=new_boot,machine_id=MACHINE),'NEW_BOOT_OR_MACHINE_ID_MISMATCH')
        processes=backend.process_inventory()
        require(not processes['active_campaign_processes'],'ACTIVE_D09_OR_T6_PROCESS_IN_NEW_BOOT')
        require(not exists_any(foreign_system),'UNEXPECTED_T6_SYSTEM_LOCK')
        return dict(identity=identity,processes=processes,old_process_identity=dict(boot_id=OLD_BOOT,d09=OLD_PROCESSES,t6=T6_OLD_PROCESSES),
                    old_activity_termination_evidence='Explicit full-power-cycle completion confirmation plus verified different boot; no new-boot MMIO access')
    initial=lifecycle()
    locks={name:read_regular(path,roots[name]) for name,path in paths.items()}
    lock_identities={name:file_identity(path) for name,path in paths.items()}
    for name in paths:require(digest(locks[name])==expected[name],'EXACT_OLD_LOCK_REQUIRED:'+name)
    pins=config['post_boot_evidence_sha256']
    def pinned(path,root,expected_hash,label):
        raw=read_regular(path,root);require(digest(raw)==expected_hash,'PINNED_EVIDENCE_CHANGED:'+label);return decode(raw)
    request=pinned(job/'request.json',job,REQUEST_SHA,'d09_request')
    t6_request=pinned(t6_job/'request.json',t6_job,T6_REQUEST_SHA,'t6_request')
    require(request['lock_binding']==decode(locks['d09_board.lock']) and t6_request['lock_binding']==decode(locks['t6_board.lock']),'JOB_REQUEST_LOCK_BINDING_MISMATCH')
    require(request['board_id']==t6_request['board_id']=='machine-id:'+MACHINE,'OLD_JOB_BOARD_IDENTITY_MISMATCH')
    binding=decode(locks['d09_system_conditions.lock'])
    require(binding['owner_token']==SYSTEM_OWNER and binding['condition_id']=='d09_cpu_freeze01' and binding['identity']==dict(boot_id=OLD_BOOT,machine_id=MACHINE),'SYSTEM_OWNER_OR_OLD_BOOT_MISMATCH')
    original=pinned(system/'original.json',system,ORIGINAL_SHA,'original')
    pinned(system/'config.json',system,CONFIG_SHA,'system_config')
    state=pinned(system/'state.json',system,pins['system_state'],'system_state')
    require(state['lock_binding']==binding and state['status']=='APPLIED' and state['restore_attempted'] is False and state['original_sha256']==ORIGINAL_SHA and state['config_sha256']==CONFIG_SHA,'SYSTEM_TRANSACTION_NOT_EXACT_APPLIED_OWNER')
    for label,directory,job_id,old_pids in [('d09',job,JOB_ID,OLD_PROCESSES),('t6',t6_job,T6_JOB_ID,T6_OLD_PROCESSES)]:
        old_status=pinned(directory/'status.json',directory,pins[label+'_status'],label+'_status')
        require(old_status['job_id']==job_id and old_status['state']=='DMA_QUARANTINED','EXACT_QUARANTINED_JOB_REQUIRED:'+label)
        for role,pid in zip(('worker','runner'),old_pids):
            require(old_status[role+'_pid']==pid and old_status[role+'_process']['start_ticks']==old_pids[pid],'OLD_PROCESS_IDENTITY_CHANGED:'+label)
        retention=pinned(directory/'output/device_buffer_retention.json',directory,QUARANTINE_SHA,label+'_retention')
        require(retention['buffers_freed'] is False and retention['reset_performed'] is False,'OLD_BUFFER_RETENTION_DIFFERS:'+label)
    pinned(job/'output/cpu_conditions_restore.json',job,CPU_DEFERRED_SHA,'d09_cpu_deferred')
    selected={};sources={};total=0
    trees=[('d09_job',job),('t6_job',t6_job),('d09_system_conditions',system)]
    for prefix,root in trees:
        for path in sorted(root.rglob('*')):
            safe_path(path,root)
            if path.is_dir():continue
            require(len(selected)<MAX_FILES,'EVIDENCE_FILE_COUNT_LIMIT')
            raw=read_regular(path,root);total+=len(raw);require(total<=MAX_BYTES,'TOTAL_EVIDENCE_SIZE_LIMIT')
            name=prefix+'/'+path.relative_to(root).as_posix();selected[name]=raw;sources[name]=(path,root,file_identity(path))
    selected.update({'old_locks/'+name:raw for name,raw in locks.items()})
    before=backend.conditions();deviations=difference(original,before)
    archive=safe_path(campaign/'recovery'/('coordinated_power_cycle_'+new_boot),campaign)
    archive.parent.mkdir(exist_ok=True);safe_path(archive.parent,campaign)
    archive.mkdir(exist_ok=False);safe_path(archive,campaign)
    manifest=dict(schema='d09_t6_coordinated_power_cycle_archive_v1',created_utc=utc(),config=config,lifecycle=initial,original=original,
        post_boot_conditions=before,condition_differences=deviations,files={name:digest(raw) for name,raw in selected.items()},
        original_roots_preserved=[str(root) for _,root in trees],lock_move_order=list(paths),lock_file_identities=lock_identities,
        automatic_retry_permitted=False,coordination_scope='ROOT_IS_SOLE_BOARD_WRITER_DURING_RECOVERY; NO CLAIM OF GLOBAL_MUTEX_FOR_UNKNOWN_EXTERNAL_PROGRAMS')
    moved=[]
    after=before;final=initial
    def exact_remaining_and_archived():
        for name,path in paths.items():
            if name in moved:
                require(not exists_any(path),'NEW_LOCK_APPEARED_DURING_ARCHIVE:'+name)
                copy=archive/(name+'.archived')
                require(file_identity(copy)==lock_identities[name] and read_regular(copy,archive)==locks[name],'ARCHIVED_LOCK_CHANGED:'+name)
            else:
                require(file_identity(path)==lock_identities[name] and read_regular(path,roots[name])==locks[name],'OLD_LOCK_CHANGED_BEFORE_ARCHIVE:'+name)
    try:
        backend.fsync_directory(campaign);backend.fsync_directory(archive.parent)
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
            for name,raw in sorted(selected.items()):
                info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;tar.addfile(info,io.BytesIO(raw))
        raw_archive=buffer.getvalue()
        with (archive/'original_evidence.tar.gz').open('xb') as stream:stream.write(raw_archive);stream.flush();os.fsync(stream.fileno())
        manifest['archive_sha256']=digest(raw_archive);manifest['archive_bytes']=len(raw_archive)
        save_new(archive/'00_prepared.json',manifest);backend.fsync_directory(archive)
        stored=read_regular(archive/'original_evidence.tar.gz',archive)
        require(digest(stored)==manifest['archive_sha256'],'ARCHIVE_VERIFICATION_FAILED')
        with tarfile.open(fileobj=io.BytesIO(stored),mode='r:gz') as tar:
            require(set(tar.getnames())==set(selected),'ARCHIVE_MEMBER_SET_CHANGED')
            for member in tar.getmembers():
                require(member.isfile() and tar.extractfile(member).read()==selected[member.name],'ARCHIVE_MEMBER_BYTES_CHANGED')
        if deviations:
            result=dict(ok=False,status='FAIL_POST_POWER_CYCLE_CONDITIONS_ALL_LOCKS_RETAINED',condition_differences=deviations,lock_paths_archived=[],archive_dir=str(archive),automatic_retry_permitted=False)
            save_new(archive/'result.json',result);backend.fsync_directory(archive);return result
        lifecycle()
        current_names={prefix+'/'+p.relative_to(root).as_posix() for prefix,root in trees for p in root.rglob('*') if not p.is_dir()}
        require(current_names==set(sources),'OLD_EVIDENCE_FILE_SET_CHANGED')
        for name,(path,root,identity) in sources.items():require(file_identity(path)==identity and read_regular(path,root)==selected[name],'OLD_EVIDENCE_CHANGED_BEFORE_ARCHIVE')
        for name,path in paths.items():
            final=lifecycle();exact_remaining_and_archived()
            # Preserve the D09 board lock as the last known-workflow barrier.
            if name=='d09_board.lock':
                after=backend.conditions();deviations=difference(original,after)
                require(not deviations,'CONDITION_DIFFERENCE_BEFORE_FINAL_LOCK_COMMIT')
                final=lifecycle();exact_remaining_and_archived()
            target=archive/(name+'.archived');safe_path(path,roots[name]);safe_path(target,archive)
            backend.rename_noreplace(path,target);moved.append(name)
            backend.fsync_directory(path.parent);backend.fsync_directory(archive)
            exact_remaining_and_archived()
            save_new(archive/(str(len(moved)).zfill(2)+'_lock_archived.json'),dict(name=name,target=str(target),sha256=digest(locks[name]),created_utc=utc()))
            backend.fsync_directory(archive)
        final=lifecycle();exact_remaining_and_archived()
        result=dict(ok=True,status='PASS_COORDINATED_RECOVERY_BY_AUTHORIZED_POWER_CYCLE',created_utc=utc(),archive_dir=str(archive),archive_sha256=manifest['archive_sha256'],
            lock_paths_archived=moved,condition_differences=deviations,original=original,restored=after,lifecycle=final,
            original_job_evidence_modified=False,system_guard_restore_called=False,device_operations='NONE_NO_MMIO_RESET_KILL_FREE_SIGNAL_OR_PROGRAM',service_or_irq_mutations=False,
            automatic_retry_permitted=False,formal_cost_eligible=False,board_ready_for_measurement=False,next_gate='A NEW SHARED BOARD OWNERSHIP GATE IS STILL REQUIRED BEFORE LAUNCH')
        save_new(archive/'result.json',result);backend.fsync_directory(archive);return result
    except BaseException as error:
        failure=dict(ok=False,status='FAIL_COMMITTED_ARCHIVE_RECEIPT_OR_POSTCHECK' if len(moved)==3 else ('FAIL_PARTIAL_ARCHIVE' if moved else 'FAIL_ARCHIVE_PREPARATION_ALL_LOCKS_RETAINED'),
            error_type=type(error).__name__,error=str(error),archive_dir=str(archive),lock_paths_archived=moved,source_lock_present={name:exists_any(path) for name,path in paths.items()},
            archived_targets_present={name:exists_any(archive/(name+'.archived')) for name in locks},condition_differences=deviations,automatic_retry_permitted=False,created_utc=utc())
        try:save_new(archive/'failure.json',failure);backend.fsync_directory(archive)
        except OSError:pass
        raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    answer=recover(json.loads(Path(args.config).read_text(encoding='utf-8')))
    print('PILOT_TRANSPORT_JSON='+json.dumps(answer,ensure_ascii=True))
    raise SystemExit(0 if answer['status']=='PASS_COORDINATED_RECOVERY_BY_AUTHORIZED_POWER_CYCLE' else 1)
