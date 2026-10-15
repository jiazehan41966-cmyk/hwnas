"""Archive only the two pinned D09 locks after a user-confirmed full power cycle.

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
MAX_FILES=512
MAX_BYTES=128*1024*1024

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
        self.pilot_root=PILOT_ROOT;self.campaign_root=CAMPAIGN_ROOT
    def identity(self):
        return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),machine_id=Path('/etc/machine-id').read_text().strip())
    def process_inventory(self):
        old_pids={};active=[]
        for folder in Path('/proc').iterdir():
            if not folder.name.isdigit():continue
            pid=int(folder.name)
            try:
                raw=(folder/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
                item=dict(pid=pid,state=parts[0],start_ticks=parts[19])
                argv=(folder/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,ProcessLookupError):continue
            # Permission errors must not be interpreted as an absent worker.
            if pid in OLD_PROCESSES:old_pids[str(pid)]=item
            if pid==os.getpid():continue
            tokens=[x.decode('utf-8',errors='replace') for x in argv if x]
            scope=any(x.startswith(str(self.pilot_root)+'/') or x.startswith(str(self.campaign_root)+'/') for x in tokens)
            worker=('_worker' in tokens or any(Path(x).name in ('board_runner.py','d09_cpu_runner.py') for x in tokens))
            if scope and worker:active.append(item)
        return dict(old_pid_observations=old_pids,active_campaign_processes=active,argv_persisted=False)
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
    require(isinstance(config,dict) and config.get('schema')=='d09_power_cycle_recovery_v1','INVALID_RECOVERY_CONFIG')
    require(config.get('complete_power_cycle') is True,'COMPLETED_FULL_POWER_CYCLE_CONFIRMATION_REQUIRED')
    require(config.get('confirmation_kind')=='USER_CONFIRMED_FULL_ZCU104_POWER_CYCLE_COMPLETED','EXPLICIT_PHYSICAL_POWER_CYCLE_REQUIRED')
    require(isinstance(config.get('confirmation_source'),str) and bool(config['confirmation_source'].strip()),'USER_CONFIRMATION_SOURCE_REQUIRED')
    confirmed=datetime.fromisoformat(config.get('confirmation_received_utc',''))
    require(confirmed.tzinfo is not None,'CONFIRMATION_TIMEZONE_REQUIRED')
    new=config.get('expected_new_boot_id')
    require(isinstance(new,str) and str(uuid.UUID(new))==new and new!=OLD_BOOT,'DISTINCT_CANONICAL_NEW_BOOT_REQUIRED')
    require(config.get('helper_sha256')==digest(Path(__file__).read_bytes()),'RECOVERY_HELPER_HASH_CHANGED')
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
    pilot=safe_path(backend.pilot_root,backend.pilot_root);campaign=safe_path(backend.campaign_root,backend.campaign_root)
    job=safe_path(pilot/'jobs'/JOB_ID,pilot);system=safe_path(campaign/'system_conditions/cpu_freeze01',campaign)
    board_lock=pilot/'board.lock';system_lock=pilot/'system_conditions.lock'
    def lifecycle():
        identity=backend.identity()
        require(identity==dict(boot_id=new_boot,machine_id=MACHINE),'NEW_BOOT_OR_MACHINE_ID_MISMATCH')
        processes=backend.process_inventory()
        require(not processes['active_campaign_processes'],'ACTIVE_CAMPAIGN_PROCESS_IN_NEW_BOOT')
        return dict(identity=identity,processes=processes,old_process_identity=dict(boot_id=OLD_BOOT,pid_start_ticks=OLD_PROCESSES),old_activity_termination_evidence='Explicit full-power-cycle completion confirmation plus verified different boot; no new-boot MMIO access')
    initial=lifecycle()
    locks={'board.lock':read_regular(board_lock,pilot),'system_conditions.lock':read_regular(system_lock,pilot)}
    lock_identities={'board.lock':file_identity(board_lock),'system_conditions.lock':file_identity(system_lock)}
    require(digest(locks['board.lock'])==BOARD_LOCK_SHA,'EXACT_OLD_BOARD_LOCK_REQUIRED')
    require(digest(locks['system_conditions.lock'])==SYSTEM_LOCK_SHA,'EXACT_OLD_SYSTEM_LOCK_REQUIRED')
    request=read_regular(job/'request.json',job);require(digest(request)==REQUEST_SHA,'EXACT_OLD_JOB_REQUEST_REQUIRED')
    binding=decode(locks['system_conditions.lock'])
    require(binding['owner_token']==SYSTEM_OWNER and binding['condition_id']=='d09_cpu_freeze01' and binding['identity']==dict(boot_id=OLD_BOOT,machine_id=MACHINE),'SYSTEM_OWNER_OR_OLD_BOOT_MISMATCH')
    original_bytes=read_regular(system/'original.json',system);config_bytes=read_regular(system/'config.json',system)
    require(digest(original_bytes)==ORIGINAL_SHA and digest(config_bytes)==CONFIG_SHA,'ORIGINAL_OR_CONFIG_EVIDENCE_CHANGED')
    original=decode(original_bytes);state=decode(read_regular(system/'state.json',system));old_status=decode(read_regular(job/'status.json',job))
    require(state['lock_binding']==binding and state['status']=='APPLIED' and state['restore_attempted'] is False and state['original_sha256']==ORIGINAL_SHA and state['config_sha256']==CONFIG_SHA,'SYSTEM_TRANSACTION_NOT_EXACT_APPLIED_OWNER')
    require(old_status['job_id']==JOB_ID and old_status['state']=='DMA_QUARANTINED','EXACT_QUARANTINED_JOB_REQUIRED')
    for key,pid in [('worker',5547),('runner',5548)]:
        require(old_status[key+'_pid']==pid and old_status[key+'_process']['start_ticks']==OLD_PROCESSES[pid],'OLD_PROCESS_IDENTITY_CHANGED')
    require(digest(read_regular(job/'output/device_buffer_retention.json',job))==QUARANTINE_SHA,'QUARANTINE_EVIDENCE_CHANGED')
    require(digest(read_regular(job/'output/cpu_conditions_restore.json',job))==CPU_DEFERRED_SHA,'CPU_DEFERRED_EVIDENCE_CHANGED')
    selected={};sources={};total=0
    for prefix,root in [('old_job',job),('system_conditions',system)]:
        for path in sorted(root.rglob('*')):
            safe_path(path,root)
            if path.is_dir():continue
            require(len(selected)<MAX_FILES,'EVIDENCE_FILE_COUNT_LIMIT')
            raw=read_regular(path,root);total+=len(raw);require(total<=MAX_BYTES,'TOTAL_EVIDENCE_SIZE_LIMIT')
            name=prefix+'/'+path.relative_to(root).as_posix();selected[name]=raw;sources[name]=(path,root,file_identity(path))
    selected.update({'old_locks/'+name:raw for name,raw in locks.items()})
    before=backend.conditions();deviations=difference(original,before)
    archive=safe_path(campaign/'recovery'/('power_cycle_'+new_boot),campaign)
    archive.parent.mkdir(exist_ok=True);safe_path(archive.parent,campaign)
    archive.mkdir(exist_ok=False);safe_path(archive,campaign)
    manifest=dict(schema='d09_power_cycle_archive_v1',created_utc=utc(),config=config,lifecycle=initial,original=original,post_boot_conditions=before,condition_differences=deviations,
                  files={name:digest(raw) for name,raw in selected.items()},original_roots_preserved=[str(job),str(system)],lock_move_order=['system_conditions.lock','board.lock'],
                  lock_file_identities=lock_identities,pilot_root_entries=sorted(p.name for p in pilot.iterdir()),automatic_retry_permitted=False)
    moved=[]
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
        require(digest(read_regular(archive/'original_evidence.tar.gz',archive))==manifest['archive_sha256'],'ARCHIVE_VERIFICATION_FAILED')
        if deviations:
            result=dict(ok=False,status='FAIL_POST_POWER_CYCLE_CONDITIONS_LOCKS_RETAINED',condition_differences=deviations,lock_paths_archived=[],archive_dir=str(archive),automatic_retry_permitted=False)
            save_new(archive/'result.json',result);backend.fsync_directory(archive);return result
        lifecycle()
        for name,(path,root,identity) in sources.items():require(file_identity(path)==identity and read_regular(path,root)==selected[name],'OLD_EVIDENCE_CHANGED_BEFORE_ARCHIVE')
        for name,path in [('system_conditions.lock',system_lock),('board.lock',board_lock)]:
            final=lifecycle()
            # Both old locks must remain exact until the first move; board barrier stays last.
            require(file_identity(board_lock)==lock_identities['board.lock'] and read_regular(board_lock,pilot)==locks['board.lock'],'BOARD_LOCK_CHANGED_BEFORE_ARCHIVE')
            if not moved:require(file_identity(system_lock)==lock_identities['system_conditions.lock'] and read_regular(system_lock,pilot)==locks['system_conditions.lock'],'SYSTEM_LOCK_CHANGED_BEFORE_ARCHIVE')
            if name=='board.lock':
                after=backend.conditions();deviations=difference(original,after)
                require(not deviations,'CONDITION_DIFFERENCE_BEFORE_BOARD_LOCK_COMMIT')
                final=lifecycle()
                require(file_identity(board_lock)==lock_identities['board.lock'] and read_regular(board_lock,pilot)==locks['board.lock'],'BOARD_LOCK_CHANGED_AT_FINAL_COMMIT')
                require(not exists_any(system_lock),'NEW_SYSTEM_LOCK_BEFORE_BOARD_COMMIT')
                archived_system=archive/'system_conditions.lock.archived'
                require(file_identity(archived_system)==lock_identities['system_conditions.lock'] and read_regular(archived_system,archive)==locks['system_conditions.lock'],'SYSTEM_ARCHIVE_CHANGED_BEFORE_BOARD_COMMIT')
            target=archive/(name+'.archived');safe_path(path,pilot);safe_path(target,archive)
            backend.rename_noreplace(path,target);moved.append(name)
            # Register the rename before any operation that may fail after it.
            backend.fsync_directory(path.parent);backend.fsync_directory(archive)
            require(file_identity(target)==lock_identities[name] and read_regular(target,archive)==locks[name],'ARCHIVED_LOCK_IDENTITY_OR_BYTES_CHANGED')
            save_new(archive/(str(len(moved)).zfill(2)+'_lock_archived.json'),dict(name=name,target=str(target),sha256=digest(locks[name]),created_utc=utc()))
            backend.fsync_directory(archive)
        require(not exists_any(board_lock) and not exists_any(system_lock),'NEW_LOCK_APPEARED_AFTER_ARCHIVE')
        result=dict(ok=True,status='PASS_RESTORED_BY_AUTHORIZED_POWER_CYCLE',created_utc=utc(),archive_dir=str(archive),archive_sha256=manifest['archive_sha256'],lock_paths_archived=moved,condition_differences=deviations,original=original,restored=after,lifecycle=final,
                    original_job_evidence_modified=False,system_guard_restore_called=False,device_operations='NONE_NO_MMIO_RESET_KILL_FREE_OR_PROGRAM',service_or_irq_mutations=False,automatic_retry_permitted=False,formal_cost_eligible=False)
        save_new(archive/'result.json',result);backend.fsync_directory(archive);return result
    except BaseException as error:
        failure=dict(ok=False,status='FAIL_COMMITTED_ARCHIVE_RECEIPT_OR_POSTCHECK' if len(moved)==2 else ('FAIL_PARTIAL_ARCHIVE' if moved else 'FAIL_ARCHIVE_PREPARATION_LOCKS_RETAINED'),error_type=type(error).__name__,error=str(error),archive_dir=str(archive),lock_paths_archived=moved,board_lock_present=exists_any(board_lock),system_lock_present=exists_any(system_lock),archived_targets_present={name:exists_any(archive/(name+'.archived')) for name in locks},condition_differences=deviations,automatic_retry_permitted=False,created_utc=utc())
        try:save_new(archive/'failure.json',failure);backend.fsync_directory(archive)
        except OSError:pass
        raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    answer=recover(json.loads(Path(args.config).read_text(encoding='utf-8')))
    print('PILOT_TRANSPORT_JSON='+json.dumps(answer,ensure_ascii=True))
    raise SystemExit(0 if answer['status']=='PASS_RESTORED_BY_AUTHORIZED_POWER_CYCLE' else 1)
