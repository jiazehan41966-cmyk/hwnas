"""Scoped, reversible service/IRQ controls for the authorized D09 CPU campaign.

Import is inert. Only apply/check/restore perform I/O. The fixed allowlists
exclude SSH, Jupyter, networking, time, D-Bus, udev, local timers/IPIs and PL IRQs.
No service is disabled, no process is killed, and no failed restoration is retried.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

CAMPAIGN_ROOT = Path('/home/xilinx/d09_cpu_freeze_20261013')
LOCK_PATH = Path('/home/xilinx/hardware_measurement_pilot_20261012/system_conditions.lock')
BOARD_LOCK_PATH = Path('/home/xilinx/hardware_measurement_pilot_20261012/board.lock')
STOP_UNITS = ('apt-daily.timer', 'apt-daily-upgrade.timer', 'cron.service',
              'unattended-upgrades.service', 'pynq-x11.service', 'nmbd.service', 'smbd.service')
IRQ_IDENTITIES = {'39':'eth0', '47':'ahci-ceva', '48':'mmc0', '49':'xuartps', '57':'xhci-hcd'}
IRQ_TARGET = b'0-2\n'
SERVICE_FIELDS = ('Id', 'LoadState', 'ActiveState', 'SubState', 'UnitFileState')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _save(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name+'.new')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, path)


class LinuxBackend:
    def identity(self):
        return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    machine_id=Path('/etc/machine-id').read_text().strip())

    def _run(self, argv):
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=10)
        if result.returncode:
            raise RuntimeError('READ_OR_CONTROL_COMMAND_FAILED:'+argv[0]+':'+str(result.returncode))
        return result.stdout

    def service(self, unit):
        if unit not in STOP_UNITS+('apt-daily.service','apt-daily-upgrade.service'):
            raise ValueError('SERVICE_NOT_ALLOWLISTED')
        output = self._run(['systemctl','show',unit,'--no-pager',
                           *[arg for name in SERVICE_FIELDS for arg in ('-p',name)]])
        result = dict(line.split('=',1) for line in output.splitlines() if '=' in line)
        if not all(name in result for name in SERVICE_FIELDS):
            raise ValueError('INCOMPLETE_SERVICE_STATE:'+unit)
        return {name:result[name] for name in SERVICE_FIELDS}

    def service_action(self, unit, action):
        if unit not in STOP_UNITS or action not in ('stop','start'):
            raise ValueError('SERVICE_ACTION_NOT_ALLOWLISTED')
        self._run(['systemctl',action,unit])

    def irq(self, irq):
        if irq not in IRQ_IDENTITIES:
            raise ValueError('IRQ_NOT_ALLOWLISTED')
        lines = Path('/proc/interrupts').read_text().splitlines()
        count = len(lines[0].split())
        matching = [line.split(':',1)[1].split() for line in lines[1:]
                    if ':' in line and line.split(':',1)[0].strip()==irq]
        if len(matching)!=1 or len(matching[0])<count:
            raise ValueError('IRQ_IDENTITY_MISSING:'+irq)
        description = ' '.join(matching[0][count:])
        if IRQ_IDENTITIES[irq] not in description:
            raise ValueError('IRQ_IDENTITY_CHANGED:'+irq)
        root = Path('/proc/irq')/irq
        return dict(description=description,
                    requested_b64=base64.b64encode((root/'smp_affinity_list').read_bytes()).decode(),
                    effective_b64=base64.b64encode((root/'effective_affinity_list').read_bytes()).decode())

    def irq_write(self, irq, value):
        if irq not in IRQ_IDENTITIES:
            raise ValueError('IRQ_NOT_ALLOWLISTED')
        with (Path('/proc/irq')/irq/'smp_affinity_list').open('wb') as stream:
            stream.write(value)

    def usage(self):
        smb = self._run(['smbstatus','--processes'])
        smb_count = sum(bool(line.split()) and line.split()[0].isdigit() for line in smb.splitlines())
        login = self._run(['loginctl','list-sessions','--no-legend','--no-pager'])
        gui = []
        for line in login.splitlines():
            parts=line.split()
            if not parts or not re.fullmatch(r'[A-Za-z0-9_-]+',parts[0]):continue
            typ=self._run(['loginctl','show-session',parts[0],'-p','Type','--value']).strip()
            if typ in ('x11','wayland'):gui.append(parts[0])
        apt=[]
        for path in Path('/proc').glob('[0-9]*/comm'):
            try:name=path.read_text().strip()
            except OSError:continue
            if name in ('apt','apt-get','aptitude','dpkg','dpkg-deb','dpkg-preconfigu'):
                apt.append(dict(pid=int(path.parent.name),name=name))
        apt_services={name:self.service(name)['ActiveState']
                      for name in ('apt-daily.service','apt-daily-upgrade.service')}
        return dict(smb_session_count=smb_count, gui_sessions=gui, apt_processes=apt,
                    apt_services=apt_services, argv_and_environ_read=False)


class SystemConditions:
    def __init__(self, backend=None, *, campaign_root=None, lock_path=None, board_lock_path=None):
        self.backend=backend or LinuxBackend()
        self.root=Path(campaign_root or CAMPAIGN_ROOT).resolve()
        self.lock=Path(lock_path or LOCK_PATH)
        self.board_lock=Path(board_lock_path or BOARD_LOCK_PATH)

    def _directory(self, evidence_dir):
        path=Path(evidence_dir).resolve()
        if not path.is_relative_to(self.root) or path==self.root:
            raise ValueError('EVIDENCE_DIRECTORY_OUTSIDE_CAMPAIGN')
        return path

    def _event(self, path, event, **fields):
        with (path/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(created_utc=_utc(),event=event,**fields),sort_keys=True)+'\n')
            stream.flush();os.fsync(stream.fileno())

    def _require_identity(self, config):
        current=self.backend.identity()
        if current['boot_id']!=config.get('expected_boot_id') or current['machine_id']!=config.get('expected_machine_id'):
            raise ValueError('BOOT_OR_MACHINE_ID_CHANGED')
        return current

    def _snapshot(self):
        return dict(services={unit:self.backend.service(unit) for unit in STOP_UNITS},
                    irqs={irq:self.backend.irq(irq) for irq in IRQ_IDENTITIES})

    def _initial_gate(self, config):
        identity=self._require_identity(config)
        if self.board_lock.exists():raise ValueError('MEASUREMENT_BOARD_LOCK_PRESENT')
        usage=self.backend.usage()
        if usage['smb_session_count'] or usage['gui_sessions'] or usage['apt_processes'] or any(
                value not in ('inactive','failed') for value in usage['apt_services'].values()):
            raise ValueError('SERVICE_CURRENTLY_IN_USE')
        snapshot=self._snapshot()
        for unit,row in snapshot['services'].items():
            if row['LoadState']!='loaded' or row['ActiveState'] not in ('active','inactive'):
                raise ValueError('SERVICE_NOT_STABLE:'+unit)
        for irq,row in snapshot['irqs'].items():
            if base64.b64decode(row['requested_b64']).strip()!=b'0-3' or base64.b64decode(row['effective_b64']).strip()!=b'0':
                raise ValueError('IRQ_BASELINE_CHANGED:'+irq)
        return identity,usage,snapshot

    def _owned(self,path):
        config=_read_json(path/'config.json');state=_read_json(path/'state.json')
        self._require_identity(config)
        if _sha(path/'config.json')!=state['config_sha256'] or _sha(path/'original.json')!=state['original_sha256']:
            raise ValueError('IMMUTABLE_CONDITION_EVIDENCE_CHANGED')
        if not self.lock.is_file() or _read_json(self.lock)!=state['lock_binding']:
            raise ValueError('CONDITION_LOCK_OWNERSHIP_CHANGED')
        return config,state,_read_json(path/'original.json')

    def apply(self, config, evidence_dir):
        if config.get('schema')!='d09_system_conditions_config_v1' or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',config.get('condition_id','')):
            raise ValueError('INVALID_CONDITION_CONFIG')
        path=self._directory(evidence_dir)
        if path.exists():raise ValueError('EVIDENCE_DIRECTORY_ALREADY_EXISTS')
        identity,usage,original=self._initial_gate(config)
        path.mkdir(parents=True,exist_ok=False)
        _save(path/'config.json',config,exclusive=True)
        _save(path/'original.json',original,exclusive=True)
        _save(path/'usage_before.json',usage,exclusive=True)
        binding=dict(condition_id=config['condition_id'],identity=identity,evidence_dir=str(path),
                     config_sha256=_sha(path/'config.json'),original_sha256=_sha(path/'original.json'),
                     owner_token=uuid.uuid4().hex)
        self.lock.parent.mkdir(parents=True,exist_ok=True)
        _save(self.lock,binding,exclusive=True)
        state=dict(status='PREPARED_NO_MUTATIONS',created_utc=_utc(),lock_binding=binding,
                   config_sha256=binding['config_sha256'],original_sha256=binding['original_sha256'],
                   attempted_operations=[],restore_attempted=False)
        _save(path/'state.json',state)
        try:
            # The baseline can change between inspection and lock acquisition.
            self._require_identity(config)
            if self.board_lock.exists() or self._snapshot()!=original:
                raise ValueError('BASELINE_CHANGED_BEFORE_APPLY')
            state['status']='APPLYING';_save(path/'state.json',state)
            for unit in STOP_UNITS:
                if original['services'][unit]['ActiveState']!='active':continue
                operation=dict(kind='service',id=unit)
                state['attempted_operations'].append(operation);_save(path/'state.json',state)
                self._event(path,'STOP_SERVICE_INTENT',unit=unit)
                self.backend.service_action(unit,'stop')
                now=self.backend.service(unit)
                if now['ActiveState']!='inactive' or now['UnitFileState']!=original['services'][unit]['UnitFileState']:
                    raise ValueError('SERVICE_STOP_DID_NOT_ESTABLISH_CONDITION:'+unit)
                self._event(path,'STOP_SERVICE_VERIFIED',unit=unit,state=now)
            for irq in IRQ_IDENTITIES:
                operation=dict(kind='irq',id=irq)
                state['attempted_operations'].append(operation);_save(path/'state.json',state)
                self._event(path,'IRQ_AFFINITY_INTENT',irq=irq,target_b64=base64.b64encode(IRQ_TARGET).decode())
                self.backend.irq_write(irq,IRQ_TARGET)
                now=self.backend.irq(irq)
                if base64.b64decode(now['requested_b64']).strip()!=b'0-2' or base64.b64decode(now['effective_b64']).strip()!=b'0':
                    raise ValueError('IRQ_AFFINITY_NOT_ESTABLISHED:'+irq)
                self._event(path,'IRQ_AFFINITY_VERIFIED',irq=irq,state=now)
            state['expected_applied']=self._snapshot()
            state['status']='APPLIED';state['applied_utc']=_utc();_save(path/'state.json',state)
            checked=self.check(path)
            return dict(ok=True,status='PASS_APPLIED',evidence_dir=str(path),check=checked,
                        original_sha256=state['original_sha256'],config_sha256=state['config_sha256'])
        except BaseException as error:
            self._event(path,'APPLY_FAILED',error_type=type(error).__name__,error=str(error))
            recovery=self.restore(path,allow_incomplete_apply=True)
            return dict(ok=True,status='FAIL_APPLY_RESTORED' if recovery['status']=='PASS_RESTORED' else 'FAIL_APPLY_RESTORE_FAILED',
                        error_type=type(error).__name__,error=str(error),recovery=recovery,evidence_dir=str(path))

    def check(self,evidence_dir):
        path=self._directory(evidence_dir);config,state,original=self._owned(path)
        if state['status']!='APPLIED':raise ValueError('CONDITIONS_NOT_APPLIED')
        current=self._snapshot()
        if current!=state['expected_applied']:
            self._event(path,'CHECK_FAILED',current=current)
            raise ValueError('SYSTEM_CONDITIONS_DRIFT')
        checked=dict(ok=True,status='PASS_UNCHANGED',created_utc=_utc(),condition_id=config['condition_id'],
                     config_sha256=state['config_sha256'],original_sha256=state['original_sha256'],
                     boot_id=config['expected_boot_id'],services_and_irqs=current)
        self._event(path,'CHECK_PASS')
        _save(path/'latest_check.json',checked)
        return checked

    def restore(self,evidence_dir,*,allow_incomplete_apply=False):
        path=self._directory(evidence_dir);config,state,original=self._owned(path)
        if state.get('restore_attempted'):raise ValueError('RESTORE_ALREADY_ATTEMPTED_NO_AUTOMATIC_RETRY')
        if self.board_lock.exists():raise ValueError('MEASUREMENT_BOARD_LOCK_PRESENT')
        if state['status']!='APPLIED' and not allow_incomplete_apply:
            raise ValueError('INCOMPLETE_APPLY_REQUIRES_EXPLICIT_REVIEW')
        state['restore_attempted']=True;state['status']='RESTORING';_save(path/'state.json',state)
        failures=[]
        for operation in reversed(state['attempted_operations']):
            kind,identifier=operation['kind'],operation['id']
            try:
                self._require_identity(config)
                if _read_json(self.lock)!=state['lock_binding']:raise ValueError('LOCK_OWNERSHIP_CHANGED')
                self._event(path,'RESTORE_INTENT',operation=operation)
                if kind=='irq':
                    now=self.backend.irq(identifier);old=original['irqs'][identifier]
                    wanted=base64.b64decode(old['requested_b64'])
                    actual=base64.b64decode(now['requested_b64'])
                    if actual!=wanted:
                        if actual.strip()!=IRQ_TARGET.strip():raise ValueError('IRQ_DRIFT_OUTSIDE_OWNED_CHANGE')
                        self.backend.irq_write(identifier,wanted)
                    if self.backend.irq(identifier)!=old:raise ValueError('IRQ_NOT_EXACTLY_RESTORED')
                else:
                    old=original['services'][identifier];now=self.backend.service(identifier)
                    if now!=old:
                        if now['ActiveState']!='inactive' or now['UnitFileState']!=old['UnitFileState']:
                            raise ValueError('SERVICE_DRIFT_OUTSIDE_OWNED_CHANGE')
                        self.backend.service_action(identifier,'start')
                    if self.backend.service(identifier)!=old:raise ValueError('SERVICE_NOT_EXACTLY_RESTORED')
                self._event(path,'RESTORE_VERIFIED',operation=operation)
            except BaseException as error:
                failures.append(dict(operation=operation,error_type=type(error).__name__,error=str(error)))
                self._event(path,'RESTORE_STEP_FAILED',failure=failures[-1])
                # Other owned changes still get one bounded restore attempt.
        try:
            current=self._snapshot()
            if current!=original:failures.append(dict(error='FINAL_STATE_DIFFERS_FROM_ORIGINAL'))
        except BaseException as error:
            current=None
            failures.append(dict(error='FINAL_STATE_UNREADABLE',error_type=type(error).__name__))
        if not self.lock.is_file() or _read_json(self.lock)!=state['lock_binding']:
            failures.append(dict(error='LOCK_OWNERSHIP_CHANGED_BEFORE_RELEASE'))
        result=dict(ok=True,status='FAIL_RESTORE' if failures else 'PASS_RESTORED',created_utc=_utc(),
                    condition_id=config['condition_id'],boot_id=config['expected_boot_id'],failures=failures,
                    original=original,restored=current,lock_retained=bool(failures),
                    enabled_state_changed=False,automatic_retry_permitted=False)
        state['status']=result['status'];state['finished_utc']=result['created_utc'];_save(path/'state.json',state)
        _save(path/'restore.json',result,exclusive=True)
        if not failures:
            if _read_json(self.lock)!=state['lock_binding']:raise ValueError('LOCK_OWNERSHIP_CHANGED_BEFORE_RELEASE')
            self.lock.unlink()
        return result


def apply(config,evidence_dir):
    return SystemConditions().apply(config,evidence_dir)


def check(evidence_dir):
    return SystemConditions().check(evidence_dir)


def restore(evidence_dir):
    return SystemConditions().restore(evidence_dir)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('apply','check','restore'))
    parser.add_argument('--evidence-dir',required=True)
    parser.add_argument('--config')
    args=parser.parse_args()
    answer=apply(_read_json(args.config),args.evidence_dir) if args.action=='apply' else globals()[args.action](args.evidence_dir)
    print('PILOT_TRANSPORT_JSON='+json.dumps(answer,ensure_ascii=True))
