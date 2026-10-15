"""Read and verify an existing terminal-job snapshot after a client timeout.

Does not launch measurement, create a remote archive, or modify remote files.
The existing frozen transport verifies each downloaded chunk and tar member.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from . import board_transfer as transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-dir', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--snapshot-name')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder, config = transfer.config_for(args.transport_dir)
    job_id = transfer.valid_job(args.job_id)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = transfer.call(folder, config, 'status', job_id=job_id)
    if (state.get('state') not in ('PASS', 'FAIL') or state.get('board_lock_retained') is not False or
            state.get('worker_pid_current_process') or state.get('runner_pid_current_process') or state.get('lock')):
        raise ValueError('requires a terminal original job, absent worker/runner and absent task lock')
    code = rf'''import hashlib,json,re,tarfile
from pathlib import Path
try:
 job=Path({config['board_root']!r})/'jobs'/{job_id!r}
 expected={config!r}
 request=json.loads((job/'request.json').read_text())
 assert request['config']==expected
 status=json.loads((job/'status.json').read_text())
 assert status['state'] in ('PASS','FAIL') and status['board_lock_retained'] is False
 candidates=[p for p in job.glob('snapshot_*.tar.gz') if re.fullmatch(r'snapshot_[0-9]+\.tar\.gz',p.name)]
 selected={args.snapshot_name!r}
 if selected is not None:
  assert re.fullmatch(r'snapshot_[0-9]+',selected)
  candidates=[p for p in candidates if p.name==selected+'.tar.gz']
 assert len(candidates)==1,'require exactly one existing archive or specify its exact name'
 archive=candidates[0]
 assert not any(p.is_symlink() for p in (archive,*archive.parents))
 before=archive.stat()
 with tarfile.open(archive,'r:gz') as tar:
  manifest=json.load(tar.extractfile('snapshot_manifest.json'))
 assert manifest['schema']=='pilot_board_snapshot_v1' and manifest['job_id']=={job_id!r}
 assert manifest['observed_state']==status['state']
 for name in ('request.json','status.json','output/summary.json'):
  assert hashlib.sha256((job/name).read_bytes()).hexdigest()==manifest['files'][name]
 digest=hashlib.sha256()
 with archive.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
 after=archive.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'archive still changing'
 result=dict(ok=True,snapshot_name=archive.name[:-7],archive_sha256=digest.hexdigest(),archive_bytes=after.st_size,
             observed_state=manifest['observed_state'],existing_archive_only=True,remote_writes=False)
except Exception as error:
 result=dict(ok=False,error=type(error).__name__+': '+str(error))
print({transfer.MARKER!r}+json.dumps(result))
'''
    metadata = transfer.bridge_call(output, config, code, 'existing_archive')
    transfer.save(output / 'download.json', metadata, exclusive=True)
    archive = output / 'snapshot.tar.gz'
    offset = 0
    with archive.open('xb') as stream:
        while offset < metadata['archive_bytes']:
            receipt = transfer.call(folder, config, 'fetch_chunk', job_id=job_id,
                                    snapshot_name=metadata['snapshot_name'], offset=offset)
            data = base64.b64decode(receipt['data'], validate=True)
            if (receipt['offset'] != offset or receipt['bytes'] != len(data) or
                    hashlib.sha256(data).hexdigest() != receipt['digest'] or not data):
                raise ValueError('existing archive chunk differs')
            stream.write(data)
            offset += len(data)
            print(json.dumps(dict(stage='existing_snapshot_download', bytes=offset, total=metadata['archive_bytes'])), flush=True)
    if archive.stat().st_size != metadata['archive_bytes'] or transfer.sha(archive) != metadata['archive_sha256']:
        raise ValueError('existing archive SHA/size differs')
    manifest = transfer.unpack(archive, output / 'evidence', manifest_name='snapshot_manifest.json')
    result = dict(status='VERIFIED_EXISTING_TERMINAL_SNAPSHOT', job_id=job_id,
                  files=len(manifest['files']), observed_state=manifest['observed_state'],
                  fetcher_sha256=transfer.sha(__file__), remote_writes=False, measurement_restarted=False)
    transfer.save(output / 'recovery.json', result, exclusive=True)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
