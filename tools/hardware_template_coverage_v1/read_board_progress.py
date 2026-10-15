"""Read only the existing job's on-disk batch metadata; never access PL/DMA."""
import argparse
import json
from pathlib import Path
from . import board_transfer as transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-dir', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, config = transfer.config_for(args.transport_dir)
    job = transfer.valid_job(args.job_id)
    args.output.mkdir(parents=True, exist_ok=False)
    code = f'''import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
try:
 root=Path({config['board_root']!r})/'jobs'/{job!r}
 request=json.loads((root/'request.json').read_text())
 assert request['config']=={config!r}
 owner=request['config']['ownership_config']
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['expected_boot_id']
 assert hashlib.sha256(Path(owner['reservation']['path']).read_bytes()).hexdigest()==owner['reservation']['sha256']
 manifest=json.loads((root/'output/timing_manifest.json').read_text())
 result=dict(ok=True,observed_utc=datetime.now(timezone.utc).isoformat(),job_id={job!r},
  state=json.loads((root/'status.json').read_text())['state'],
  batches=[{{k:row.get(k) for k in ('batch_id','started_utc','finished_utc')}} for row in manifest['batches']],
  timing_csv_bytes=(root/'output/timing_raw.csv').stat().st_size,
  filesystem_read_only=True,hardware_access=False,not_acceptance_evidence=True)
except Exception as error:
 result=dict(ok=False,error=type(error).__name__+': '+str(error))
print({transfer.MARKER!r}+json.dumps(result))
'''
    result = transfer.bridge_call(args.output, config, code, 'progress')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
