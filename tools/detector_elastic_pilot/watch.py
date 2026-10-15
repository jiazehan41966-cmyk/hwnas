"""Bounded read-only watcher for this one E0/E1 pilot queue."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.detector_elastic_pilot import remote


def snapshot():
    code='''from pathlib import Path
import json
root=Path(REMOTE);result=root/RESULT
queue=json.loads((result/'queue_status.json').read_bytes())
out={'queue_status':queue['status'],'active':queue.get('active'),'pid':queue.get('pid'),'prototypes':{}}
for proto in ['E0','E1']:
 folder=result/'formal'/proto/'seed_42'
 p=folder/'epoch_log.json'
 if not p.is_file():
  out['prototypes'][proto]={'epochs':0,'status':'PENDING'}
  continue
 history=json.loads(p.read_bytes())
 best=max(history,key=lambda r:r['valid']['mAP50_95'])
 status=json.loads((folder/'status.json').read_bytes())
 out['prototypes'][proto]={'epochs':len(history),'status':status['status'],
  'best_epoch':best['epoch'],'best_mAP50_95':best['valid']['mAP50_95'],
  'best_AP50':best['valid']['AP50'],'best_Precision':best['valid']['Precision'],'best_Recall':best['valid']['Recall'],
  'last_mAP50_95':history[-1]['valid']['mAP50_95'],
  'last_epoch_seconds':history[-1]['wall_seconds'],'train_seconds':sum(r['wall_seconds'] for r in history)}
print(json.dumps(out))
'''
    return json.loads(remote.execute(remote.expand(code),40))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-checks',type=int,default=120)
    parser.add_argument('--interval',type=float,default=45)
    args=parser.parse_args()
    if not 1<=args.max_checks<=160 or not 1<=args.interval<=60:
        raise ValueError('Watcher must be bounded and responsive')
    path=ROOT/'results/detector_elastic_pilot_20261005/progress_snapshots.jsonl'
    errors=0
    for index in range(args.max_checks):
        try:
            value=snapshot();errors=0
            value['observed_at']=datetime.now(timezone.utc).isoformat()
            with path.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(value)+'\n')
            print(json.dumps(value),flush=True)
            if value['queue_status'] in ('COMPLETE','FAILED_NO_AUTORETRY'):
                return
        except Exception as error:
            errors+=1
            print(json.dumps({'status':'WATCH_ERROR','error_type':type(error).__name__,
                              'error':str(error),'consecutive_errors':errors}),flush=True)
            if errors>=3:
                raise
        if index+1<args.max_checks:
            time.sleep(args.interval)


if __name__=='__main__':
    main()
