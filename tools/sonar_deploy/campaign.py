"""Two fixed software-evaluation lanes; no training, sampling or test access."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.sonar_deploy.run import export_evaluate,RUN,write,utc

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',type=int,choices=[0,1],required=True)
    args=p.parse_args()
    runs=[(n,s) for n in ['D','C','B1'] for s in [42,43,44]][args.lane::2]
    for n,s in runs:
        try:
            result=export_evaluate(RUN,n,s,'cuda:0')
            print(n,s,result['status'],flush=True)
        except Exception as exc:
            write(RUN/n/f'seed_{s}'/'failure.json',dict(error=str(exc),time=utc(),test_evaluated=False))
            raise
    print('SOFTWARE_LANE_COMPLETE',args.lane,flush=True)

if __name__=='__main__':main()
