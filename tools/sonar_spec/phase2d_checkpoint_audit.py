"""Inspect existing witness checkpoints without evaluation or state mutation."""
import argparse
import json
from pathlib import Path
import torch
from formal_protocol import ROOT,sha,dump
from hwnas_fpga.models.sonar_supernet import architecture_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reports/phase2d_ranking_bias');args=p.parse_args()
    witness_path=ROOT/'reports/phase2c_supernet/ranking_witness_manifest.json'
    witnesses=json.loads(witness_path.read_text())['witnesses'];rows=[]
    for index,item in enumerate(witnesses):
        folder=ROOT/'results'/f'witness_{index:02d}';path=folder/'best.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['active_architecture']==item['architecture']
        assert architecture_id(state['active_architecture'])==item['architecture_id']
        assert state['binding']['witness_index']==index and state['binding']['witness_sha256']==sha(witness_path)
        metric=json.loads((folder/'best_valid_metrics.json').read_text())['mAP50_95']
        assert state['best_valid']==metric
        rows.append(dict(index=index,architecture_id=item['architecture_id'],checkpoint_sha256=sha(path),
            best_epoch=state['epoch'],best_valid_mAP50_95=metric,architecture_equal=True))
    dump(args.output/'witness_checkpoint_identity.json',dict(status='PASS',n=12,rows=rows))
    print('12/12 witness checkpoint architecture identities PASS')


if __name__=='__main__':main()
