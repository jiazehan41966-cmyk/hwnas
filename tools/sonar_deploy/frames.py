"""Bind fixed train8 traces and full valid99 integer heads to a board frame set."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'tools/sonar_analysis'),str(ROOT/'scripts')]
from tools.sonar_deploy import admission as gate


def prepare_frames(root,network,device='cuda'):
    import torch
    import baseline as frozen
    from phase5a_quantization.common import baseline
    from tools.sonar_deploy.ptq_export import IntegerGraphModel
    root=Path(root).resolve();binding=gate.load_binding(root)
    expected=next(r for r in binding['runs'] if r['network']==network and r['seed']==42)
    result=gate.load_result(root,expected)
    quality=gate.accuracy_admission(root,binding)
    if not quality['allowed'] or not quality['gates'][network]['passed']:
        raise ValueError('integer accuracy admission required before board frame preparation')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA_ENVIRONMENT_NOT_READY')
    torch.set_num_threads(2);frozen.seed_all(42)
    folder=root/network/'seed_42';traces=(folder/'parity/train_traces').resolve()
    trace=gate.read_json(traces/'TRACE_MANIFEST.json')
    package_path=Path(result['package_manifest_path']);package=package_path.parent if package_path.is_absolute() else (root/package_path).parent
    graph=gate.read_json(package/'graph.json')
    if trace.get('split')!='train' or trace.get('count')!=8 or trace.get('test_evaluated') is not False:
        raise ValueError('fixed train8 trace manifest required')
    if trace['graph_sha256']!=gate.sha256(package/'graph.json') or trace['calibration_manifest_sha256']!=expected['calibration_sha256']:
        raise ValueError('trace graph/calibration binding mismatch')
    for name,digest in trace['files'].items():
        path=(traces/name).resolve()
        if not path.is_relative_to(traces) or gate.sha256(path)!=digest:raise ValueError('trace artifact changed: '+name)
    out=folder/'hardware/frame_bundle'
    if (out/'frames.json').is_file():
        previous=gate.read_json(out/'frames.json')
        if previous['graph_sha256']!=trace['graph_sha256']:raise ValueError('frame set belongs to another graph')
        for name,digest in previous['files'].items():
            if gate.sha256(out/name)!=digest:raise ValueError('existing frame file changed')
        return previous
    out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    rows=[];files={}
    for row in trace['frames']:
        layers=gate.read_json(traces/row['folder']/'layers.json')
        hashes={layer['node']:layer['sha256'] for layer in layers}
        path=f"frames/input_{len(rows):03d}.npy"
        shutil.copy2(traces/row['input_path'],out/path);files[path]=gate.sha256(out/path)
        rows.append(dict(sample_id=row['source']['image'],split='train',input=path,
            source=row['source'],expected_layers=hashes,
            expected_heads={level:hashes[node] for level,node in graph['outputs'].items()}))
    config=baseline();data=frozen.resolve(config['dataset']['data_dir'])
    loader=frozen.make_loader(config,data,'valid',42,0)
    if not isinstance(loader.sampler,torch.utils.data.SequentialSampler):raise ValueError('valid frame order must be deterministic')
    valid=frozen.bind_dataset(loader);pinned=gate.read_json(root/'VALID_MANIFEST.json')
    if valid['images']!=99 or valid['entries']!=pinned['entries']:raise ValueError('valid split/order changed')
    integer=IntegerGraphModel(package).to(device).eval();count=0
    with torch.inference_mode():
        for images,targets in loader:
            heads,_=integer.forward_quantized(images.to(device))
            exp=graph['inputs']['image']['exp']
            quantized=torch.round(images.to(torch.float64)/(2.0**exp)).clamp(-128,127).to(torch.int8)
            for index,target in enumerate(targets):
                entry=valid['entries'][count]
                sample_id=Path(target['path']).relative_to(loader.dataset.root).as_posix()
                if sample_id!=entry['image'].replace('\\','/'):raise ValueError('valid loader/manifest order mismatch')
                path=f"frames/input_{len(rows):03d}.npy"
                np.save(out/path,quantized[index].numpy(),allow_pickle=False);files[path]=gate.sha256(out/path)
                raw={level:value[index].cpu().numpy().transpose(1,2,0).copy().tobytes() for level,value in heads.items()}
                rows.append(dict(sample_id=sample_id,split='valid',input=path,source=entry,
                    expected_heads={level:hashlib.sha256(value).hexdigest() for level,value in raw.items()}))
                count+=1
            gate.atomic_json(out/'progress.json',dict(status='PREPARING',train=8,valid=count,test_evaluated=False))
    if count!=99:raise ValueError('valid count mismatch')
    payload=dict(version=1,graph_sha256=trace['graph_sha256'],network=network,seed=42,
        package_manifest_sha256=result['package_manifest_sha256'],
        trace_manifest_sha256=gate.sha256(traces/'TRACE_MANIFEST.json'),
        calibration_manifest_sha256=expected['calibration_sha256'],valid_split_manifest_sha256=expected['split_manifest_sha256'],
        calibration_first8_sample_ids=[row['sample_id'] for row in rows[:8]],frames=rows,files=files,
        test_evaluated=False,warmup=1,repetitions=10,reference='same canonical integer model used in valid accuracy admission')
    gate.atomic_json(out/'frames.json',payload)
    gate.atomic_json(out/'progress.json',dict(status='PREPARED_NOT_BOARD_EXECUTED',train=8,valid=99,test_evaluated=False))
    return payload


def install_frames(build,bundle):
    """Attach an immutable frame manifest to a completed build before packaging."""
    build,bundle=Path(build).resolve(),Path(bundle).resolve()
    manifest=gate.read_json(bundle/'frames.json');source=gate.read_json(build/'manifest.json')
    if manifest['graph_sha256']!=source['graph_sha256']:raise ValueError('frame/build graph mismatch')
    for name,digest in {**manifest['files'],'frames.json':gate.sha256(bundle/'frames.json')}.items():
        src=(bundle/name).resolve();dst=(build/name).resolve()
        if not src.is_relative_to(bundle) or not dst.is_relative_to(build) or gate.sha256(src)!=digest:
            raise ValueError('unsafe frame binding')
        if dst.exists() and gate.sha256(dst)!=digest:raise ValueError('cannot overwrite different bound frame')
        if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    return dict(status='FRAMES_BOUND',frames=107,graph_sha256=manifest['graph_sha256'],test_evaluated=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--network',choices=['D','C','B1'],required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--install',action='store_true')
    args=parser.parse_args();result=prepare_frames(args.root,args.network,args.device)
    if args.install:
        folder=args.root/args.network/'seed_42'/'hardware'
        result=install_frames(folder/'graph_build',folder/'frame_bundle')
    print(json.dumps({k:v for k,v in result.items() if k not in ('frames','files')},indent=2))
