"""Frozen baseline train/valid wrapper. No test evaluation or architecture search."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from detection_controlled_study import make_model, make_loader, seed_all, state_hash
from hwnas_fpga.runtime import load_config
from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss,DetectionLossConfig,decode_detection_predictions
from hwnas_fpga.training.detection_coco_eval import evaluate_detection_predictions_coco,COCO_IOU_THRESHOLDS
from hwnas_fpga.training.detection_metrics import box_iou_xyxy
from hwnas_fpga.training.trainer import create_optimizer
from hwnas_fpga.deploy.export import get_model_info
from analyze_dataset import read_labels,sha256,write_csv


def resolve(value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def dump(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def source_hashes():
    paths=[Path(__file__),ROOT/'scripts/detection_controlled_study.py',ROOT/'src/hwnas_fpga/runtime.py',
        ROOT/'tools/sonar_analysis/analyze_dataset.py']
    for directory in ('models','training','data','search_space','deploy'):
        paths.extend(sorted((ROOT/'src/hwnas_fpga'/directory).glob('*.py')))
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def bind_dataset(loader):
    ds=loader.dataset
    entries=[]
    for path in ds.samples:
        label=ds.labels_dir/(path.stem+'.txt')
        read_labels(label,len(ds.class_names))
        entries.append(dict(image=str(path.relative_to(ds.root)),image_sha256=sha256(path),
            annotation=str(label.relative_to(ds.root)),annotation_sha256=sha256(label)))
    digest=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    return dict(images=len(entries),sha256=digest,entries=entries,class_names=ds.class_names)


def precision_recall(predictions,targets,threshold):
    """Micro P/R at IoU .5. Count FPs even for classes with no GT in this subset."""
    tp=fp=total=0
    for pred,target in zip(predictions,targets):
        pb=torch.as_tensor(pred['boxes'],dtype=torch.float32).reshape(-1,4)
        ps=torch.as_tensor(pred['scores']);pl=torch.as_tensor(pred['labels'])
        gb=torch.as_tensor(target['boxes'],dtype=torch.float32).reshape(-1,4)
        gl=torch.as_tensor(target['labels']);total+=len(gb)
        used=set()
        for i in ps.argsort(descending=True).tolist():
            if float(ps[i])<threshold: continue
            candidates=[j for j in range(len(gb)) if j not in used and int(gl[j])==int(pl[i])]
            if not candidates:
                fp+=1;continue
            ious=box_iou_xyxy(pb[i:i+1],gb[candidates])[0]
            best=int(ious.argmax())
            if float(ious[best])>=.5:
                used.add(candidates[best]);tp+=1
            else: fp+=1
    return dict(Precision=tp/(tp+fp) if tp+fp else 0.,Recall=tp/total if total else 0.,
        true_positives=tp,false_positives=fp,ground_truths=total,score_threshold=threshold)


def metrics_from_records(records,config):
    predictions=[r['prediction'] for r in records];targets=[r['target'] for r in records]
    proto=config['evaluation_protocol']
    coco=evaluate_detection_predictions_coco(predictions,targets,
        num_classes=config['dataset']['num_classes'],class_names=config['dataset']['class_names'],
        iou_thresholds=COCO_IOU_THRESHOLDS,max_detections=proto['max_detections_per_image_per_class'],
        area_ranges={'all':(0.,float('inf'))})
    return dict(AP50=coco['coco_ap50'],AP75=coco['coco_ap75'],mAP50_95=coco['coco_ap'],
        **precision_recall(predictions,targets,proto['precision_recall_score_threshold']),
        coco_details=coco,protocol=proto,num_images=len(records))


def evaluate(model,loader,criterion,config,device,max_batches=None):
    model.eval();records=[];losses=[]
    proto=config['evaluation_protocol']
    with torch.inference_mode():
        for i,(images,targets) in enumerate(loader):
            if max_batches is not None and i>=max_batches: break
            output=model(images.to(device));finite_outputs(output)
            loss,parts=criterion(output,targets)
            if not math.isfinite(float(loss)) or not all(math.isfinite(float(x)) for x in parts.values()):
                raise ValueError('Nonfinite validation loss')
            losses.append(float(loss))
            predictions=decode_detection_predictions(output,config=criterion.config,
                score_threshold=proto['ap_decode_score_threshold'],nms_iou_threshold=proto['nms_iou_threshold'],
                pre_nms_topk=proto['pre_nms_topk'],max_detections=proto['post_nms_max_detections_per_image'])
            for pred,target in zip(predictions,targets):
                records.append(dict(image=str(Path(target['path']).relative_to(loader.dataset.root)),
                    prediction={k:pred[k].tolist() for k in ('boxes','scores','labels')},
                    target={k:target[k].tolist() for k in ('boxes','labels')}))
    if not records: raise ValueError('Empty validation set')
    summary=metrics_from_records(records,config);summary['validation_loss']=float(np.mean(losses))
    return summary,records


def finite_outputs(output):
    for name,tensors in output.items():
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError('Nonfinite model output '+name)


def gradient_summary(model):
    result={}
    for group in ('backbone','neck','head'):
        params=[(n,p) for n,p in model.named_parameters() if n.startswith(group+'.') and p.requires_grad]
        missing=[n for n,p in params if p.grad is None]
        nonfinite=[n for n,p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
        nonzero=sum(bool(torch.count_nonzero(p.grad)) for _,p in params if p.grad is not None)
        result[group]=dict(parameter_tensors=len(params),missing=missing,nonfinite=nonfinite,nonzero_gradient_tensors=nonzero)
        if missing or nonfinite or not nonzero: raise ValueError('Gradient validation failed '+str(result))
    return result


def prepare(config_path,manifest_path):
    config=load_config(str(config_path))
    seed_all(config['project']['seed'])
    model,arch=make_model(config)
    arch=json.loads(json.dumps(arch))
    data=resolve(config['dataset']['data_dir'])
    bindings={sp:bind_dataset(make_loader(config,data,sp,42,0)) for sp in ('train','valid')}
    original=load_config(str(resolve(config['baseline_freeze']['source'])))
    _,original_arch=make_model(original)
    if arch!=json.loads(json.dumps(original_arch)) or DetectionLossConfig.from_config(config)!=DetectionLossConfig.from_config(original):
        raise ValueError('Frozen architecture/loss differ from existing baseline')
    manifest=dict(config_sha256=sha256(config_path),architecture=arch,datasets=bindings,
        source_sha256=source_hashes(),test_performance_accessed=False,initial_state_sha256=state_hash(model),
        loss_config=asdict(DetectionLossConfig.from_config(config)),status='PROTOCOL_FROZEN_SANITY_PENDING')
    if manifest_path.exists(): raise FileExistsError(manifest_path)
    dump(manifest_path,manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','sanity','train','evaluate'])
    p.add_argument('--config',default='configs/baseline_sonar_frozen.yaml')
    p.add_argument('--output',help='New output directory for sanity/train/evaluate')
    p.add_argument('--checkpoint',help='Frozen-wrapper checkpoint only; evaluated on valid')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--workers',type=int,default=0,help='Resource override recorded in effective config')
    args=p.parse_args()
    torch.set_num_threads(2)
    config_path=resolve(args.config);config=load_config(str(config_path))
    manifest_path=resolve(config['baseline_freeze']['manifest'])
    if args.mode=='freeze':
        result=prepare(config_path,manifest_path);print('FROZEN',manifest_path);return
    if not args.output: p.error('--output required')
    if args.mode=='evaluate' and not args.checkpoint: p.error('--checkpoint required')
    out=resolve(args.output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if sha256(config_path)!=manifest['config_sha256'] or source_hashes()!=manifest['source_sha256']:
        raise ValueError('Frozen source/config hash drift; a new reviewed freeze is required')
    seed=config['project']['seed'];seed_all(seed)
    model,arch=make_model(config)
    arch=json.loads(json.dumps(arch))
    if arch!=manifest['architecture'] or state_hash(model)!=manifest['initial_state_sha256']:
        raise ValueError('Frozen architecture/seeded initialization mismatch')
    root=resolve(config['dataset']['data_dir'])
    train=make_loader(config,root,'train',seed,args.workers,train=True)
    valid=make_loader(config,root,'valid',seed,args.workers)
    for sp,loader in [('train',train),('valid',valid)]:
        if bind_dataset(loader)!=manifest['datasets'][sp]: raise ValueError('Frozen data drift: '+sp)
    out.mkdir(parents=True,exist_ok=True)
    tc=config['training'];criterion=DenseAnchorDetectionLoss(DetectionLossConfig.from_config(config))
    effective=copy.deepcopy(config)
    effective['runtime']=dict(mode=args.mode,device=args.device,workers=args.workers,torch_threads=2,
        epochs=1 if args.mode=='sanity' else tc['epochs'],max_train_batches=2 if args.mode=='sanity' else None,
        max_valid_batches=2 if args.mode=='sanity' else None,test_accessed=False)
    dump(out/'effective_config.json',effective)
    dump(out/'run_binding.json',dict(config_sha256=sha256(config_path),freeze_manifest_sha256=sha256(manifest_path),
        torch_version=torch.__version__,python=sys.version,numpy_version=np.__version__,args=vars(args)))
    model.to(args.device);criterion.to(args.device)
    # Reuse existing Conv/Linear MAC counter; isolate its random dummy input from training RNG.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device=='cuda' else []):
        info=get_model_info(model,(1,1,640,640))
    info['estimated_FLOPs_2x_MACs']=2*info['estimated_macs']
    info['scope']='Conv2d/Linear batch-1 estimate; excludes BN, activations, resize, decode and NMS'
    dump(out/'model_info.json',info)
    optimizer=create_optimizer(model,tc['optimizer'],tc['lr'],tc['weight_decay'])
    if args.mode=='evaluate':
        checkpoint=torch.load(resolve(args.checkpoint),map_location='cpu',weights_only=False)
        if checkpoint.get('config_sha256')!=sha256(config_path) or checkpoint.get('freeze_manifest_sha256')!=sha256(manifest_path):
            raise ValueError('Checkpoint does not match frozen protocol/source')
        model.load_state_dict(checkpoint['model_state_dict'])
        result,records=evaluate(model,valid,criterion,config,args.device)
        dump(out/'valid_metrics.json',result);dump(out/'valid_predictions.json',records)
        print(json.dumps({k:result[k] for k in ('AP50','AP75','mAP50_95','Precision','Recall')}));return
    started=time.perf_counter();history=[];best=-1.;sanity={}
    epochs=effective['runtime']['epochs'];limit=effective['runtime']['max_train_batches']
    for epoch in range(epochs):
        model.train();losses=[];epoch_start=time.perf_counter()
        for i,(images,targets) in enumerate(train):
            if limit is not None and i>=limit: break
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            output=model(images.to(args.device));finite_outputs(output)
            shapes=[list(t.shape) for t in output['predictions']]
            expected=[[len(images),21,80,80],[len(images),21,40,40],[len(images),21,20,20]]
            if shapes!=expected: raise ValueError(f'Unexpected shapes {shapes}')
            loss,parts=criterion(output,targets)
            if not torch.isfinite(loss) or not all(math.isfinite(float(x)) for x in parts.values()):
                raise ValueError('Nonfinite train loss')
            loss.backward();gradients=gradient_summary(model);optimizer.step()
            losses.append(float(loss.detach()))
            if epoch==i==0:
                sanity=dict(prediction_shapes=shapes,finite_outputs=True,loss_components=parts,gradients=gradients,
                    actual_train_images=[t['path'] for t in targets],batch_size=len(images),forward_loss_backward_pass=True)
            print(f'epoch={epoch+1} batch={i+1} loss={losses[-1]:.6f} seconds={time.perf_counter()-tick:.2f}',flush=True)
        metrics,records=evaluate(model,valid,criterion,config,args.device,effective['runtime']['max_valid_batches'])
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),train_batches=len(losses),
            valid=metrics,epoch_seconds=time.perf_counter()-epoch_start)
        history.append(record)
        with (out/'epoch_metrics.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(record)+'\n')
        if metrics['mAP50_95']>best:
            best=metrics['mAP50_95'];best_epoch=epoch+1
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                epoch=best_epoch,config_sha256=sha256(config_path),freeze_manifest_sha256=sha256(manifest_path),
                architecture=arch,loss_config=asdict(criterion.config),valid_metrics=metrics,
                evaluation_protocol=config['evaluation_protocol'],sanity_only=args.mode=='sanity'),out/'best_model.pt')
            dump(out/'best_valid_metrics.json',metrics);dump(out/'best_valid_predictions.json',records)
        print(f'validation epoch={epoch+1} mAP50_95={metrics["mAP50_95"]:.6f}',flush=True)
    sanity.update(validation_pass=True,short_train_pass=True,trained_batches=sum(r['train_batches'] for r in history))
    dump(out/'sanity_checks.json',sanity)
    status=dict(status='COMPLETE',mode=args.mode,sanity_only=args.mode=='sanity',epochs=epochs,best_epoch=best_epoch,
        total_train_and_validation_seconds=time.perf_counter()-started,best_valid=json.loads((out/'best_valid_metrics.json').read_text()),
        checkpoint=str(out/'best_model.pt'),checkpoint_sha256=sha256(out/'best_model.pt'),checkpoint_file_bytes=(out/'best_model.pt').stat().st_size,
        model_info=info,test_accessed=False,formal_80_epoch_metrics_available=args.mode=='train')
    dump(out/'status.json',status)
    print('COMPLETE',out,flush=True)


if __name__=='__main__': main()
