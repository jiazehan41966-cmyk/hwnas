"""Frozen valid-only COCO-style all-area AP wrapper; no threshold sweep."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import torch
from .loss import decode_detection_predictions
from .detection_metrics import box_iou_xyxy
from .detection_coco_eval import evaluate_detection_predictions_coco, COCO_IOU_THRESHOLDS

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
    if getattr(loader.dataset,"split",None) not in {"valid", "val"}:
        raise ValueError("Evaluation requires the valid split; test is sealed")
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
