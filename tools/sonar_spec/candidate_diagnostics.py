"""Observations of unchanged assignment/decode; no training policy changes."""
import statistics
import torch
from hwnas_fpga.training.detection_trainer import (
    _assign_targets, _anchor_iou, _reshape_predictions, _decode_boxes_for_loss,
    _nms_xyxy, decode_detection_predictions,
)


@torch.no_grad()
def assignment_diagnostics(predictions, targets, config, levels):
    assigned = _assign_targets(predictions, targets, config, device=predictions[0].device)
    total_gt = sum(len(t['boxes']) for t in targets)
    result = []
    for li, (level, slots) in enumerate(zip(levels, assigned)):
        obj = slots['objectness']
        indices = (obj > .5).nonzero()
        ious = []
        for b, a, y, x in indices.tolist():
            box = slots['box_xyxy'][b, a, y, x]
            aw, ah = config.anchor_sizes[li][a]
            ious.append(_anchor_iou(max(1., float(box[2]-box[0])), max(1., float(box[3]-box[1])), aw, ah))
        matched = 0
        for b, target in enumerate(targets):
            positive = obj[b] > .5
            boxes, labels = slots['box_xyxy'][b][positive], slots['label'][b][positive]
            for box, label in zip(target['boxes'], target['labels']):
                matched += int(bool(((boxes == box.to(boxes.device)).all(-1) & (labels == int(label))).any()))
        result.append(dict(level=level, positive=int((obj>.5).sum()), ignore=int((obj<0).sum()),
            negative=int((obj==0).sum()), matched_iou_mean=statistics.mean(ious) if ious else None,
            matched_iou_median=statistics.median(ious) if ious else None,
            gt_matched_count=matched, gt_total=total_gt, gt_coverage=matched/total_gt if total_gt else None))
    return result


@torch.no_grad()
def topk_diagnostics(predictions, config, levels, protocol):
    decoded = decode_detection_predictions(predictions, config=config,
        score_threshold=protocol['ap_decode_score_threshold'], nms_iou_threshold=protocol['nms_iou_threshold'],
        pre_nms_topk=protocol['pre_nms_topk'], max_detections=protocol['post_nms_max_detections_per_image'])
    parts = []
    for i, raw in enumerate(predictions):
        p = _reshape_predictions(raw, outputs_per_anchor=7, anchors_per_level=3)
        boxes = _decode_boxes_for_loss(p[..., :4], level_index=i, config=config)
        prob, label = p[..., 5:].softmax(-1).max(-1)
        score = p[..., 4].sigmoid()*prob
        parts.append((boxes.flatten(1, 3), score.flatten(1), label.flatten(1)))
    records = []
    for b in range(predictions[0].shape[0]):
        boxes, scores, labels, origins = [], [], [], []
        counts = []
        for i, (box, score, label) in enumerate(parts):
            mask = score[b] >= protocol['ap_decode_score_threshold']
            counts.append(dict(level=levels[i], raw_candidates=score.shape[1], before_global_topk=int(mask.sum())))
            boxes.append(box[b][mask]); scores.append(score[b][mask]); labels.append(label[b][mask])
            origins.append(torch.full_like(label[b][mask], i))
        boxes, scores, labels, origins = map(torch.cat, (boxes, scores, labels, origins))
        if scores.numel() > protocol['pre_nms_topk']:
            selected = scores.topk(protocol['pre_nms_topk']).indices
            boxes, scores, labels, origins = (v[selected] for v in (boxes, scores, labels, origins))
        for i, row in enumerate(counts): row['global_top500'] = int((origins==i).sum())
        keep = []
        for label in labels.unique().tolist():
            ids = torch.where(labels==label)[0]
            keep.extend(ids[_nms_xyxy(boxes[ids], scores[ids], iou_threshold=protocol['nms_iou_threshold'])].tolist())
        keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
        keep = keep[scores[keep].argsort(descending=True)][:protocol['post_nms_max_detections_per_image']]
        for i, row in enumerate(counts): row['post_nms'] = int((origins[keep]==i).sum())
        for key, value in [('boxes', boxes[keep]), ('scores', scores[keep]), ('labels', labels[keep])]:
            torch.testing.assert_close(value.cpu(), decoded[b][key], rtol=0, atol=0)
        assert torch.isfinite(decoded[b]['boxes']).all()
        assert ((decoded[b]['boxes'] >= 0) & (decoded[b]['boxes'] <= 640)).all()
        assert (decoded[b]['boxes'][:, 2:] >= decoded[b]['boxes'][:, :2]).all()
        records.append(dict(image_in_batch=b, levels=counts, decode_parity=True))
    return records
