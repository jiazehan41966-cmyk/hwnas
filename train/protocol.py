"""Frozen B1/ELASTIC loss configuration, kept outside the model layer."""
from dataclasses import replace
from .loss import DetectionLossConfig


def loss_config_for_model(model, baseline_config):
    return replace(DetectionLossConfig.from_config(baseline_config),
        num_classes=model.num_classes,image_size=640,strides=model.feature_strides,
        anchor_sizes=model.anchor_sizes,
        class_loss_weights=tuple(1.0 for _ in range(model.num_classes)))


def validate_protocol(config, genotype):
    dc,tc,ep = config["dataset"],config["training"],config["evaluation_protocol"]
    if dc["input_channels"] != 1 or dc["image_size"] != 640 or dc["num_classes"] != genotype.num_classes:
        raise ValueError("Dataset input/class count differs from genotype")
    if len(dc["class_names"]) != genotype.num_classes:
        raise ValueError("Dataset class_names must cover genotype.num_classes")
    if (str(tc["optimizer"]).lower() != "adamw" or tc["lr"] != .001
        or tc["weight_decay"] != .0001 or tc.get("amp") is not False
        or tc.get("scheduler") != "none"):
        raise ValueError("Frozen training requires AdamW .001/.0001, no AMP/scheduler")
    expected = dict(checkpoint_tie_break="earliest epoch",ap_decode_score_threshold=0.,
        precision_recall_score_threshold=.5,precision_recall_iou=.5,
        nms_iou_threshold=.5,pre_nms_topk=500,post_nms_max_detections_per_image=100,
        max_detections_per_image_per_class=100)
    if any(ep.get(k) != v for k,v in expected.items()):
        raise ValueError("Frozen evaluation thresholds/caps/earliest-epoch selection changed")
    if ep.get("iou_thresholds") != [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]:
        raise ValueError("Frozen AP requires IoU .50:.05:.95")
    if ep.get("development_splits") != ["train","valid"] or not str(ep.get("test_policy","")).startswith("sealed"):
        raise ValueError("Frozen protocol requires train/valid only and sealed test")
    return config
