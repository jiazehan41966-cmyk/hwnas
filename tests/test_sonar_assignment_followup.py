"""Frozen assignment audit regression contracts; no model or dataset inference."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"tools/sonar_operator_pool_p0"))
from assignment_followup import read_records, zero_reference
from hwnas_fpga.data.detection import _build_class_balanced_sampler
from hwnas_fpga.training.detection_trainer import (
    DetectionLossConfig, DenseAnchorDetectionLoss, _assign_targets,
    _decode_boxes_for_loss, _weighted_objectness_loss,
)


def test_no_test_split():
    with pytest.raises(ValueError,match="Only train and valid"):
        read_records(Path("not-opened"),"test",{})


def test_ignore_has_no_gradient_and_fixed_weight_bg_sign():
    cfg=DetectionLossConfig(num_classes=2, objectness_normalization="fixed_weights")
    stats=zero_reference({"objectness":torch.tensor([-1.,1.,0.,0.])},cfg)
    assert stats["reference_zero_logit_ignored_gradient_l1"] == 0
    assert stats["reference_zero_logit_positive_gradient_l1"] > 0
    z=torch.tensor([-8.,-2.,0.,2.,8.],requires_grad=True)
    y=torch.zeros_like(z)
    loss=_weighted_objectness_loss(z,y,valid_obj_mask=y>=0,positive_weight=1,
        negative_weight=.1,focal_gamma=2,focal_alpha=.75,normalization="fixed_weights")
    grad,=torch.autograd.grad(loss,z)
    assert torch.all(grad>0)  # Minimization decreases all background logits.


@pytest.mark.parametrize("label",[0,1])
def test_assignment_roundtrip_and_positive_only_classification(label):
    cfg=DetectionLossConfig(num_classes=2,image_size=64)
    outputs=[torch.zeros(1,21,64//s,64//s,requires_grad=True) for s in cfg.strides]
    targets=[{"boxes":[[22.,21.,36.,38.]],"labels":[label]}]
    assigned=_assign_targets(outputs,targets,cfg,device=torch.device("cpu"))
    total,parts=DenseAnchorDetectionLoss(cfg)(outputs,targets)
    grads=torch.autograd.grad(total,outputs)
    assert all(torch.isfinite(g).all() for g in grads)
    assert parts["positive_anchors"] > 0
    for li,(g,a) in enumerate(zip(grads,assigned)):
        pos=a["objectness"]>.5
        g=g.reshape(1,3,7,g.shape[-2],g.shape[-1]).permute(0,1,3,4,2)
        assert torch.all(g[...,5:][~pos]==0)
        if pos.any():
            assert torch.all(g[...,5:][pos].abs().sum(-1)>0)
            decoded=_decode_boxes_for_loss(a["box"],level_index=li,config=cfg)
            torch.testing.assert_close(decoded[pos],a["box_xyxy"][pos],atol=1e-4,rtol=0)


def test_balancing_is_per_class_total_not_per_image():
    ds=SimpleNamespace(sample_labels=[[0]]*234+[[1]]*96)
    sampler=_build_class_balanced_sampler(ds)
    p=sampler.weights/sampler.weights.sum()
    assert sampler.replacement and sampler.num_samples==330
    assert abs(float(p[:234].sum())-.5)<1e-12
    assert abs(float(p[234:].sum())-.5)<1e-12
    assert abs(float(p[-1]/p[0])-234/96)<1e-12
