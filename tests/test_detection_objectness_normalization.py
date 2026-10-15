from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from hwnas_fpga.training.detection_trainer import (
    _weighted_objectness_loss, DetectionLossConfig, DenseAnchorDetectionLoss,
)


def loss(x,y,mode="fixed_weights",mask=None):
    return _weighted_objectness_loss(x,y,valid_obj_mask=torch.ones_like(x,dtype=torch.bool) if mask is None else mask,
        positive_weight=1.,negative_weight=.2,focal_gamma=2.,focal_alpha=.75,normalization=mode)


def test_reproduces_legacy_wrong_background_gradient():
    x=torch.tensor([-4.,-2.,0.,2.],requires_grad=True)
    loss(x,torch.zeros_like(x),"legacy_dynamic").backward()
    assert (x.grad<0).sum()==3


@pytest.mark.parametrize("target",[0.,1.])
def test_fixed_normalizer_moves_every_anchor_toward_its_target(target):
    x=torch.tensor([-8.,-4.,-2.,0.,2.,4.,8.],requires_grad=True)
    loss(x,torch.full_like(x,target)).backward()
    assert bool((x.grad>0).all()) if target==0 else bool((x.grad<0).all())


def test_mixed_targets_match_elementwise_focal_and_constant_reduction():
    x=torch.tensor([-2.,1.,3.,-.7],requires_grad=True)
    y=torch.tensor([0.,1.,0.,1.])
    mask=torch.tensor([True,True,False,True])
    actual=loss(x,y,mask=mask)
    p=x[mask].sigmoid()
    target=y[mask]
    pt=p*target+(1-p)*(1-target)
    alpha=.75*target+.25*(1-target)
    weights=(target+.2*(1-target))*alpha
    expected=(F.binary_cross_entropy_with_logits(x[mask],target,reduction="none")*(1-pt)**2*weights).sum()/weights.sum()
    assert torch.allclose(actual,expected)
    actual.backward()
    assert x.grad[2]==0
    assert x.grad[0]>0 and x.grad[1]<0 and x.grad[3]<0


def test_all_ignored_has_connected_zero_loss():
    x=torch.tensor([1.,2.],requires_grad=True)
    loss(x,torch.zeros_like(x),mask=torch.zeros(2,dtype=torch.bool)).backward()
    assert torch.equal(x.grad,torch.zeros_like(x))


def test_invalid_normalization_fails_closed():
    with pytest.raises(ValueError,match="normalization"):
        DenseAnchorDetectionLoss(DetectionLossConfig(num_classes=2,objectness_normalization="typo"))
