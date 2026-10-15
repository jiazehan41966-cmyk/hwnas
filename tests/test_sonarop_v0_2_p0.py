"""Correctness tests for the isolated SonarOp V0.2-P0 prototype."""

from copy import deepcopy
import hashlib

import pytest
import torch

from hwnas_fpga.models.sonar_candidate import SPEC_PATH
from hwnas_fpga.models.sonar_op import ReparamDualRFDepthwise, build_sonarop_candidate


@pytest.mark.parametrize("batch", [1, 4])
def test_operator_shape_and_backward(batch):
    torch.manual_seed(7)
    op = ReparamDualRFDepthwise(480, stride=1).train()
    x = torch.randn(batch, 480, 20, 20, requires_grad=True)
    y = op(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    for branch in (op.branch3, op.branch5):
        assert branch[0].weight.grad is not None
        assert branch[1].weight.grad is not None
        assert branch[1].bias.grad is not None
        assert torch.isfinite(branch[0].weight.grad).all()


@pytest.mark.parametrize("batch", [1, 4])
def test_fusion_equivalence_and_determinism(batch):
    torch.manual_seed(11)
    op = ReparamDualRFDepthwise(32).train()
    for _ in range(3):
        op(torch.randn(4, 32, 20, 20))
    op.eval()
    k1, b1 = op.get_equivalent_kernel_bias()
    k2, b2 = op.get_equivalent_kernel_bias()
    assert torch.equal(k1, k2) and torch.equal(b1, b2)
    x = torch.randn(batch, 32, 20, 20)
    with torch.no_grad():
        expected = op(x)
        deployed = deepcopy(op).switch_to_deploy()
        actual = deployed(x)
    error = (actual - expected).abs()
    assert error.max().item() <= 1e-5
    assert deployed.reparam_conv.groups == 32
    assert deployed.reparam_conv.kernel_size == (5, 5)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in deployed.modules())


def test_train_and_deploy_state_dict_roundtrip(tmp_path):
    train = ReparamDualRFDepthwise(16).eval()
    train_path = tmp_path / "train.pt"
    torch.save(train.state_dict(), train_path)
    restored_train = ReparamDualRFDepthwise(16).eval()
    restored_train.load_state_dict(torch.load(train_path, weights_only=True), strict=True)
    x = torch.randn(2, 16, 20, 20)
    with torch.no_grad():
        torch.testing.assert_close(train(x), restored_train(x), rtol=0, atol=0)

    deploy = deepcopy(train).switch_to_deploy()
    deploy_path = tmp_path / "deploy.pt"
    torch.save(deploy.state_dict(), deploy_path)
    restored_deploy = ReparamDualRFDepthwise(16, deploy=True).eval()
    restored_deploy.load_state_dict(torch.load(deploy_path, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(deploy(x), restored_deploy(x), rtol=0, atol=0)


def test_candidate_stage7_only_and_v0_1_hash_protection():
    before = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    model = build_sonarop_candidate()
    op = model.sonarop()
    assert op.channels == 480 and op.stride == 1
    assert model.levels == ("P3", "P4", "P5")
    assert len([m for m in model.modules() if isinstance(m, ReparamDualRFDepthwise)]) == 1
    assert hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest() == before == "ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4"
