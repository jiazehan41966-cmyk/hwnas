"""Bounded FCOS architecture choices; no dataset or weight download required."""
import gc
from collections import OrderedDict
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_fcos_pilot.model import (
    COMBINATIONS, DEFAULT_CACHE, build_model, dilation_witness,
    initialization_audit, model_contract, state_digest,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_all_eight_candidates_have_identical_initial_tensors_and_parameter_count():
    reference_digest = None
    reference_count = None
    for combo in COMBINATIONS:
        model = build_model(combo, num_classes=11, seed=42, pretrained=False)
        contract = model_contract(model)
        digest = state_digest(model)
        if reference_digest is None:
            reference_digest = digest
            reference_count = contract["parameter_count"]
        assert digest == reference_digest
        assert contract["parameter_count"] == reference_count
        assert contract["all_backbone_convolutions_trainable"]
        assert len(model.backbone.fpn.layer_blocks) == 3
        assert model.backbone.fpn.extra_blocks.p6.dilation == (1, 1)
        assert model.backbone.fpn.extra_blocks.p7.dilation == (1, 1)
        for index, block in enumerate(model.backbone.fpn.layer_blocks):
            dilation = 1 if combo[index] == "A" else 2
            assert block[0].dilation == (dilation, dilation)
            assert block[0].padding == (dilation, dilation)
            assert block[0].weight.shape == (256, 256, 3, 3)
        assert model.head.classification_head.cls_logits.out_channels == 11
        del model
        gc.collect()


def test_fpn_all_levels_keep_shape_and_changed_operations_have_nonzero_gradient():
    model = build_model("BAB", num_classes=3, seed=42, pretrained=False)
    features = OrderedDict((name, torch.rand(1, channels, side, side, requires_grad=True)) for name, channels, side in [("0", 512, 16), ("1", 1024, 8), ("2", 2048, 4)])
    outputs = model.backbone.fpn(features)
    assert [tuple(value.shape) for value in outputs.values()] == [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4), (1, 256, 2, 2), (1, 256, 1, 1)]
    sum(value.square().mean() for value in outputs.values()).backward()
    for block in model.backbone.fpn.layer_blocks:
        gradient = block[0].weight.grad
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    witness = dilation_witness()
    assert witness["output_difference_l1"] == 16.0
    assert witness["input_gradient_difference_l1"] == 16.0
    assert witness["finite"]


def test_transform_preserves_letterboxed_geometry_and_targets():
    model = build_model("AAA", num_classes=3, seed=42, pretrained=False)
    image = torch.zeros(3, 640, 640)
    target = {"boxes": torch.tensor([[12., 24., 200., 300.]]), "labels": torch.tensor([1])}
    transformed, targets = model.transform([image], [target])
    assert transformed.tensors.shape == (1, 3, 640, 640)
    torch.testing.assert_close(targets[0]["boxes"], target["boxes"])
    assert model.score_thresh == 0.001
    assert model.detections_per_img == 100
    contract = model_contract(model)
    assert contract["foreground_label_range"] == [1, 2]
    assert contract["pretrained"]["enabled"] is False


def test_candidate_seed_changes_only_initialization_and_invalid_requests_fail():
    first = build_model("AAA", num_classes=3, seed=42, pretrained=False)
    first_logits = first.head.classification_head.cls_logits.weight.detach().clone()
    del first
    second = build_model("AAA", num_classes=3, seed=43, pretrained=False)
    assert not torch.equal(first_logits, second.head.classification_head.cls_logits.weight)
    with pytest.raises(ValueError, match="combo"):
        build_model("AAC", 3, 42, pretrained=False)
    with pytest.raises(ValueError, match="background"):
        build_model("AAA", 1, 42, pretrained=False)
    with pytest.raises(ValueError, match="640"):
        build_model("AAA", 3, 42, pretrained=False, image_size=320)


@pytest.mark.skipif(not (DEFAULT_CACHE / "checkpoints" / "fcos_resnet50_fpn_coco-99b0c9b7.pth").is_file(), reason="Official weights not locally cached; tests never download")
def test_pretrained_tensor_parity_and_aaa_official_eval_equivalence():
    result = initialization_audit(pretrained=True)
    assert result["status"] == "PASS"
    assert result["all_initial_tensor_states_identical"]
    assert result["AAA_matches_independent_official_eval"]
    assert len(result["candidates"]) == 8
