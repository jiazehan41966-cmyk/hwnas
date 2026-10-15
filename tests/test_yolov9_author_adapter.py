"""Boundary tests for author targets, branch choice, transfer and inference cost."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.sonar_yolov9_baseline.author_adapter import (
    AUTHOR_LOSS_GAINS,
    author_loss,
    build_author_model,
    build_main_inference_model,
    decode_author,
    forward_author,
    gray_to_three_channels,
    targets_to_yolo,
)


def test_gray_repeat_retains_preprocessed_pixels_and_rejects_rgb():
    gray = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
    repeated = gray_to_three_channels(gray)
    assert repeated.shape == (1, 3, 2, 2)
    for channel in range(3):
        torch.testing.assert_close(repeated[:, channel], gray[:, 0], rtol=0, atol=0)
    repeated[:, 2, 0, 0] = 0.1
    with pytest.raises(ValueError, match="identical grayscale"):
        gray_to_three_channels(repeated)


def test_native_targets_preserve_class_zero_and_batch_index_with_empty_image():
    targets = [
        {"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.long)},
        {"boxes": torch.tensor([[160, 240, 320, 400], [0, 0, 64, 128]]),
         "labels": torch.tensor([0, 1])},
    ]
    native = targets_to_yolo(targets)
    torch.testing.assert_close(native, torch.tensor([
        [1, 0, 0.375, 0.5, 0.25, 0.25],
        [1, 1, 0.05, 0.1, 0.1, 0.2],
    ]))
    assert targets_to_yolo(targets[:1]).shape == (0, 6)
    with pytest.raises(ValueError, match="foreground labels"):
        targets_to_yolo([{"boxes": [[0, 0, 4, 4]], "labels": [2]}])


def test_main_branch_and_frozen_nms_caps():
    model = torch.nn.Identity().eval()
    # Four main boxes: same-class duplicates, an overlapping other class and
    # a low-scoring distant box. Auxiliary boxes must never reach evaluator.
    main = torch.tensor([[
        [10, 10, 10, 30], [10, 10, 10, 30], [6, 6, 6, 6], [6, 6, 6, 6],
        [0.9, 0.8, 0.1, 0.4], [0.1, 0.2, 0.85, 0.1],
    ]], dtype=torch.float32)
    auxiliary = main.clone()
    auxiliary[:, :2] += 100
    output = ([auxiliary, main], [[], []])
    prediction = decode_author(model, output, pre_nms_topk=3, max_detections=100)[0]
    assert prediction["labels"].tolist() == [0, 1]
    torch.testing.assert_close(prediction["scores"], torch.tensor([0.9, 0.85]))
    torch.testing.assert_close(prediction["boxes"], torch.tensor([[7, 7, 13, 13], [7, 7, 13, 13]]).float())
    capped = decode_author(model, output, pre_nms_topk=3, max_detections=1)[0]
    assert len(capped["scores"]) == 1
    assert model.training is False


def test_scratch_rejects_checkpoint_and_hidden_assigner_override(monkeypatch):
    with pytest.raises(ValueError, match="rejects any checkpoint"):
        build_author_model("not_needed", "scratch", "anything.pt")
    monkeypatch.setenv("YOLOM", "5")
    with pytest.raises(ValueError, match="assignment environment"):
        build_author_model("not_needed", "scratch")


@pytest.fixture
def author_model():
    source = ROOT / "reference/yolov9"
    if not source.is_dir():
        pytest.skip("Pinned author source not staged")
    dependencies = ("cv2", "IPython", "seaborn", "requests", "psutil")
    if any(importlib.util.find_spec(name) is None for name in dependencies):
        pytest.skip("Native author environment not provisioned")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    try:
        yield build_author_model(source, "scratch")
    finally:
        torch.set_num_threads(previous_threads)


def test_native_dual_loss_backpropagates_to_both_branches(author_model):
    model = author_model.train()
    images = torch.rand(2, 1, 64, 64)
    targets = [
        {"boxes": torch.tensor([[15.0, 18.0, 42.0, 48.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[20.0, 16.0, 48.0, 44.0]]), "labels": torch.tensor([1])},
    ]
    output = forward_author(model, images)
    loss, parts = author_loss(model, output, targets, image_size=64)
    assert torch.isfinite(loss) and loss.item() > 0
    assert set(parts) == {"box", "cls", "dfl"}
    loss.backward()
    head = model.model[-1]
    for branch in (head.cv3, head.cv5):
        assert branch[0][-1].weight.grad is not None
        assert torch.isfinite(branch[0][-1].weight.grad).all()
        assert branch[0][-1].weight.grad.abs().sum() > 0
    assert model.author_adapter_metadata["loss_gains"] == AUTHOR_LOSS_GAINS
    assert model.author_adapter_metadata["nc_gain_scaling"] is False


def test_main_inference_pruning_preserves_predictions_and_rng(author_model):
    model = author_model.eval()
    images = torch.rand(1, 1, 64, 64)
    rng = torch.get_rng_state().clone()
    inference_model = build_main_inference_model(model)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    with torch.inference_mode():
        dual_main = forward_author(model, images)[0][1]
        single_main = forward_author(inference_model, images)[0]
    torch.testing.assert_close(dual_main, single_main, rtol=0, atol=0)
    assert sum(p.numel() for p in inference_model.parameters()) < sum(p.numel() for p in model.parameters())
    assert model.training is False
    assert type(model.model[-1]).__name__ == "DualDDetect"
    assert type(inference_model.model[-1]).__name__ == "DDetect"


def test_converted_transfer_reaches_main_regression_and_leaves_auxiliary_random():
    checkpoint = ROOT / "results/sonar_yolov9_baseline_20261005/author/yolov9-t-converted.pt"
    source = ROOT / "reference/yolov9"
    if not checkpoint.is_file() or any(importlib.util.find_spec(name) is None for name in ("cv2", "IPython", "seaborn", "requests", "psutil")):
        pytest.skip("Official tiny checkpoint and native environment not staged")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    scratch = build_author_model(source, "scratch")
    torch.manual_seed(42)
    pretrained = build_author_model(source, "coco_pretrained", checkpoint)
    receipt = pretrained.author_adapter_metadata
    assert receipt["checkpoint_architecture"] == "converted_single"
    assert receipt["auxiliary_pretrained"] is False
    assert receipt["key_mapping"]["model.22.cv2.0.2.weight"] == "model.29.cv4.0.2.weight"
    # The converted detection head occupies layer 22. It must not be loaded
    # accidentally into the same-numbered auxiliary SPPELAN layer.
    for key, tensor in pretrained.state_dict().items():
        if key.startswith(("model.22.", "model.23.", "model.24.", "model.25.", "model.26.", "model.27.", "model.28.", "model.29.cv2.", "model.29.cv3.")):
            torch.testing.assert_close(tensor, scratch.state_dict()[key], rtol=0, atol=0)
    assert "model.29.cv5.0.2.weight" in receipt["shape_mismatches"]
    assert receipt["loaded_tensor_count"] > 0
