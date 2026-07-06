import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwnas_fpga.deploy.ptq_eval import (
    FakeQuantizedOp,
    apply_ptq,
    fold_batch_norms_inplace,
    quantize_dequantize,
    restore_identity_bn_export_model,
    stratified_calibration_indices,
)


class QuantizeDequantizeTests(unittest.TestCase):
    def test_roundtrip_error_bounded_by_half_scale(self) -> None:
        tensor = torch.linspace(-1.0, 1.0, steps=101)
        scale = 1.0 / 127.0
        recovered = quantize_dequantize(tensor, scale)
        self.assertLessEqual((tensor - recovered).abs().max().item(), scale / 2 + 1e-8)

    def test_clamps_to_int8_range(self) -> None:
        tensor = torch.tensor([10.0, -10.0])
        result = quantize_dequantize(tensor, 0.01)
        self.assertAlmostEqual(result[0].item(), 1.27, places=6)
        self.assertAlmostEqual(result[1].item(), -1.27, places=6)

    def test_zero_scale_is_identity(self) -> None:
        tensor = torch.randn(8)
        self.assertTrue(torch.equal(quantize_dequantize(tensor, 0.0), tensor))


class FakeQuantizedOpTests(unittest.TestCase):
    def test_weights_are_quantized_at_wrap_time(self) -> None:
        linear = nn.Linear(4, 4, bias=False)
        original = linear.weight.data.clone()
        op = FakeQuantizedOp(linear)
        # Weight now sits on the INT8 grid defined by its scale.
        grid = torch.round(linear.weight.data / op.weight_scale)
        self.assertTrue(
            torch.allclose(linear.weight.data, grid * op.weight_scale, atol=1e-6)
        )
        self.assertLessEqual(
            (original - linear.weight.data).abs().max().item(),
            op.weight_scale / 2 + 1e-8,
        )

    def test_calibration_tracks_max_and_freezes(self) -> None:
        op = FakeQuantizedOp(nn.Linear(4, 2))
        op(torch.full((1, 4), 2.0))
        op(torch.full((1, 4), 5.0))
        op.freeze_calibration()
        self.assertAlmostEqual(op.act_scale.item(), 5.0 / 127.0, places=6)
        self.assertFalse(op.calibrating)


class ApplyPtqTests(unittest.TestCase):
    def _make_loader(self) -> DataLoader:
        torch.manual_seed(0)
        inputs = torch.randn(32, 8)
        targets = (inputs.sum(dim=1) > 0).long()
        return DataLoader(TensorDataset(inputs, targets), batch_size=8)

    def test_wraps_all_quantizable_ops(self) -> None:
        model = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)
        )
        meta = apply_ptq(model, self._make_loader(), device="cpu")
        self.assertEqual(meta["num_quantized_ops"], 2)
        self.assertEqual(len(meta["activation_scales"]), 2)
        self.assertTrue(all(scale > 0 for scale in meta["activation_scales"]))

    def test_ptq_model_output_stays_close_to_fp32(self) -> None:
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        probe = torch.randn(4, 8)
        fp32_out = model(probe).detach()
        apply_ptq(model, self._make_loader(), device="cpu")
        int8_out = model(probe).detach()
        # INT8 simulation should perturb, not destroy, the outputs.
        self.assertLess((fp32_out - int8_out).abs().max().item(), 0.2)
        self.assertFalse(torch.equal(fp32_out, int8_out))

    def test_rejects_model_without_quantizable_ops(self) -> None:
        with self.assertRaises(ValueError):
            apply_ptq(nn.Sequential(nn.ReLU()), self._make_loader(), device="cpu")

    def test_bn_is_folded_before_weight_quantization(self) -> None:
        model = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        ).eval()
        inputs = torch.randn(4, 1, 8, 8)
        loader = DataLoader(
            TensorDataset(inputs, torch.zeros(4, dtype=torch.long)),
            batch_size=2,
        )
        meta = apply_ptq(model, loader, device="cpu")
        self.assertEqual(meta["bn_folding"]["folded_pair_count"], 1)
        self.assertIsInstance(model[1], nn.Identity)
        self.assertTrue(meta["fixed_point_contract"]["bn_folding"])

    def test_fused_qat_weights_can_return_to_original_topology(self) -> None:
        template = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
        ).eval()
        fused = __import__("copy").deepcopy(template)
        pairs = fold_batch_norms_inplace(fused)
        restored = restore_identity_bn_export_model(
            fused,
            __import__("copy").deepcopy(template),
            pairs,
        ).eval()
        probe = torch.randn(2, 1, 8, 8)
        self.assertTrue(
            torch.allclose(fused(probe), restored(probe), atol=1e-5)
        )


class CalibrationSubsetTests(unittest.TestCase):
    def test_is_deterministic_and_keeps_rare_class(self) -> None:
        labels = [0] * 20 + [1] * 2 + [2]
        first = stratified_calibration_indices(
            labels,
            list(range(len(labels))),
            max_samples=10,
            seed=42,
        )
        second = stratified_calibration_indices(
            labels,
            list(range(len(labels))),
            max_samples=10,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual({labels[index] for index in first}, {0, 1, 2})

    def test_rejects_empty_candidates(self) -> None:
        with self.assertRaises(ValueError):
            stratified_calibration_indices([], [], max_samples=10)


if __name__ == "__main__":
    unittest.main()
