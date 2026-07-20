import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwnas_fpga.deploy.qat import (
    QATQuantizedOp,
    fake_quant_ste,
    finalize_qat,
    prepare_qat,
)


class QATTests(unittest.TestCase):
    def test_ste_preserves_gradient(self) -> None:
        tensor = torch.tensor([0.13], requires_grad=True)
        fake_quant_ste(tensor, 0.1).sum().backward()
        self.assertEqual(tensor.grad.item(), 1.0)

    def test_prepare_and_finalize(self) -> None:
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)),
            batch_size=4,
        )
        metadata = prepare_qat(model, loader, device="cpu")
        self.assertEqual(metadata["num_quantized_ops"], 2)
        self.assertIsInstance(model[0], QATQuantizedOp)
        output = model(torch.randn(2, 4))
        output.sum().backward()
        self.assertIsNotNone(model[0].module.weight.grad)
        finalized = finalize_qat(model)
        self.assertIsInstance(finalized[0], nn.Linear)
        self.assertFalse(any(isinstance(item, QATQuantizedOp) for item in finalized.modules()))


if __name__ == "__main__":
    unittest.main()
