"""Retained adjacent convolution/batch-normalization folding."""
import torch
from torch import nn

def fold_batch_norms_inplace(model: nn.Module) -> list[dict[str, str]]:
    """Fold adjacent Conv2d/BatchNorm2d pairs while preserving their paths."""

    model.eval()
    folded: list[dict[str, str]] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        children = list(parent.named_children())
        for index in range(len(children) - 1):
            conv_name, conv = children[index]
            bn_name, bn = children[index + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                fused = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                setattr(parent, conv_name, fused)
                setattr(parent, bn_name, nn.Identity())
                folded.append(
                    {
                        "conv": f"{prefix}{conv_name}",
                        "bn": f"{prefix}{bn_name}",
                    }
                )
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.Identity):
                visit(child, f"{prefix}{name}.")

    visit(model, "")
    return folded
