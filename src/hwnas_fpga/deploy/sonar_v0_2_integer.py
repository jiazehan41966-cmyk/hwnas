"""V0.2 exact INT32 inference reference; preserves the frozen V1 arithmetic.

Explicit FP64 dot products operate on integer values with a proved <2**53
bound. Conversion to INT64, modulo-32 wrap and INT64 nearest-even requantization
are explicit. This is an inference reference, not a differentiable QAT layer.
"""
from copy import deepcopy
from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.deploy.h2_ptq import H2FakeQuantConv, quantize_h2

INTEGER_REFERENCE = "SONAR_V0_2_EXACT_INT32_REFERENCE_V1"


def requantize_v1(values, multiplier, shift):
    if type(multiplier) is not int or not 1 <= multiplier <= 2**31-1:
        raise ValueError("invalid V1 multiplier")
    if type(shift) is not int or not 0 <= shift <= 31:
        raise ValueError("invalid V1 shift")
    product = values.to(torch.int64) * multiplier
    magnitude = product.abs()
    result = magnitude >> shift
    if shift:
        remainder = magnitude & ((1 << shift)-1)
        half = 1 << (shift-1)
        result += ((remainder > half) | ((remainder == half) & ((result & 1) == 1))).to(torch.int64)
    return torch.where(product < 0, -result, result).clamp(-128,127).to(torch.int8)


class V02ExactConv(nn.Module):
    """Standalone canonical convolution initialized from validated W8A8 buffers."""
    def __init__(self, source: H2FakeQuantConv):
        super().__init__()
        if (source.input_bits, source.weight_bits, source.output_bits) != (8,8,8):
            raise ValueError("V0.2 exact reference requires W8A8")
        if source.weight_granularity != "per_tensor" or not all((
                source.quantize_input, source.quantize_weight_values, source.quantize_output)):
            raise ValueError("V0.2 exact reference requires fully quantized per-tensor convolutions")
        for name in ("stride", "padding", "dilation", "groups", "input_exp", "output_exp",
                     "weight_exps", "activation", "multiplier", "shift"):
            setattr(self, name, deepcopy(getattr(source,name)))
        self.register_buffer("int_weight", source.int_weight.detach().clone())
        self.register_buffer("int_bias", source.int_bias.detach().clone())
        if self.dilation != (1,1):
            raise ValueError("dilation is outside the V0.2 operator contract")

    def forward(self, inputs):
        if inputs.ndim != 4 or not inputs.is_floating_point():
            raise ValueError("exact reference expects dequantized NCHW floating inputs")
        weight = self.int_weight
        co, ci_group, kh, kw = weight.shape
        batch, ci, height, width = inputs.shape
        if ci_group * self.groups != ci or self.groups not in (1,ci) or (self.groups != 1 and co != ci):
            raise ValueError("unsupported dense/depthwise geometry")
        terms = prod(weight.shape[1:])
        # Bounds cover every intermediate dot-product subtotal plus INT32 bias.
        if terms*128*128 + (1 << 31) >= 1 << 53:
            raise ValueError("FP64 integer representability bound exceeded")
        quantized = quantize_h2(inputs,self.input_exp).to(torch.float64)
        patches = F.unfold(quantized,(kh,kw),padding=self.padding,stride=self.stride)
        w64 = weight.to(torch.float64)
        if self.groups == 1:
            accumulator = torch.matmul(w64.reshape(co,-1),patches)
        else:
            accumulator = (patches.reshape(batch,ci,kh*kw,-1)
                           * w64.reshape(1,ci,kh*kw,1)).sum(dim=2)
        oh = (height + 2*self.padding[0]-kh)//self.stride[0]+1
        ow = (width + 2*self.padding[1]-kw)//self.stride[1]+1
        accumulator = accumulator.to(torch.int64).reshape(batch,co,oh,ow)
        accumulator += self.int_bias.to(torch.int64).view(1,-1,1,1)
        wrapped = ((accumulator + (1 << 31)) & ((1 << 32)-1)) - (1 << 31)
        output = requantize_v1(wrapped,self.multiplier,self.shift)
        if self.activation == "relu":
            output = output.clamp_min(0)
        elif self.activation == "relu6":
            q6 = min(127,max(0,round(6.0/(2.0**self.output_exp))))
            output = output.clamp(0,q6)
        elif self.activation != "linear":
            raise ValueError("invalid V0.2 activation")
        return output.to(inputs.dtype) * (2.0**self.output_exp)


def promote_exact_v0_2_convs(model):
    """Replace float32 approximate convolution references in an independent model."""
    for name, child in list(model.named_children()):
        if isinstance(child,H2FakeQuantConv):
            exact = V02ExactConv(child)
            exact.train(child.training)
            setattr(model,name,exact)
        else:
            promote_exact_v0_2_convs(child)
    return model
