"""Explicit arithmetic boundary regressions independent of graph-export code."""
import pytest
import torch
from torch import nn

from hwnas_fpga.deploy.h2_ptq import H2FakeQuantConv
from hwnas_fpga.deploy.sonar_v0_2_integer import V02ExactConv, requantize_v1


def conv_reference(weight, bias, *, shift=24, activation="linear", groups=1, padding=0):
    co, ci, kh, kw = weight.shape
    conv = nn.Conv2d(ci*groups,co,(kh,kw),bias=True,groups=groups,padding=padding)
    with torch.no_grad():
        conv.weight.fill_(1.)
        conv.bias.zero_()
    source = H2FakeQuantConv(conv,input_exp=0,output_exp=0,
        weight_exps=[-6],activation=activation,weight_granularity="per_tensor")
    source.int_weight.copy_(weight.to(torch.int8))
    source.int_bias.copy_(bias.to(torch.int32))
    source.multiplier,source.shift = 1,shift
    return V02ExactConv(source)


def test_positive_and_negative_overflow_wrap_before_requantization():
    weight = torch.tensor([[[[1]]],[[[-1]]]],dtype=torch.int8)
    reference = conv_reference(weight,torch.tensor([2**31-1,-2**31]),shift=24)
    output = reference(torch.tensor([[[[1.]]]]))
    # +2147483648 -> -2147483648; -2147483649 -> +2147483647.
    torch.testing.assert_close(output,torch.tensor([[[[-128.]],[[127.]]]]),rtol=0,atol=0)


def test_large_bias_retains_integer_bit_lost_by_float32_accumulation():
    reference = conv_reference(torch.ones(1,1,1,1,dtype=torch.int8),
                               torch.tensor([2**24+2**17]),shift=18)
    output = reference(torch.tensor([[[[1.]]]]))
    # True quotient is 64.5 + 2**-18 -> 65. Float32 loses the +1
    # accumulator bit, producing an exact 64.5 tie that rounds to 64.
    assert output.item() == 65.


def test_nearest_even_signed_ties_are_exact():
    values = torch.tensor([-7,-5,-3,-1,1,3,5,7],dtype=torch.int64)
    expected = torch.tensor([-4,-2,-2,0,0,2,2,4],dtype=torch.int8)
    torch.testing.assert_close(requantize_v1(values,1,1),expected,rtol=0,atol=0)


def test_depthwise_padding_and_relu():
    weights = torch.tensor([[[[1,2,3],[4,5,6],[7,8,9]]],
                            [[[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]]]],dtype=torch.int8)
    reference = conv_reference(weights,torch.zeros(2,dtype=torch.int32),shift=0,activation="relu",groups=2,padding=1)
    inputs = torch.ones(1,2,3,3)
    expected = torch.tensor([[[[28.,39.,24.],[33.,45.,27.],[16.,21.,12.]],
                              [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]]])
    torch.testing.assert_close(reference(inputs),expected,rtol=0,atol=0)


@pytest.mark.parametrize("multiplier,shift", [(0,0),(2**31,0),(1,32),(1,-1),(True,0)])
def test_invalid_numeric_parameters_are_rejected(multiplier,shift):
    with pytest.raises(ValueError):
        requantize_v1(torch.tensor([1]),multiplier,shift)
