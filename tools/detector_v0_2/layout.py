"""Reversible byte layouts for the V0.2 standalone 128-bit HLS operators.

This specifies packing only. It does not implement a graph scheduler or AXI
integration, and a packed parameter file is not evidence of board deployment.
"""
from __future__ import annotations

import numpy as np

LAYOUT_CONTRACT = "SONAR_V0_2_OPERATOR_NHWC_PAD16_AXI128_V1"
LANES = 16


def _array(value, ndim):
    if not isinstance(value, np.ndarray) or value.dtype != np.int8 or value.ndim != ndim:
        raise ValueError(f"expected signed INT8 NumPy array with {ndim} dimensions")
    if any(size < 1 for size in value.shape):
        raise ValueError("empty tensor dimension")
    return value


def _shape(shape, ndim):
    if len(shape) != ndim or any(type(size) is not int or size < 1 for size in shape):
        raise ValueError(f"expected {ndim} positive integer dimensions")
    return tuple(shape)


def _payload(payload, shape):
    if not isinstance(payload, bytes) or len(payload) != int(np.prod(shape)):
        raise ValueError("packed byte count does not match tensor dimensions")
    return np.frombuffer(payload, dtype=np.int8).reshape(shape)


def _padded(channels):
    return ((channels + LANES - 1) // LANES) * LANES


def pack_activation(chw: np.ndarray) -> bytes:
    """Pack one CHW image/feature; lane zero is the first byte of each word."""
    chw = _array(chw, 3)
    channels, height, width = chw.shape
    packed = np.zeros((height, width, _padded(channels)), dtype=np.int8)
    packed[:, :, :channels] = chw.transpose(1, 2, 0)
    return packed.tobytes(order="C")


def unpack_activation(payload: bytes, chw_shape: tuple[int, int, int]) -> np.ndarray:
    channels, height, width = _shape(chw_shape, 3)
    packed = _payload(payload, (height, width, _padded(channels)))
    if np.any(packed[:, :, channels:]):
        raise ValueError("nonzero activation padding lanes")
    return packed[:, :, :channels].transpose(2, 0, 1).copy()


def pack_weights(oihw: np.ndarray, *, groups: int = 1) -> bytes:
    """Dense: [oc,ky,kx,Cin-pad16]. DW3: nine taps in one word per channel."""
    oihw = _array(oihw, 4)
    co, ci, kh, kw = oihw.shape
    if type(groups) is not int or groups < 1:
        raise ValueError("invalid groups")
    if kh != kw or kh not in (1, 3):
        raise ValueError("V0.2 operator accepts square k1/k3 weights")
    if groups == 1:
        packed = np.zeros((co, kh, kw, _padded(ci)), dtype=np.int8)
        packed[:, :, :, :ci] = oihw.transpose(0, 2, 3, 1)
    elif groups == co and ci == 1 and kh == 3:
        packed = np.zeros((co, LANES), dtype=np.int8)
        packed[:, :9] = oihw[:, 0].reshape(co, 9)
    else:
        raise ValueError("only dense or k3 depthwise channel groups are supported")
    return packed.tobytes(order="C")


def unpack_weights(payload: bytes, oihw_shape: tuple[int, int, int, int], *, groups: int = 1) -> np.ndarray:
    co, ci, kh, kw = _shape(oihw_shape, 4)
    if type(groups) is not int or groups < 1 or kh != kw or kh not in (1, 3):
        raise ValueError("invalid V0.2 kernel/groups")
    if groups == 1:
        packed = _payload(payload, (co, kh, kw, _padded(ci)))
        if np.any(packed[:, :, :, ci:]):
            raise ValueError("nonzero weight padding lanes")
        return packed[:, :, :, :ci].transpose(0, 3, 1, 2).copy()
    if groups == co and ci == 1 and kh == 3:
        packed = _payload(payload, (co, LANES))
        if np.any(packed[:, 9:]):
            raise ValueError("nonzero depthwise padding lanes")
        return packed[:, :9].reshape(co, 1, 3, 3).copy()
    raise ValueError("only dense or k3 depthwise channel groups are supported")
