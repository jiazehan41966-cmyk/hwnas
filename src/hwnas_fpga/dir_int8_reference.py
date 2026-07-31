"""Integer reference for ``dir_mbconv3_split11_e3_v1``.

The reference freezes CHW layout, branch order, signed round-to-nearest with
ties away from zero, int8 saturation, int32 accumulation and residual order.
It intentionally models an isolated operator; checkpoint-derived scales and
weights are bound later by the post-training parity gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DIR_INT8_CONTRACT = {
    "layout": "CHW",
    "input_shape": [32, 28, 28],
    "expanded_shape": [96, 28, 28],
    "split": [48, 48],
    "branch_order": ["dw_1x3", "dw_3x1"],
    "accumulator": "signed_int32",
    "rounding": "round_to_nearest_ties_away_from_zero",
    "saturation": "signed_int8_-128_127",
    "expand_shift": 10,
    "depthwise_shift": 8,
    "project_shift": 12,
    "relu6_quantized_max": 96,
    "padding": "explicit_zero",
    "residual_order": "saturate(project_requant_plus_input)",
}


@dataclass(frozen=True)
class DirInt8Weights:
    expand: np.ndarray
    dw_1x3: np.ndarray
    dw_3x1: np.ndarray
    project: np.ndarray

    def validate(self) -> None:
        expected = {
            "expand": (96, 32),
            "dw_1x3": (48, 3),
            "dw_3x1": (48, 3),
            "project": (32, 96),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != np.int8:
                raise ValueError(
                    f"{name} must be int8 with shape {shape}, got "
                    f"{value.dtype} {value.shape}"
                )


def round_shift_signed(values: np.ndarray, shift: int) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.int64)
    half = 1 << (int(shift) - 1)
    positive = (values64 + half) >> int(shift)
    negative = -(((-values64) + half) >> int(shift))
    return np.where(values64 >= 0, positive, negative)


def saturate_int8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -128, 127).astype(np.int8)


def relu6_int8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, DIR_INT8_CONTRACT["relu6_quantized_max"]).astype(
        np.int8
    )


def dir_mbconv3_split11_e3_v1_int8(
    inputs: np.ndarray,
    weights: DirInt8Weights,
) -> np.ndarray:
    inputs = np.asarray(inputs)
    if inputs.dtype != np.int8 or inputs.shape != (32, 28, 28):
        raise ValueError("inputs must be int8 CHW [32,28,28]")
    weights.validate()

    expanded_acc = np.einsum(
        "oc,chw->ohw",
        weights.expand.astype(np.int32),
        inputs.astype(np.int32),
        dtype=np.int32,
        optimize=True,
    )
    expanded = relu6_int8(
        saturate_int8(
            round_shift_signed(
                expanded_acc, DIR_INT8_CONTRACT["expand_shift"]
            )
        )
    )

    directional_acc = np.zeros((96, 28, 28), dtype=np.int32)
    first = expanded[:48].astype(np.int32)
    second = expanded[48:].astype(np.int32)
    for tap in range(3):
        x_offset = tap - 1
        target_start = max(0, -x_offset)
        target_stop = min(28, 28 - x_offset)
        source_start = target_start + x_offset
        source_stop = target_stop + x_offset
        directional_acc[
            :48, :, target_start:target_stop
        ] += (
            first[:, :, source_start:source_stop]
            * weights.dw_1x3[:, tap, None, None].astype(np.int32)
        )
        y_offset = tap - 1
        target_start = max(0, -y_offset)
        target_stop = min(28, 28 - y_offset)
        source_start = target_start + y_offset
        source_stop = target_stop + y_offset
        directional_acc[
            48:, target_start:target_stop, :
        ] += (
            second[:, source_start:source_stop, :]
            * weights.dw_3x1[:, tap, None, None].astype(np.int32)
        )
    directional = relu6_int8(
        saturate_int8(
            round_shift_signed(
                directional_acc, DIR_INT8_CONTRACT["depthwise_shift"]
            )
        )
    )
    projected_acc = np.einsum(
        "oc,chw->ohw",
        weights.project.astype(np.int32),
        directional.astype(np.int32),
        dtype=np.int32,
        optimize=True,
    )
    projected = saturate_int8(
        round_shift_signed(
            projected_acc, DIR_INT8_CONTRACT["project_shift"]
        )
    ).astype(np.int16)
    return saturate_int8(projected + inputs.astype(np.int16))


def deterministic_weights(seed: int = 20260731) -> DirInt8Weights:
    rng = np.random.default_rng(seed)
    return DirInt8Weights(
        expand=rng.integers(-8, 9, size=(96, 32), dtype=np.int8),
        dw_1x3=rng.integers(-16, 17, size=(48, 3), dtype=np.int8),
        dw_3x1=rng.integers(-16, 17, size=(48, 3), dtype=np.int8),
        project=rng.integers(-8, 9, size=(32, 96), dtype=np.int8),
    )
