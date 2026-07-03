"""Bridge official ProxylessNAS learned nets to HW-NAS artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, StageSpec


def _parse_block_from_layer(layer_cfg: Mapping[str, Any]) -> Optional[tuple[str, int, int, int]]:
    name = str(layer_cfg.get("name", ""))
    if name == "ZeroLayer":
        return None
    if name == "ConvLayer":
        return (
            "conv",
            int(layer_cfg.get("kernel_size", 1)),
            1,
            int(layer_cfg.get("stride", 1)),
        )
    if name == "MBInvertedConvLayer":
        return (
            "mbconv",
            int(layer_cfg.get("kernel_size", 3)),
            int(layer_cfg.get("expand_ratio", 1)),
            int(layer_cfg.get("stride", 1)),
        )
    return None


def proxyless_net_config_to_architecture(
    net_config: Mapping[str, Any],
    *,
    input_channels: int,
    num_classes: int,
    width_stages: Sequence[int],
    n_cell_stages: Sequence[int],
    stride_stages: Sequence[int],
    stem_stride: int = 2,
    post_stem_downsample_stride: int = 1,
    head_conv_channels: Optional[int] = None,
    skip_fixed_first_block: bool = True,
) -> ArchitectureSpec:
    """Map an official ProxylessNAS net.config to HW-NAS ArchitectureSpec."""
    first_conv = net_config.get("first_conv") or {}
    stem_channels = int(first_conv.get("out_channels", width_stages[0]))

    blocks = list(net_config.get("blocks") or [])
    if not blocks:
        raise ValueError("official proxyless net.config has no blocks")

    stage_blocks: list[list[BlockSpec]] = [[] for _ in width_stages]
    block_cursor = 0

    # Official ProxylessNAS normally has a fixed first block before searchable cells.
    if skip_fixed_first_block and block_cursor < len(blocks):
        block_cursor += 1

    for stage_idx, (channels, depth, stage_stride) in enumerate(
        zip(width_stages, n_cell_stages, stride_stages)
    ):
        for cell_idx in range(int(depth)):
            if block_cursor >= len(blocks):
                raise ValueError(
                    "official proxyless net.config does not contain enough blocks "
                    f"for stage layout {list(width_stages)}/{list(n_cell_stages)}"
                )
            block_cfg = blocks[block_cursor]
            block_cursor += 1
            conv_cfg = (block_cfg.get("mobile_inverted_conv") or {})
            stride = stage_stride if cell_idx == 0 else 1
            parsed = _parse_block_from_layer(conv_cfg)
            if parsed is None:
                stage_blocks[stage_idx].append(
                    BlockSpec(op="skip", kernel_size=1, expand_ratio=1, stride=stride)
                )
                continue
            op, kernel_size, expand_ratio, _ = parsed
            stage_blocks[stage_idx].append(
                BlockSpec(
                    op=op,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    stride=stride,
                )
            )

    stages = tuple(
        StageSpec(
            channels=int(channels),
            depth=len(blocks_for_stage),
            stride=int(stage_stride),
            blocks=tuple(blocks_for_stage),
        )
        for (channels, stage_stride, blocks_for_stage) in zip(
            width_stages,
            stride_stages,
            stage_blocks,
        )
    )
    return ArchitectureSpec(
        input_channels=int(input_channels),
        stem_channels=int(stem_channels),
        stages=stages,
        stem_stride=int(stem_stride),
        post_stem_downsample_stride=int(post_stem_downsample_stride),
        head_conv_channels=head_conv_channels,
        head_channels=head_conv_channels,
        num_classes=int(num_classes),
    )
