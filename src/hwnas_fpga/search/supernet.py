"""SPOS-style single-path weight-sharing supernet for the mobile-anchor space.

Phase 3 of the rebuild replaces 300x 3-epoch scratch trainings (whose proxy
macro_f1 spread was 0.016 across the eventual top-4, i.e. no ranking signal)
with one shared-weight supernet:

- every block slot holds one weight set per (op, kernel, expand) choice at the
  stage's maximum channel width; a sampled path slices weights to its channel
  selection (slimmable-style), so all candidates inherit trained weights;
- training uses uniform single-path sampling per batch (SPOS, Guo et al.,
  ECCV 2020);
- candidate evaluation first recalibrates BatchNorm running statistics on
  training-side batches for the specific path, then scores it on the inner
  validation set. The outer protocol fold is never touched here.

Scope: operators ``mbconv`` (kernel/expand choices from the space config) and
``skip``, matching the semantic-safe operator policy. Sonar operators join
only after the numeric-parity gate exists.
"""

from __future__ import annotations

from random import Random
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwnas_fpga.search_space import (
    ArchitectureSpec,
    BlockSpec,
    SearchSpaceConfig,
    StageSpec,
)

SUPPORTED_OPS = ("mbconv", "skip")


class SlicedBatchNorm2d(nn.Module):
    """BatchNorm2d over the maximum width, sliceable to the active channels.

    Running statistics are views into the max-width buffers, so training and
    recalibration update the slice in place.
    """

    def __init__(self, max_channels: int, *, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.max_channels = int(max_channels)
        self.momentum: Optional[float] = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(max_channels))
        self.bias = nn.Parameter(torch.zeros(max_channels))
        self.register_buffer("running_mean", torch.zeros(max_channels))
        self.register_buffer("running_var", torch.ones(max_channels))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.num_batches_tracked.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if self.training:
            self.num_batches_tracked += 1
        momentum = self.momentum
        if momentum is None:  # cumulative moving average (BN recalibration)
            momentum = 1.0 / float(max(1, int(self.num_batches_tracked.item())))
        return F.batch_norm(
            x,
            self.running_mean[:channels],
            self.running_var[:channels],
            self.weight[:channels],
            self.bias[:channels],
            self.training,
            momentum,
            self.eps,
        )


class SupernetMBConv(nn.Module):
    """Weight container for one MBConv choice, sliceable in both widths."""

    def __init__(self, max_in: int, max_out: int, kernel_size: int, expand_ratio: int, stride: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.expand_ratio = int(expand_ratio)
        self.stride = int(stride)
        self.max_in = int(max_in)
        self.max_out = int(max_out)
        max_hidden = max_in * self.expand_ratio

        self.use_expand = self.expand_ratio > 1
        if self.use_expand:
            self.expand_weight = nn.Parameter(torch.empty(max_hidden, max_in, 1, 1))
            self.expand_bn = SlicedBatchNorm2d(max_hidden)
        self.dw_weight = nn.Parameter(
            torch.empty(max_hidden, 1, self.kernel_size, self.kernel_size)
        )
        self.dw_bn = SlicedBatchNorm2d(max_hidden)
        self.project_weight = nn.Parameter(torch.empty(max_out, max_hidden, 1, 1))
        self.project_bn = SlicedBatchNorm2d(max_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.use_expand:
            nn.init.kaiming_normal_(self.expand_weight, mode="fan_out")
        nn.init.kaiming_normal_(self.dw_weight, mode="fan_out")
        nn.init.kaiming_normal_(self.project_weight, mode="fan_out")

    def forward(self, x: torch.Tensor, *, in_channels: int, out_channels: int) -> torch.Tensor:
        identity = x
        hidden = in_channels * self.expand_ratio
        if self.use_expand:
            x = F.conv2d(x, self.expand_weight[:hidden, :in_channels])
            x = self.expand_bn(x)
            x = F.relu6(x, inplace=True)
        x = F.conv2d(
            x,
            self.dw_weight[:hidden],
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=hidden,
        )
        x = self.dw_bn(x)
        x = F.relu6(x, inplace=True)
        x = F.conv2d(x, self.project_weight[:out_channels, :hidden])
        x = self.project_bn(x)
        if self.stride == 1 and in_channels == out_channels:
            x = x + identity
        return x


class SupernetChoiceBlock(nn.Module):
    """One block slot with independent weights per operator choice."""

    def __init__(
        self,
        *,
        max_in: int,
        max_out: int,
        stride: int,
        kernel_choices: Sequence[int],
        expand_choices: Sequence[int],
        allow_skip: bool,
    ):
        super().__init__()
        self.stride = int(stride)
        self.choices = nn.ModuleDict()
        for kernel in kernel_choices:
            for expand in expand_choices:
                key = choice_key("mbconv", kernel, expand)
                self.choices[key] = SupernetMBConv(
                    max_in, max_out, kernel, expand, stride
                )
        self.allow_skip = bool(allow_skip) and self.stride == 1

    def choice_keys(self, *, in_channels: int, out_channels: int) -> list[str]:
        keys = list(self.choices.keys())
        if self.allow_skip and in_channels == out_channels:
            keys.append("skip")
        return keys

    def forward(
        self,
        x: torch.Tensor,
        *,
        choice: str,
        in_channels: int,
        out_channels: int,
    ) -> torch.Tensor:
        if choice == "skip":
            if not (self.allow_skip and in_channels == out_channels):
                raise ValueError("skip is illegal for this block placement")
            return x
        return self.choices[choice](
            x, in_channels=in_channels, out_channels=out_channels
        )


def choice_key(op: str, kernel: int, expand: int) -> str:
    return f"{op}_k{kernel}_e{expand}"


def parse_choice_key(key: str) -> BlockSpec:
    if key == "skip":
        return BlockSpec(op="skip", kernel_size=1, expand_ratio=1, stride=1)
    op, kernel_token, expand_token = key.rsplit("_", 2)
    return BlockSpec(
        op=op,
        kernel_size=int(kernel_token.lstrip("k")),
        expand_ratio=int(expand_token.lstrip("e")),
        stride=1,
    )


class MobileAnchorSupernet(nn.Module):
    """Single-path supernet over a stage-based mobile-anchor search space."""

    def __init__(self, config: SearchSpaceConfig, *, num_classes: Optional[int] = None):
        super().__init__()
        if config.stage_channel_choices is None or config.stage_depth_choices is None:
            raise ValueError(
                "MobileAnchorSupernet requires per-stage channel and depth choices "
                "(use the mobile_anchor family profile)"
            )
        ops = tuple(op for op in config.op_choices if op in SUPPORTED_OPS)
        if "mbconv" not in ops:
            raise ValueError("supernet requires mbconv in op_choices")
        self.config = config
        self.num_classes = int(num_classes or config.num_classes or 8)
        self.allow_skip = "skip" in ops

        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, config.stem_channels, 3,
                      stride=config.stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(config.stem_channels),
            nn.ReLU(inplace=True),
        )

        self.stage_max_channels: list[int] = []
        self.stage_max_depth: list[int] = []
        self.blocks = nn.ModuleList()
        previous_max = config.stem_channels
        for stage_index, stride in enumerate(config.stage_strides):
            channel_choices = config.stage_channel_choices[stage_index]
            depth_choices = config.stage_depth_choices[stage_index]
            max_out = max(channel_choices)
            max_depth = max(depth_choices)
            stage_slots = nn.ModuleList()
            for block_index in range(max_depth):
                block_stride = stride if block_index == 0 else 1
                max_in = previous_max if block_index == 0 else max_out
                stage_slots.append(
                    SupernetChoiceBlock(
                        max_in=max_in,
                        max_out=max_out,
                        stride=block_stride,
                        kernel_choices=config.kernel_choices,
                        expand_choices=config.expand_choices,
                        allow_skip=self.allow_skip,
                    )
                )
            self.blocks.append(stage_slots)
            self.stage_max_channels.append(max_out)
            self.stage_max_depth.append(max_depth)
            previous_max = max_out

        head_conv_channels = config.head_conv_channels or 0
        if head_conv_channels > 0:
            self.head_conv_weight = nn.Parameter(
                torch.empty(head_conv_channels, previous_max, 1, 1)
            )
            nn.init.kaiming_normal_(self.head_conv_weight, mode="fan_out")
            self.head_conv_bn = nn.BatchNorm2d(head_conv_channels)
            classifier_in = head_conv_channels
        else:
            self.head_conv_weight = None
            classifier_in = previous_max
        self.classifier_in = classifier_in
        self.fc = nn.Linear(classifier_in, self.num_classes)

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------

    def sample_path(self, rng: Optional[Random] = None, seed: Optional[int] = None) -> dict[str, Any]:
        random = rng or Random(seed)
        stages: list[dict[str, Any]] = []
        previous_channels = self.config.stem_channels
        for stage_index in range(len(self.config.stage_strides)):
            channels = random.choice(self.config.stage_channel_choices[stage_index])
            depth = random.choice(self.config.stage_depth_choices[stage_index])
            choices: list[str] = []
            for block_index in range(depth):
                in_channels = previous_channels if block_index == 0 else channels
                slot = self.blocks[stage_index][block_index]
                keys = slot.choice_keys(in_channels=in_channels, out_channels=channels)
                choices.append(random.choice(keys))
            stages.append({"channels": int(channels), "depth": int(depth), "choices": choices})
            previous_channels = channels
        return {"stages": stages}

    def path_to_architecture(self, path: dict[str, Any]) -> ArchitectureSpec:
        stages: list[StageSpec] = []
        for stage_index, stage in enumerate(path["stages"]):
            stride = self.config.stage_strides[stage_index]
            blocks = []
            for block_index, key in enumerate(stage["choices"]):
                block_stride = stride if block_index == 0 else 1
                spec = parse_choice_key(key)
                blocks.append(
                    BlockSpec(
                        op=spec.op,
                        kernel_size=spec.kernel_size,
                        expand_ratio=spec.expand_ratio,
                        stride=block_stride,
                    )
                )
            stages.append(
                StageSpec(
                    channels=stage["channels"],
                    depth=stage["depth"],
                    stride=stride,
                    blocks=tuple(blocks),
                )
            )
        return ArchitectureSpec(
            input_channels=self.config.input_channels,
            stem_channels=self.config.stem_channels,
            stem_stride=self.config.stem_stride,
            post_stem_downsample_stride=self.config.post_stem_downsample_stride,
            stages=tuple(stages),
            head_conv_channels=self.config.head_conv_channels,
            head_channels=self.config.head_channels,
            num_classes=self.num_classes,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, path: dict[str, Any]) -> torch.Tensor:
        x = self.stem(x)
        previous_channels = self.config.stem_channels
        for stage_index, stage in enumerate(path["stages"]):
            channels = stage["channels"]
            for block_index, key in enumerate(stage["choices"]):
                in_channels = previous_channels if block_index == 0 else channels
                slot = self.blocks[stage_index][block_index]
                x = slot(x, choice=key, in_channels=in_channels, out_channels=channels)
            previous_channels = channels

        if self.head_conv_weight is not None:
            x = F.conv2d(x, self.head_conv_weight[:, :previous_channels])
            x = self.head_conv_bn(x)
            x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ----------------------------------------------------------------------
# Training and candidate evaluation
# ----------------------------------------------------------------------


def train_supernet(
    supernet: MobileAnchorSupernet,
    train_loader,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 3,
    device: Optional[str] = None,
    criterion: Optional[nn.Module] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Uniform single-path SPOS training."""
    from hwnas_fpga.training.recipe import build_warmup_cosine_scheduler

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    supernet = supernet.to(device)
    criterion = (criterion or nn.CrossEntropyLoss(label_smoothing=0.1)).to(device)
    optimizer = torch.optim.AdamW(supernet.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, epochs=epochs, warmup_epochs=warmup_epochs
    )
    rng = Random(seed)

    history: dict[str, Any] = {"train_loss": [], "train_acc": [], "lr": []}
    for epoch in range(int(epochs)):
        supernet.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            path = supernet.sample_path(rng)
            optimizer.zero_grad(set_to_none=True)
            outputs = supernet(inputs, path)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total_samples += targets.size(0)

        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
        history["train_loss"].append(total_loss / max(1, total_samples))
        history["train_acc"].append(total_correct / max(1, total_samples))
        if verbose:
            print(
                f"Supernet epoch {epoch + 1}/{epochs}: "
                f"loss={history['train_loss'][-1]:.4f} acc={history['train_acc'][-1]:.4f}"
            )
    return history


class _BoundPath(nn.Module):
    def __init__(self, supernet: MobileAnchorSupernet, path: dict[str, Any]):
        super().__init__()
        self.supernet = supernet
        self.path = path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.supernet(x, self.path)


@torch.no_grad()
def recalibrate_bn(
    supernet: MobileAnchorSupernet,
    path: dict[str, Any],
    loader,
    *,
    num_batches: int = 20,
    device: Optional[str] = None,
) -> None:
    """Reset and re-estimate BN running stats for one path (SPOS practice)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    supernet = supernet.to(device)
    saved_momentum: list[tuple[nn.Module, Optional[float]]] = []
    for module in supernet.modules():
        if isinstance(module, (SlicedBatchNorm2d, nn.BatchNorm2d)):
            module.reset_running_stats()
            saved_momentum.append((module, module.momentum))
            module.momentum = None
    supernet.train()
    batches = 0
    for inputs, _ in loader:
        if batches >= num_batches:
            break
        supernet(inputs.to(device), path)
        batches += 1
    for module, momentum in saved_momentum:
        module.momentum = momentum
    supernet.eval()


def evaluate_candidate(
    supernet: MobileAnchorSupernet,
    path: dict[str, Any],
    *,
    bn_loader,
    eval_loader,
    num_classes: int,
    bn_batches: int = 20,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Score one path with inherited weights after BN recalibration.

    ``bn_loader`` must be training-side data; ``eval_loader`` should be the
    inner validation set. Never pass the outer protocol fold here.
    """
    from hwnas_fpga.training.trainer import evaluate_classifier

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    recalibrate_bn(supernet, path, bn_loader, num_batches=bn_batches, device=device)
    bound = _BoundPath(supernet, path)
    summary = evaluate_classifier(
        bound,
        eval_loader,
        criterion=nn.CrossEntropyLoss().to(device),
        device=device,
        num_classes=num_classes,
    )
    summary["architecture"] = supernet.path_to_architecture(path).to_dict()
    return summary
