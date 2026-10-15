"""Node-bound V0.2 PTQ; preserves V1 arithmetic without replaying legacy observers."""
from copy import deepcopy
from math import isfinite
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.deploy.h2_ptq import wrap_h2_ptq
from hwnas_fpga.deploy.sonar_v0_2_integer import INTEGER_REFERENCE, promote_exact_v0_2_convs
from hwnas_fpga.deploy.ptq_eval import fold_batch_norms_inplace

CALIBRATION_CONTRACT = "SONAR_V0_2_NODE_BOUND_CALIBRATION_V1"


@torch.no_grad()
def observe_v0_2_extrema(model: nn.Module, images: Iterable[torch.Tensor], *, device: str) -> dict:
    """Aggregate each Add by forward level index across all train batches.

    The result contains exactly levels-1 Add records in forward index order,
    even though the actual top-down execution visits those indices in reverse.
    Hooks and the temporary neck override are restored on errors as well.
    """
    if model.training:
        raise ValueError("calibration requires eval mode")
    neck = model.neck
    level_count = len(neck.lateral_convs)
    if level_count < 2 or len(neck.output_convs) != level_count:
        raise ValueError("V0.2 requires aligned multiscale lateral/output modules")
    if len(model.levels) != level_count:
        raise ValueError("level names do not match neck modules")
    stats = {}
    adds = {index: {"node_id": f"neck.fuse.{index}", "level": model.levels[index],
                    "left_max_abs": 0.0, "right_max_abs": 0.0, "output_max_abs": 0.0,
                    "observed_batches": 0}
            for index in range(level_count-1)}
    handles = []
    batches = sample_count = 0

    def hook_input(name):
        def capture(_module, inputs):
            value = float(inputs[0].detach().abs().max().item())
            if not isfinite(value):
                raise ValueError(f"nonfinite calibration input: {name}")
            record = stats.setdefault(name, {"input_max_abs":0.0, "output_max_abs":0.0})
            record["input_max_abs"] = max(record["input_max_abs"], value)
        return capture

    def hook_output(name):
        def capture(_module, _inputs, output):
            value = float(output.detach().abs().max().item())
            if not isfinite(value):
                raise ValueError(f"nonfinite calibration output: {name}")
            stats[name]["output_max_abs"] = max(stats[name]["output_max_abs"], value)
        return capture

    def observing_neck(features):
        lateral = [conv(x) for conv,x in zip(neck.lateral_convs, features)]
        fused = list(lateral)
        for index in range(level_count-2, -1, -1):
            right = F.interpolate(fused[index+1], size=fused[index].shape[-2:], mode="nearest")
            left = fused[index]
            result = left + right
            record = adds[index]
            for key, value in (("left_max_abs",left), ("right_max_abs",right), ("output_max_abs",result)):
                maximum = float(value.detach().abs().max().item())
                if not isfinite(maximum):
                    raise ValueError(f"nonfinite calibration Add: {record['node_id']}")
                record[key] = max(record[key], maximum)
            record["observed_batches"] += 1
            fused[index] = result
        return tuple(conv(value) for conv,value in zip(neck.output_convs,fused))

    had_instance_forward = "forward" in neck.__dict__
    instance_forward = neck.__dict__.get("forward")
    try:
        for name,module in model.named_modules():
            if isinstance(module,nn.Conv2d):
                handles.append(module.register_forward_pre_hook(hook_input(name)))
                handles.append(module.register_forward_hook(hook_output(name)))
        neck.forward = observing_neck
        for batch in images:
            if not isinstance(batch,torch.Tensor) or batch.ndim != 4:
                raise ValueError("calibration iterable must yield NCHW image tensors")
            if not batch.shape[0]:
                raise ValueError("empty calibration batch")
            model(batch.to(device))
            batches += 1
            sample_count += batch.shape[0]
    finally:
        for handle in handles:
            handle.remove()
        if had_instance_forward:
            neck.forward = instance_forward
        else:
            neck.__dict__.pop("forward",None)
    if not batches:
        raise ValueError("empty calibration iterable")
    if any(record["observed_batches"] != batches for record in adds.values()):
        raise ValueError("a fusion node was not observed in every calibration batch")
    return {"calibration_contract":CALIBRATION_CONTRACT, "convs":stats,
            "adds":[adds[index] for index in range(level_count-1)],
            "calibration_batches":batches, "calibration_images":sample_count}


def prepare_v0_2_ptq(model: nn.Module, calibration_images: Iterable[torch.Tensor], *, device: str):
    """Prepare independent folded/fake-quant models using the existing V1 numeric rules."""
    if model.training:
        raise ValueError("PTQ requires an eval-mode checkpoint")
    if not getattr(model,"levels",None):
        raise ValueError("missing V0.2 level contract")
    folded = deepcopy(model).to(device).eval()
    pairs = fold_batch_norms_inplace(folded)
    if any(isinstance(module,nn.BatchNorm2d) for module in folded.modules()):
        raise ValueError("unfolded BatchNorm remains")
    extrema = observe_v0_2_extrema(folded,calibration_images,device=device)
    quantized,profile = wrap_h2_ptq(deepcopy(folded),extrema)
    # The old quantizer's arithmetic is reusable once its input records are
    # explicitly accumulated and ordered by node identity.
    profile["calibration_contract"] = CALIBRATION_CONTRACT
    profile["calibration"] = {k:extrema[k] for k in ("calibration_batches","calibration_images")}
    profile["fpn_add_node_ids"] = [record["node_id"] for record in extrema["adds"]]
    for record,observation in zip(profile["fpn_adds"],extrema["adds"]):
        record.update(node_id=observation["node_id"], level=observation["level"],
                      observed_batches=observation["observed_batches"])
    profile["bn_folding"] = {"enabled":True,"folded_pair_count":len(pairs),"pairs":pairs}
    profile["hardware_cost_status"] = "V0_2_NEW_ENGINE_NEEDS_MEASUREMENT"
    profile["mixed_precision"]["hardware_cost_status"] = "V0_2_NEW_ENGINE_NEEDS_MEASUREMENT"
    profile["parity_ready"] = False
    profile["software_reference"] = INTEGER_REFERENCE
    profile["legacy_float32_fake_quant"] = "diagnostic_only_without_canonical_INT32_wrap"
    return folded,promote_exact_v0_2_convs(quantized),extrema,profile
