"""Find the first float32 fake-quant / canonical integer conv divergence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from torch.nn import functional as F
from hwnas_fpga.deploy.h2_ptq import quantize_h2
from hwnas_fpga.models.sonar_candidate_v0_2 import build_sonar_candidate_v0_2
from tools.detector_v0_2.ptq_export import validate_v0_2_profile
from tools.sonar_deploy.ptq_export import (
    IntegerGraphModel, requant_torch, restore_archived_ptq, wrap32_torch,
)


def diagnose(folder: Path, prototype: str):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_sonar_candidate_v0_2(architecture=prototype).eval()
    profile_path = folder / f"{prototype}_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    # This diagnostic deliberately isolates the old approximate float32
    # wrapper. V0.2 deployment uses its own canonical integer reference.
    folded, fake = restore_archived_ptq(model, validate_v0_2_profile(model, profile))
    image = torch.rand(1, 1, 640, 640, generator=torch.Generator().manual_seed(20261002))
    oracle = IntegerGraphModel(folder / f"{prototype}_package", model)
    _, exact_values = oracle.forward_quantized(image, retain=True)
    nodes = {node["id"]: node for node in oracle.graph["nodes"] if node["op"] == "conv"}
    first = []
    differing = []
    hooks = []

    def hook(name):
        def capture(module, inputs, output):
            node = nodes[name]
            exact = exact_values[name]
            fake_int = torch.round(output / (2. ** node["output_exp"])).to(torch.int64)
            diff = fake_int - exact.to(torch.int64)
            if not bool((diff != 0).any()):
                return
            differing.append(dict(node_id=name, differing_elements=int((diff != 0).sum()),
                                  max_abs_output_LSB_difference=int(diff.abs().max())))
            if first:
                return
            a = exact_values[node["inputs"][0]]
            qi = quantize_h2(inputs[0], module.input_exp)
            weight, bias = module.int_weight, module.int_bias
            ci, co, kh, kw = weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3]
            patches = F.unfold(a.to(torch.float64), (kh, kw), padding=module.padding, stride=module.stride)
            if module.groups == 1:
                raw = torch.matmul(weight.to(torch.float64).reshape(co, -1), patches)
            else:
                raw = (patches.reshape(1, co, kh * kw, -1) *
                       weight.to(torch.float64).reshape(1, co, kh * kw, 1)).sum(dim=2)
            raw = raw.reshape(exact.shape).to(torch.int64) + bias.to(torch.int64).view(1, -1, 1, 1)
            approximate = F.conv2d(qi, weight.to(torch.float32), bias.to(torch.float32),
                stride=module.stride, padding=module.padding, groups=module.groups)
            quantized_bias_unclipped = torch.round(folded.get_submodule(name).bias.to(torch.float64) /
                                                   (2. ** (module.input_exp + module.weight_exps[0])))
            overflow = (raw < -(1 << 31)) | (raw > (1 << 31) - 1)
            precision = approximate.to(torch.float64) - raw.to(torch.float64)
            wrapped = wrap32_torch(raw)
            replay = requant_torch(wrapped, node["multiplier"], node["shift"])
            if node["activation"] == "relu":
                replay = replay.clamp_min(0)
            elif node["activation"] == "relu6":
                replay = replay.clamp(0, node["q6"])
            coordinate = torch.nonzero(diff != 0)[0].tolist()
            key = tuple(coordinate)
            first.append(dict(node_id=name, input_exp=node["input_exp"], output_exp=node["output_exp"],
                weight_exp=node["weight_exp"], activation=node["activation"],
                actual_fake_conv_integer_input_matches_oracle=bool(torch.equal(qi.to(torch.int8), a)),
                integer_bias_min=int(bias.min()), integer_bias_max=int(bias.max()),
                unclipped_bias_outside_INT32_count=int(((quantized_bias_unclipped < -(1 << 31)) |
                                                      (quantized_bias_unclipped > (1 << 31) - 1)).sum()),
                float32_cast_changes_bias_values=int((bias.to(torch.float32).to(torch.int64) != bias.to(torch.int64)).sum()),
                raw_integer_accumulator_min=int(raw.min()), raw_integer_accumulator_max=int(raw.max()),
                raw_accumulator_outside_INT32_count=int(overflow.sum()),
                float32_accumulator_inexact_elements=int((precision != 0).sum()),
                max_abs_float32_accumulator_error=float(precision.abs().max()),
                exact_requant_replay_matches_oracle=bool(torch.equal(replay, exact)),
                first_different_coordinate=coordinate,
                first_difference=dict(integer_input_value=int(a.flatten()[0]),
                    raw_int64_accumulator=int(raw[key]), wrapped_INT32_accumulator=int(wrapped[key]),
                    fake_float32_accumulator=float(approximate[key]),
                    fake_output_int8=int(fake_int[key]), exact_output_int8=int(exact[key]))))
        return capture

    try:
        for name in nodes:
            hooks.append(fake.get_submodule(name).register_forward_hook(hook(name)))
        with torch.no_grad():
            fake(image)
    finally:
        for handle in hooks:
            handle.remove()
    result = dict(prototype=prototype, seed=42, purpose="untrained_contract_smoke",
                  first_divergent_conv=first[0] if first else None, differing_convolutions=differing,
                  scope="float32_fakequant_vs_exact_numeric_reference_not_hardware_parity")
    (folder / f"{prototype}_INTEGER_DIAGNOSIS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--prototypes", nargs="+", default=["R1", "R8"])
    args = parser.parse_args()
    for prototype in args.prototypes:
        diagnose(args.folder, prototype)
