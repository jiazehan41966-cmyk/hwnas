"""V0.2 node-bound PTQ export into an exact integer-oracle package.

Compact OIHW parameters are intentionally distinct from the standalone HLS
operator's packed AXI128 layout. Neither package implements a graph scheduler.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from torch import nn

from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH
from hwnas_fpga.deploy.sonar_v0_2_ptq import CALIBRATION_CONTRACT, prepare_v0_2_ptq
from hwnas_fpga.deploy.sonar_v0_2_integer import INTEGER_REFERENCE, promote_exact_v0_2_convs
from hwnas_fpga.models.sonar_candidate_v0_2 import build_sonar_candidate_v0_2
from tools.detector_v0_2.layout import LAYOUT_CONTRACT, pack_weights, unpack_weights
from tools.detector_v0_2.provenance import verify_completed_training
from tools.fpga_h2.interface import align, descriptors
from tools.sonar_deploy.ptq_export import (
    _ArchivedPackageBuilder, _profile, restore_archived_ptq, IntegerGraphModel,
)

DEPLOYMENT = "sonar_v0_2_integer_oracle_compact_v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile_digest(profile) -> str:
    """File identity uses original bytes; mappings use canonical finite JSON."""
    if isinstance(profile, (str, Path)):
        return sha256(Path(profile).read_bytes())
    return sha256(json.dumps(dict(profile), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode())


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Stable named-tensor identity, including BN buffers and integer counters."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("state dict must contain named tensors only")
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
                                 separators=(",", ":")).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_tree(value):
    if isinstance(value, Mapping):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite PTQ profile value")


def validate_v0_2_profile(model, profile) -> dict:
    spec = _profile(profile)
    _finite_tree(spec)
    count = len(model.levels) - 1
    expected_ids = [f"neck.fuse.{i}" for i in range(count)]
    if spec.get("calibration_contract") != CALIBRATION_CONTRACT:
        raise ValueError("V0.2 requires node-bound calibration")
    if spec.get("fpn_add_node_ids") != expected_ids:
        raise ValueError("FPN node identities/order differ from V0.2")
    adds = spec.get("fpn_adds", [])
    if len(adds) != count:
        raise ValueError("V0.2 requires exactly one accumulated record per FPN Add")
    calibration = spec.get("calibration", {})
    batches, images = calibration.get("calibration_batches"), calibration.get("calibration_images")
    if type(batches) is not int or batches < 1 or type(images) is not int or images < batches:
        raise ValueError("invalid calibration batch/image count")
    for index, record in enumerate(adds):
        if (record.get("node_id") != expected_ids[index] or
                record.get("level") != model.levels[index] or
                type(record.get("observed_batches")) is not int or
                record.get("observed_batches") != batches):
            raise ValueError("FPN observation identity/coverage differs from V0.2")
    # Frozen V1 validation and strict module/operator binding remain delegated
    # to restore_archived_ptq. Its historical observer is never invoked here.
    return spec


def restore_v0_2_ptq(model, profile):
    spec = validate_v0_2_profile(model, profile)
    folded, quantized = restore_archived_ptq(model, spec)
    return folded, promote_exact_v0_2_convs(quantized)


class V02PackageBuilder(_ArchivedPackageBuilder):
    def conv_sequence(self, prefix, source):
        module = self.model.get_submodule(prefix)
        paths = [prefix + ("." + name if name else "")
                 for name, child in module.named_modules() if isinstance(child, nn.Conv2d)]
        for path in paths:
            source = self.layer(path, source)
        return source

    def fpn_add(self, name, left, right, index):
        source = super().fpn_add(name, left, right, index)
        node = self.nodes[-1]
        node.pop("archived_fpn_record_index")
        node["calibration_node_id"] = self.profile["fpn_adds"][index]["node_id"]
        node["calibration_record_index"] = index
        return source

    def build(self):
        if self.model.backbone.post_stem_downsample is not None:
            raise ValueError("post-stem pooling is unsupported")
        if len(self.model.backbone.stages) != self.model.last_active_stage + 1:
            raise ValueError("inactive backbone tail was not physically omitted")
        source = self.conv_sequence("backbone.stem", "image")
        features = {}
        for index, blocks in enumerate(self.model.backbone.stages):
            if len(blocks) != 1:
                raise ValueError("V0.2 supports one block per stage")
            block = blocks[0]
            if getattr(block, "use_residual", False):
                raise ValueError("unrepresented backbone residual")
            source = self.conv_sequence(f"backbone.stages.{index}.0", source)
            if index in self.model.feature_indices:
                features[index] = source
        fused = [self.conv_sequence(f"neck.lateral_convs.{j}", features[i])
                 for j, i in enumerate(self.model.feature_indices)]
        for index in range(len(fused) - 2, -1, -1):
            fused[index] = self.fpn_add(f"neck.fuse.{index}", fused[index], fused[index + 1], index)
        outputs = {}
        for index, level in enumerate(self.model.levels):
            source = self.conv_sequence(f"neck.output_convs.{index}", fused[index])
            outputs[level] = self.conv_sequence(f"head.towers.{index}", source)
        if {node["id"] for node in self.nodes if node["op"] == "conv"} != set(self.layers):
            raise ValueError("graph failed to consume all V0.2 convolution records")
        return outputs


def _bind_provenance(model, profile, provenance):
    bound = deepcopy(dict(provenance))
    if model.training:
        raise ValueError("export requires an eval-mode model")
    if bound.get("test_accessed", False) or bound.get("test_evaluated", False):
        raise ValueError("test access is outside the V0.2 protocol")
    if bound.get("profile_sha256") != profile_digest(profile):
        raise ValueError("profile SHA256 binding mismatch")
    prepared_device = next(model.parameters()).device.type
    if str(bound.get("parameter_preparation_device", "")).split(":")[0] != prepared_device:
        raise ValueError("parameter preparation backend is not bound")
    purpose = bound.get("purpose")
    identity = state_digest(model.state_dict())
    if purpose == "untrained_contract_smoke":
        if bound.get("checkpoint_sha256") or bound.get("checkpoint_path"):
            raise ValueError("untrained smoke must not claim a checkpoint binding")
        real = False
    elif purpose == "trained_checkpoint":
        path = Path(bound.get("checkpoint_path", ""))
        if not path.is_file() or sha256(path.read_bytes()) != bound.get("checkpoint_sha256"):
            raise ValueError("actual checkpoint SHA256 binding mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        key = bound.get("state_dict_key", "model_state_dict")
        if not isinstance(saved, Mapping) or key not in saved or state_digest(saved[key]) != identity:
            raise ValueError("checkpoint state differs from the model being exported")
        if saved.get("mode") == "smoke" or saved.get("test_accessed") or saved.get("test_evaluated"):
            raise ValueError("non-formal/test-accessing checkpoint cannot bind trained deployment")
        training_binding = saved.get("binding", {})
        if (training_binding.get("formal_eligible") is not True or
                training_binding.get("namespace") != "sonar_candidate_v0_2" or
                training_binding.get("prototype") != model.prototype_id or
                training_binding.get("epochs") != 80 or
                training_binding.get("test_evaluated") is not False or
                saved.get("architecture") != model.candidate_architecture or
                type(saved.get("epoch")) is not int or not 1 <= saved["epoch"] <= 80):
            raise ValueError("checkpoint lacks a matching formal V0.2 training binding")
        bound["training_binding"] = deepcopy(training_binding)
        bound["checkpoint_epoch"] = saved["epoch"]
        bound.update(verify_completed_training(saved,path,bound))
        real = True
    else:
        raise ValueError("purpose must distinguish untrained smoke from trained checkpoint")
    bound.update(model_state_sha256=identity, parameter_preparation_device=prepared_device,
                 test_accessed=False, test_evaluated=False)
    return bound, real


def export_ptq_package(model, profile, output, provenance):
    """Write a new package once; does not deploy or invoke calibration itself."""
    bound, real = _bind_provenance(model, profile, provenance)
    spec = validate_v0_2_profile(model, profile)
    folded, quantized = restore_v0_2_ptq(model, spec)
    builder = V02PackageBuilder(folded, quantized, spec)
    outputs = builder.build()
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "weights.bin").write_bytes(builder.weights)
    (folder / "biases.bin").write_bytes(builder.biases)
    np.asarray(builder.records, dtype="<i4").tofile(folder / "layers.bin")
    packed = bytearray()
    packed_records = []
    for node in builder.nodes:
        if node["op"] != "conv":
            continue
        weight = quantized.get_submodule(node["id"]).int_weight.detach().cpu().numpy().astype("i1")
        payload = pack_weights(weight, groups=node["groups"])
        if not np.array_equal(unpack_weights(payload, tuple(weight.shape), groups=node["groups"]), weight):
            raise ValueError("standalone operator weight packing failed roundtrip")
        offset = align(len(packed))
        packed.extend(bytes(offset - len(packed)))
        packed.extend(payload)
        packed_records.append(dict(node_id=node["id"], offset=offset, bytes=len(payload),
                                   shape=list(weight.shape), groups=node["groups"], sha256=sha256(payload)))
    (folder / "operator_weights_axi128.bin").write_bytes(packed)
    graph = dict(
        version=1, deployment_version=DEPLOYMENT,
        package_scope="INTEGER_ORACLE_COMPACT_PACKAGE", hardware_layout_binding="NOT_IMPLEMENTED",
        real_weights_bound=real, synthetic_weights=not real,
        architecture=model.candidate_architecture, inputs={"image": builder.edges["image"]},
        nodes=builder.nodes, edges=builder.edges, outputs=outputs,
        descriptors=[descriptor.__dict__ for descriptor in descriptors(
            model.levels, dict(zip(model.levels, model.anchor_sizes)))],
        numeric_contract="NUMERIC_CONTRACT_V1", numeric_contract_hash=EXPECTED_HASH,
        calibration_contract=CALIBRATION_CONTRACT,
        fpn_calibration=dict(policy="one_accumulated_record_per_forward_node",
                             node_ids=spec["fpn_add_node_ids"],
                             calibration_batches=spec["calibration"]["calibration_batches"],
                             saved_record_count=len(spec["fpn_adds"]), unconsumed_records=0),
        output_storage_contract=dict(layout="HWC_channel0_low_byte", frame_alignment=32,
                                     ddr_alignment=64, scope="host_storage_only_no_DMA_implementation"),
        runtime_interface=dict(status="NOT_IMPLEMENTED", target_ddr_bits=128,
                               target_schedule="whole_graph_single_start", board_parity=False),
        standalone_operator_parameters=dict(layout_contract=LAYOUT_CONTRACT,
            filename="operator_weights_axi128.bin", layers=packed_records,
            scope="weight_packing_only_not_full_graph_runtime"),
        layers_record="32 little-endian int32 words; canonical compact integer-oracle conv loading",
        provenance=bound, checkpoint_sha256=bound.get("checkpoint_sha256"),
        profile_sha256=bound["profile_sha256"], test_accessed=False, test_evaluated=False,
        binary_hashes={name: sha256((folder / name).read_bytes()) for name in
                      ("weights.bin", "biases.bin", "layers.bin", "operator_weights_axi128.bin")},
        arithmetic_reference="IntegerGraphModel_canonical_INT32_wrap_nearest_even",
        software_reference=INTEGER_REFERENCE,
        float_fakequant_reference_status="approximate_float32_no_INT32_wrap_not_a_parity_reference",
    )
    encoded = json.dumps(graph, indent=2, sort_keys=True, allow_nan=False).encode()
    (folder / "graph.json").write_bytes(encoded)
    manifest = dict(version=1, graph_sha256=sha256(encoded), binary_hashes=graph["binary_hashes"],
                    provenance=bound, real_weights_bound=real, synthetic_weights=not real,
                    hardware_layout_binding="NOT_IMPLEMENTED", test_accessed=False, test_evaluated=False)
    (folder / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    # This validates JSON/binary layer descriptors and parameter hashes without
    # claiming integer forward or hardware parity.
    IntegerGraphModel(folder, model)
    return graph


def smoke(output: Path, prototypes=("R1", "R8"), *, integer=True):
    """Synthetic-only contract replay; never opens dataset or trained checkpoint."""
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    results = []
    for prototype in prototypes:
        torch.manual_seed(42)
        model = build_sonar_candidate_v0_2(architecture=prototype).eval()
        generator = torch.Generator().manual_seed(20261002)
        images = [torch.rand(1, 1, 640, 640, generator=generator),
                  torch.rand(1, 1, 640, 640, generator=generator) * .5]
        started = time.monotonic()
        folded, quantized, extrema, profile = prepare_v0_2_ptq(model, images, device="cpu")
        profile_path = output / f"{prototype}_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        folder = output / f"{prototype}_package"
        graph = export_ptq_package(model, profile_path, folder, dict(
            purpose="untrained_contract_smoke", profile_sha256=profile_digest(profile_path),
            parameter_preparation_device="cpu", seed=42, calibration_source="synthetic_uniform_two_batches"))
        restored_folded, restored = restore_v0_2_ptq(model, profile_path)
        with torch.no_grad():
            predicted = quantized(images[0])["predictions"]
            replayed = restored(images[0])["predictions"]
        if not all(torch.equal(a, b) for a, b in zip(predicted, replayed)):
            raise ValueError("saved profile canonical reference replay differs")
        record = dict(prototype=prototype, purpose="untrained_contract_smoke",
            real_weights_bound=False, synthetic_weights=True, calibration_batches=2,
            convolution_count=len(profile["layers"]), fpn_node_ids=profile["fpn_add_node_ids"],
            prediction_shapes=[list(value.shape) for value in predicted],
            all_predictions_finite=all(bool(torch.isfinite(value).all()) for value in predicted),
            canonical_profile_replay_bitwise_equal=True, software_reference=INTEGER_REFERENCE,
            integer_forward="NOT_RUN",
            hardware_parity="NOT_RUN", hardware_layout_binding="NOT_IMPLEMENTED")
        if integer:
            oracle = IntegerGraphModel(folder, model)
            with torch.no_grad():
                exact = oracle(images[0])["predictions"]
            comparisons = []
            for level, canonical, value in zip(model.levels, predicted, exact):
                diff = (canonical - value).abs()
                exp = graph["edges"][graph["outputs"][level]]["exp"]
                comparisons.append(dict(level=level, shape=list(value.shape),
                    finite=bool(torch.isfinite(value).all()), different_elements=int((diff != 0).sum()),
                    total_elements=value.numel(), max_abs_difference=float(diff.max()),
                    max_difference_output_LSBs=float(diff.max()) / (2. ** exp)))
            record.update(integer_forward="PASS_SHAPES_FINITE", integer_vs_canonical=comparisons,
                          integer_vs_canonical_equal=all(item["different_elements"] == 0 for item in comparisons))
        record["elapsed_seconds"] = time.monotonic() - started
        results.append(record)
        (output / "SMOKE_RESULTS.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if not record["all_predictions_finite"] or (integer and not record["integer_vs_canonical_equal"]):
            raise ValueError("canonical software references disagree or contain nonfinite predictions")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototypes", nargs="+", choices=("R0", "R1", "R7", "R8"), default=["R1", "R8"])
    parser.add_argument("--no-integer", action="store_true")
    args = parser.parse_args()
    smoke(args.output, args.prototypes, integer=not args.no_integer)
