"""Export archived W8A8 PTQ into the canonical H2 binary graph contract.

No calibration is performed. The historical FPN wrapper indexes its appended
observations by forward level index; that behavior is deliberately replayed.
Float fake-quant and the exact INT32 implementation are separate references.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.deploy.h2_ptq import (
    H2FakeQuantConv, H2QuantizedFPN, _replace_fused_activations,
    conv_activation_map, ratio_from_exponents,
)
from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from hwnas_fpga.deploy.ptq_eval import fold_batch_norms_inplace
from tools.fpga_h2 import numeric
from tools.fpga_h2.interface import CONTRACT as INTERFACE_CONTRACT, align, descriptors

ACT = {"linear": 0, "relu": 1, "relu6": 2}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _profile(value: Mapping[str, Any] | str | Path) -> dict:
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return deepcopy(dict(value))


def _validate_profile(profile: Mapping[str, Any]) -> None:
    load_numeric_contract_v1()
    expected = {
        "numeric_contract": "NUMERIC_CONTRACT_V1", "numeric_contract_hash": EXPECTED_HASH,
        "zero_point": 0, "rounding": "nearest_even", "saturation": [-128, 127],
        "weight_granularity": "per_tensor", "activation_granularity": "per_tensor",
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise ValueError(f"archived profile {key} differs from frozen uniform INT8 contract")
    if profile.get("bitwidth") != {"activation": 8, "weight": 8, "accumulator": 32, "requant_product": 64}:
        raise ValueError("only uniform W8A8 is admitted")
    bypass = profile.get("fp32_bypass", {})
    for key in ("module_prefixes", "conv_layers", "weight_prefixes", "weight_fp32_layers", "activation_prefixes", "activation_fp32_layers"):
        if bypass.get(key):
            raise ValueError("FP32 bypass cannot be exported as uniform INT8")
    if bypass.get("neck_adds_quantized") is False:
        raise ValueError("unquantized FPN adds are not admitted")
    names = [r["name"] for r in profile["layers"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate layer names in archived profile")
    for record in profile["layers"]:
        if any(record.get(k, 8) != 8 for k in ("input_bits", "weight_bits", "output_bits")):
            raise ValueError("mixed-precision layer is not admitted")
        if record.get("fp32_weight") or record.get("fp32_activation"):
            raise ValueError("floating-point diagnostic layer is not admitted")
        if len(record["weight_exps"]) != 1:
            raise ValueError("per-channel weight profile is not admitted")
        for exp in [record["input_exp"], record["output_exp"], *record["weight_exps"]]:
            if type(exp) is not int or not -24 <= exp <= 16:
                raise ValueError("scale exponent outside V1")
        pair = ratio_from_exponents(record["input_exp"] + record["weight_exps"][0], record["output_exp"])
        if (record["multiplier"], record["shift"]) != pair:
            raise ValueError(f"saved multiplier/shift disagrees with exponents: {record['name']}")


def restore_archived_ptq(model: nn.Module, profile: Mapping[str, Any] | str | Path):
    """Return independent (folded FP32, historical float fake-quant) copies.

    All layer scales are read, never observed or reselected. Model device is
    preserved: Phase5A used CUDA BN folding; CPU can differ by an integer LSB.
    The caller must restore the archived preparation backend. H2FakeQuantConv
    independently checks the archived weight exponent against the folded weight.
    """
    spec = _profile(profile)
    _validate_profile(spec)
    folded = deepcopy(model).eval()
    pairs = fold_batch_norms_inplace(folded)
    saved_folding = spec.get("bn_folding", {})
    if saved_folding.get("pairs") is not None and saved_folding["pairs"] != pairs:
        raise ValueError("BN folding paths differ from archived profile")
    convs = {n: m for n, m in folded.named_modules() if isinstance(m, nn.Conv2d)}
    records = {r["name"]: r for r in spec["layers"]}
    if set(convs) != set(records):
        raise ValueError(f"profile/model layer mismatch: missing={sorted(set(convs)-set(records))}; extra={sorted(set(records)-set(convs))}")
    acts = conv_activation_map(folded)
    quantized = deepcopy(folded)
    for name, conv in convs.items():
        record = records[name]
        if acts[name] != record["activation"] or list(conv.kernel_size) != record["kernel"] or conv.groups != record["groups"]:
            raise ValueError(f"profile/model operator mismatch: {name}")
        wrapper = H2FakeQuantConv(
            quantized.get_submodule(name), input_exp=record["input_exp"],
            output_exp=record["output_exp"], weight_exps=record["weight_exps"],
            activation=record["activation"], weight_granularity="per_tensor",
        )
        parent_name, _, leaf = name.rpartition(".")
        setattr(quantized.get_submodule(parent_name), leaf, wrapper)
    _replace_fused_activations(quantized)
    num_adds = len(quantized.neck.lateral_convs) - 1
    if len(spec["fpn_adds"]) < num_adds:
        raise ValueError("missing archived FPN records")
    for item in spec["fpn_adds"]:
        if any(type(item[k]) is not int or not -24 <= item[k] <= 16 for k in ("left_exp", "right_exp", "output_exp")):
            raise ValueError("FPN exponent outside V1")
    quantized.neck = H2QuantizedFPN(quantized.neck, deepcopy(spec["fpn_adds"]))
    return folded, quantized.eval()


class _ArchivedPackageBuilder:
    def __init__(self, folded, quantized, profile):
        self.model, self.quantized, self.profile = folded, quantized, profile
        if getattr(folded, "image_size", 640) != 640 or folded.input_channels != 1:
            raise ValueError("real deployment contract requires grayscale input640")
        self.records, self.nodes = [], []
        self.weights, self.biases = bytearray(), bytearray()
        self.layers = {r["name"]: r for r in profile["layers"]}
        self.edges = {"image": {"shape": [1, 640, 640], "exp": self.layers["backbone.stem.conv.conv"]["input_exp"]}}

    def bridge(self, source, exponent, node_id):
        old = self.edges[source]["exp"]
        if old == exponent:
            return source
        mult, shift = numeric.ratio(old, exponent)
        self.nodes.append(dict(id=node_id, op="requant", inputs=[source], input_exp=old,
                               output_exp=exponent, multiplier=mult, shift=shift,
                               activation="linear", q6=numeric.q6(exponent)))
        self.edges[node_id] = dict(shape=list(self.edges[source]["shape"]), exp=exponent)
        return node_id

    def layer(self, path, source):
        spec, conv = self.layers[path], self.model.get_submodule(path)
        module = self.quantized.get_submodule(path)
        source = self.bridge(source, spec["input_exp"], path + ".input_requant")
        ci, h, w = self.edges[source]["shape"]
        if ci != conv.in_channels or conv.dilation != (1, 1) or conv.groups not in (1, ci):
            raise ValueError(f"unsupported convolution geometry: {path}")
        if conv.groups != 1 and conv.out_channels != ci:
            raise ValueError("depthwise channel multiplier is not supported")
        if any(pair[0] != pair[1] for pair in (conv.kernel_size, conv.stride, conv.padding)):
            raise ValueError("asymmetric convolution is not supported")
        wq = module.int_weight.detach().cpu().numpy().astype("i1")
        bq = module.int_bias.detach().cpu().numpy().astype("<i4")
        wo, bo = align(len(self.weights)), align(len(self.biases))
        self.weights.extend(bytes(wo - len(self.weights))); self.biases.extend(bytes(bo - len(self.biases)))
        self.weights.extend(wq.tobytes()); self.biases.extend(bq.tobytes())
        k, stride, pad = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        node = dict(id=path, op="conv", inputs=[source], Cin=ci, Cout=conv.out_channels,
                    H=h, W=w, kernel=k, stride=stride, padding=pad, groups=conv.groups,
                    input_exp=spec["input_exp"], weight_exp=spec["weight_exps"][0],
                    output_exp=spec["output_exp"], multiplier=spec["multiplier"], shift=spec["shift"],
                    activation=spec["activation"], q6=numeric.q6(spec["output_exp"]),
                    weight_offset=wo, bias_offset=bo, weight_shape=list(wq.shape), bias_count=conv.out_channels,
                    folded_weight_hash=sha256(conv.weight.detach().cpu().numpy().tobytes()),
                    folded_bias_hash=sha256(conv.bias.detach().cpu().numpy().tobytes()),
                    weight_hash=sha256(wq.tobytes()), bias_hash=sha256(bq.tobytes()), layer_id=len(self.records))
        self.records.append([node["layer_id"], ci, conv.out_channels, h, w, k, stride, pad, conv.groups,
                             node["input_exp"], node["weight_exp"], node["output_exp"], node["multiplier"],
                             node["shift"], ACT[node["activation"]], node["q6"], wo, bo, wq.size,
                             conv.out_channels, 1] + [0] * 11)
        self.nodes.append(node)
        self.edges[path] = dict(shape=[conv.out_channels, (h + 2 * pad - k) // stride + 1,
                                      (w + 2 * pad - k) // stride + 1], exp=spec["output_exp"])
        return path

    def fpn_add(self, name, left, right, index):
        # Preserve H2QuantizedFPN.forward's indexing, including historical reversal.
        spec = self.profile["fpn_adds"][index]
        left = self.bridge(left, spec["left_exp"], name + ".left_requant")
        right = self.bridge(right, spec["right_exp"], name + ".right_requant")
        ea, eb, eo = spec["left_exp"], spec["right_exp"], spec["output_exp"]
        common = max(ea, eb)
        a, b = self.edges[left]["shape"], self.edges[right]["shape"]
        if a[0] != b[0] or a[1:] != [2 * b[1], 2 * b[2]]:
            raise ValueError("FPN requires channel-equal nearest factor2")
        self.nodes.append(dict(id=name, op="upsample_add", inputs=[left, right],
                               input_exp_a=ea, input_exp_b=eb, common_exp=common, output_exp=eo,
                               numeric=list(numeric.ratio(ea, common) + numeric.ratio(eb, common) + numeric.ratio(common, eo)),
                               archived_fpn_record_index=index))
        self.edges[name] = dict(shape=list(a), exp=eo)
        return name

    def build(self):
        if self.model.backbone.post_stem_downsample is not None:
            raise ValueError("unsupported post-stem pooling")
        x = self.layer("backbone.stem.conv.conv", "image")
        x = self.layer("backbone.stages.0.0.conv", x)
        features = {}
        for i, blocks in enumerate(self.model.backbone.stages):
            if i == 0:
                continue
            if len(blocks) != 1:
                raise ValueError("frozen V0.1 requires one block per stage")
            if i == 3:
                if any(True for _ in blocks[0].parameters()):
                    raise ValueError("stage3 skip must be parameter-free")
            else:
                block, base = blocks[0], f"backbone.stages.{i}.0"
                if block.use_residual or not block.use_expand:
                    raise ValueError("unexpected residual/non-expanded path outside B1/C/D contract")
                for kind in ("expand_conv", "dw_conv", "project_conv"):
                    x = self.layer(base + "." + kind, x)
            if i in self.model.feature_indices:
                features[i] = x
        fused = [self.layer(f"neck.lateral_convs.{j}.0", features[i]) for j, i in enumerate(self.model.feature_indices)]
        for j in range(len(fused) - 2, -1, -1):
            fused[j] = self.fpn_add(f"neck.fuse.{j}", fused[j], fused[j + 1], j)
        neck = [self.layer(f"neck.output_convs.{j}.0", x) for j, x in enumerate(fused)]
        outputs = {}
        for j, (level, x) in enumerate(zip(self.model.levels, neck)):
            x = self.layer(f"head.towers.{j}.0", x)
            outputs[level] = self.layer(f"head.towers.{j}.3", x)
        emitted = {n["id"] for n in self.nodes if n["op"] == "conv"}
        if emitted != set(self.layers):
            raise ValueError("graph failed to consume all archived conv records")
        return outputs


def export_ptq_package(model, profile, output, provenance):
    """Export exact archived integer weights/scales; caller binds checkpoint identity."""
    provenance = deepcopy(dict(provenance))
    claimed_device = provenance.get("parameter_preparation_device")
    if claimed_device is not None and str(claimed_device).split(":")[0] != next(model.parameters()).device.type:
        raise ValueError("parameter preparation device disagrees with provenance; BN folding backend matters")
    if provenance.get("test_evaluated", False) or provenance.get("test_accessed", False):
        raise ValueError("test access is outside this deployment protocol")
    for name in ("checkpoint_sha256", "profile_sha256"):
        val = provenance.get(name, "")
        if len(val) != 64 or any(c not in "0123456789abcdef" for c in val):
            raise ValueError(f"missing SHA256 provenance: {name}")
    if isinstance(profile, (str, Path)) and sha256(Path(profile).read_bytes()) != provenance["profile_sha256"]:
        raise ValueError("profile file SHA mismatch")
    spec = _profile(profile)
    folded, quantized = restore_archived_ptq(model, spec)
    builder = _ArchivedPackageBuilder(folded, quantized, spec)
    outputs = builder.build()
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "weights.bin").write_bytes(builder.weights)
    (folder / "biases.bin").write_bytes(builder.biases)
    np.asarray(builder.records, dtype="<i4").tofile(folder / "layers.bin")
    count = len(model.levels) - 1
    graph = dict(
        version=1, deployment_version="real_ptq_v1", synthetic_weights=False, real_weights_bound=True,
        architecture=model.candidate_architecture, inputs={"image": builder.edges["image"]},
        nodes=builder.nodes, edges=builder.edges, outputs=outputs,
        descriptors=[d.__dict__ for d in descriptors(model.levels, dict(zip(model.levels, model.anchor_sizes)))],
        numeric_contract="NUMERIC_CONTRACT_V1", numeric_contract_hash=EXPECTED_HASH,
        interface_contract_hash=sha256(json.dumps(INTERFACE_CONTRACT, sort_keys=True, separators=(",", ":")).encode()),
        layers_record="32 little-endian int32 words; same canonical H2 conv loading contract",
        provenance=provenance, checkpoint_sha256=provenance["checkpoint_sha256"], profile_sha256=provenance["profile_sha256"],
        test_evaluated=False,
        binary_hashes={f: sha256((folder / f).read_bytes()) for f in ("weights.bin", "biases.bin", "layers.bin")},
        fpn_history=dict(policy="replay_archived_wrapper_index_without_recalibration", saved_record_count=len(spec["fpn_adds"]),
                         used_record_indices=list(range(count)), unconsumed_records=len(spec["fpn_adds"]) - count,
                         note="Observer appended reverse-level records per batch; wrapper reads record[index]. No correction in this version."),
    )
    encoded = json.dumps(graph, indent=2, sort_keys=True).encode()
    (folder / "graph.json").write_bytes(encoded)
    manifest = dict(version=1, graph_sha256=sha256(encoded), binary_hashes=graph["binary_hashes"],
                    provenance=provenance, real_weights_bound=True, synthetic_weights=False, test_evaluated=False)
    (folder / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return graph


def wrap32_torch(values):
    values = values.to(torch.int64)
    return torch.remainder(values + (1 << 31), 1 << 32) - (1 << 31)


def requant_torch(values, multiplier, shift):
    """INT32 -> INT64 multiply -> exact nearest-even -> INT8 saturation."""
    if not 1 <= int(multiplier) <= (1 << 31) - 1 or not 0 <= int(shift) <= 31:
        raise ValueError("multiplier/shift outside V1")
    a = values.to(torch.int64) * int(multiplier)
    magnitude = a.abs()
    q = magnitude >> int(shift)
    if shift:
        rem, half = magnitude & ((1 << int(shift)) - 1), 1 << (int(shift) - 1)
        q += ((rem > half) | ((rem == half) & ((q & 1) == 1))).to(torch.int64)
    return torch.where(a < 0, -q, q).clamp(-128, 127).to(torch.int8)


def exact_conv_torch(x, weight, bias, node):
    """Exact integer dot products via FP64, followed by canonical INT32 wrap.

    Every product and partial sum is integer and bounded strictly below 2**53;
    binary64 therefore represents all terms exactly. Bias is added in INT64.
    This is an accelerated integer oracle, not float32 fake quantization.
    """
    terms = int(np.prod(weight.shape[1:]))
    if terms * 128 * 128 >= 1 << 53:
        raise ValueError("FP64 exact-integer accumulation bound exceeded")
    # Explicit im2col + dot products avoids backend FFT/Winograd transforms,
    # which would invalidate the integer-representability argument even in FP64.
    batch, ci, h, width = x.shape
    co, _, kh, kw = weight.shape
    stride, padding = node["stride"], node["padding"]
    oh, ow = (h + 2 * padding - kh) // stride + 1, (width + 2 * padding - kw) // stride + 1
    patches = F.unfold(x.to(torch.float64), (kh, kw), padding=padding, stride=stride)
    w64 = weight.to(torch.float64)
    if node["groups"] == 1:
        acc = torch.matmul(w64.reshape(co, -1), patches)
    elif node["groups"] == ci == co:
        acc = (patches.reshape(batch, ci, kh * kw, -1) * w64.reshape(1, ci, kh * kw, 1)).sum(dim=2)
    else:
        raise ValueError("V1 supports only dense or depthwise conv")
    acc = acc.reshape(batch, co, oh, ow)
    acc = wrap32_torch(acc.to(torch.int64) + bias.to(torch.int64).view(1, -1, 1, 1))
    y = requant_torch(acc, node["multiplier"], node["shift"])
    if node["activation"] == "relu":
        y = y.clamp_min(0)
    elif node["activation"] == "relu6":
        y = y.clamp(0, node["q6"])
    elif node["activation"] != "linear":
        raise ValueError("invalid activation")
    return y


def _load_package(folder):
    folder = Path(folder)
    graph_raw = (folder / "graph.json").read_bytes()
    graph = json.loads(graph_raw)
    manifest = json.loads((folder / "export_manifest.json").read_bytes())
    if sha256(graph_raw) != manifest["graph_sha256"]:
        raise ValueError("graph SHA mismatch")
    for filename, digest in graph["binary_hashes"].items():
        if sha256((folder / filename).read_bytes()) != digest:
            raise ValueError(f"parameter package SHA mismatch: {filename}")
    if graph["binary_hashes"] != manifest["binary_hashes"]:
        raise ValueError("manifest binary binding mismatch")
    wb, bb = (folder / "weights.bin").read_bytes(), (folder / "biases.bin").read_bytes()
    records = np.fromfile(folder / "layers.bin", dtype="<i4").reshape(-1, 32)
    params = {}
    for n in graph["nodes"]:
        if n["op"] != "conv":
            continue
        expected = [n["layer_id"], n["Cin"], n["Cout"], n["H"], n["W"], n["kernel"], n["stride"], n["padding"],
                    n["groups"], n["input_exp"], n["weight_exp"], n["output_exp"], n["multiplier"], n["shift"],
                    ACT[n["activation"]], n["q6"], n["weight_offset"], n["bias_offset"], int(np.prod(n["weight_shape"])), n["bias_count"], 1] + [0] * 11
        if records[n["layer_id"]].tolist() != expected:
            raise ValueError("binary/JSON conv descriptor mismatch")
        w = np.frombuffer(wb, dtype="i1", count=int(np.prod(n["weight_shape"])), offset=n["weight_offset"]).reshape(n["weight_shape"]).copy()
        b = np.frombuffer(bb, dtype="<i4", count=n["bias_count"], offset=n["bias_offset"]).copy()
        if sha256(w.tobytes()) != n["weight_hash"] or sha256(b.tobytes()) != n["bias_hash"]:
            raise ValueError("layer parameter hash mismatch")
        params[n["id"]] = (w, b)
    return graph, params


class IntegerGraphModel(nn.Module):
    """Canonical exact arithmetic model returning dequantized raw detection heads."""
    def __init__(self, folder, template=None):
        super().__init__()
        self.graph, params = _load_package(folder)
        self.levels = tuple(self.graph["outputs"])
        self.feature_strides = tuple(2 ** int(p[1:]) for p in self.levels)
        self.anchor_sizes = tuple(tuple(tuple(a) for a in d["anchors"]) for d in self.graph["descriptors"])
        self.candidate_architecture = deepcopy(self.graph["architecture"])
        self.input_channels, self.image_size = 1, 640
        object.__setattr__(self, "_template", template)
        self._params = {}
        for i, (name, (w, b)) in enumerate(params.items()):
            self.register_buffer(f"w{i}", torch.from_numpy(w))
            self.register_buffer(f"b{i}", torch.from_numpy(b))
            self._params[name] = (f"w{i}", f"b{i}")
        self._refs = {}
        for n in self.graph["nodes"]:
            for source in n["inputs"]:
                self._refs[source] = self._refs.get(source, 0) + 1

    @property
    def head(self):
        if self._template is None:
            raise AttributeError("head metadata requires a candidate template")
        return self._template.head

    def loss_config(self, config):
        if self._template is None:
            raise ValueError("loss_config requires a candidate template")
        return self._template.loss_config(config)

    @torch.no_grad()
    def forward_quantized(self, image, retain=False):
        if image.ndim != 4 or tuple(image.shape[1:]) != (1, 640, 640):
            raise ValueError("input must be NCHW grayscale640")
        exp = self.graph["inputs"]["image"]["exp"]
        x = image if image.dtype == torch.int8 else torch.round(image.to(torch.float64) / (2.0 ** exp)).clamp(-128, 127).to(torch.int8)
        values, refs = {"image": x}, dict(self._refs)
        final_nodes = set(self.graph["outputs"].values())
        for n in self.graph["nodes"]:
            a = values[n["inputs"][0]]
            if n["op"] == "conv":
                wname, bname = self._params[n["id"]]
                y = exact_conv_torch(a, getattr(self, wname), getattr(self, bname), n)
            elif n["op"] == "requant":
                y = requant_torch(a, n["multiplier"], n["shift"])
            elif n["op"] in ("add", "upsample_add"):
                b = values[n["inputs"][1]]
                if n["op"] == "upsample_add":
                    b = b.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
                common = n["common_exp"]
                qa = requant_torch(a, *numeric.ratio(n["input_exp_a"], common))
                qb = requant_torch(b, *numeric.ratio(n["input_exp_b"], common))
                y = requant_torch(qa.to(torch.int32) + qb.to(torch.int32), *numeric.ratio(common, n["output_exp"]))
            else:
                raise ValueError(f"unsupported graph operation {n['op']}")
            if list(y.shape[1:]) != self.graph["edges"][n["id"]]["shape"]:
                raise ValueError(f"graph output shape mismatch: {n['id']}")
            values[n["id"]] = y
            if not retain:
                for source in n["inputs"]:
                    refs[source] -= 1
                    if refs[source] == 0 and source not in final_nodes:
                        del values[source]
        return {level: values[node] for level, node in self.graph["outputs"].items()}, values

    def forward(self, image):
        outputs, _ = self.forward_quantized(image)
        return {"predictions": tuple(outputs[level].to(torch.float32) * (2.0 ** self.graph["edges"][node]["exp"])
                                     for level, node in self.graph["outputs"].items())}


def numpy_forward(folder, image):
    """Independent canonical NumPy INT64 oracle; intended for small fixtures."""
    graph, params = _load_package(folder)
    values = {"image": np.asarray(image, dtype=np.int8)}
    for n in graph["nodes"]:
        a = values[n["inputs"][0]]
        if n["op"] == "conv":
            w, b = params[n["id"]]
            y = numeric.conv(a, w, b, n)
        elif n["op"] == "requant":
            y = numeric.requantize_int32_to_int8(a, n["multiplier"], n["shift"])
        else:
            b = values[n["inputs"][1]]
            if n["op"] == "upsample_add":
                b = np.repeat(np.repeat(b, 2, axis=1), 2, axis=2)
            y = numeric.aligned_add(a, b, n["input_exp_a"], n["input_exp_b"], n["output_exp"])
        values[n["id"]] = y
    return {p: values[n] for p, n in graph["outputs"].items()}, values
