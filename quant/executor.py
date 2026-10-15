"""Train-only PTQ execution for an externally selected elastic72 model.

Planning remains in :mod:`quant`; this module never builds a model, starts
training, selects validation data, or claims hardware measurements.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from space import Genotype, canonical_hash, validate_sha256
from quant import QuantizationPlan
from .runtime import quantization_mixed as _runtime

TrainCalibration = _runtime.TrainCalibration
PreparedElasticPTQ = _runtime.PreparedElasticPTQ
model_state_hash = _runtime.model_state_hash


def _model_genotype(model):
    try:
        architecture = model.candidate_architecture
        if set(architecture) != {"pan", "td_kernels", "pan_kernels"}:
            raise ValueError("Selected model architecture fields differ from elastic72")
        return Genotype(**architecture, num_classes=model.num_classes)
    except AttributeError as error:
        raise TypeError("Model must expose candidate_architecture and num_classes") from error


def _verify_binding(prepared):
    profile = prepared.profile
    plan = QuantizationPlan.from_dict(profile["quantization_plan"])
    genotype = Genotype.from_dict(profile["genotype"])
    if (profile["arch_hash"] != genotype.arch_hash or plan.arch_hash != genotype.arch_hash
            or profile["q_hash"] != plan.q_hash or profile["precision_profile"] != plan.profile):
        raise ValueError("PTQ plan/architecture identity drift")
    for graph in (prepared.fake_model.graph, prepared.integer_reference.graph):
        payload = deepcopy(graph)
        declared = payload.pop("numeric_graph_sha256", None)
        if declared != profile["numeric_graph_sha256"] or canonical_hash(payload) != declared:
            raise ValueError("PTQ numeric graph drift")
        if (graph["architecture"] != genotype.legacy_architecture()
                or graph["num_classes"] != genotype.num_classes):
            raise ValueError("PTQ graph differs from bound architecture")
    return plan


def prepare_ptq(model, calibration, *, plan, device="cpu"):
    """Calibrate H2 body8 with an optional head16 using frozen train files.

    ``model`` is an already selected, eval-mode FP32 module. ``calibration``
    must be ``TrainCalibration``; arbitrary tensors and valid/test paths are
    rejected. The default formal prefix is the first 64 frozen train entries.
    """
    if not isinstance(plan, QuantizationPlan):
        raise TypeError("prepare_ptq requires a QuantizationPlan from Q(genotype)")
    genotype = _model_genotype(model)
    if genotype.arch_hash != plan.arch_hash:
        raise ValueError("Quantization plan belongs to a different model architecture")
    if not isinstance(calibration, TrainCalibration):
        raise TypeError("PTQ calibration must be source-bound train-only TrainCalibration")
    prepared = _runtime.prepare_elastic_ptq(model, calibration, device=device, profile=plan.profile)
    migration = Path(_runtime.__file__).with_name("MIGRATION.json")
    prepared.profile.update(
        arch_hash=genotype.arch_hash, q_hash=plan.q_hash, genotype=genotype.to_dict(),
        quantization_plan=plan.to_dict(),
        runtime_migration_sha256=hashlib.sha256(migration.read_bytes()).hexdigest(),
    )
    _verify_binding(prepared)
    return prepared


def verify_fake_integer_parity(prepared, images):
    """Measure software fake/integer agreement, preserving its evidence scope."""
    plan = _verify_binding(prepared)
    witness = _runtime.verify_fake_integer_parity(prepared, images)
    witness.update(arch_hash=plan.arch_hash, q_hash=plan.q_hash,
                   precision_profile=plan.profile)
    return witness


def save_ptq_artifacts(output, prepared, *, witness=None):
    """Save a new artifact folder containing the architecture and plan identities."""
    plan = _verify_binding(prepared)
    if witness is not None and any(witness.get(key) != value for key, value in
                                  (("arch_hash", plan.arch_hash), ("q_hash", plan.q_hash))):
        raise ValueError("Software parity witness belongs to a different PTQ plan")
    return _runtime.save_ptq_artifacts(output, prepared, witness=witness)


def prepare_qat(*args, **kwargs):
    raise NotImplementedError("The retained elastic72 mainline has no QAT executor")


def verify_saved_artifacts(folder, *, expected_arch_hash, expected_q_hash):
    """Verify all portable PTQ leaves, their identities and integer tensor hashes.

    Source dataset/checkpoint files are not reopened here. Their bound hashes
    remain in ``profile.json`` and the software witness for the caller to link
    with the experiment receipt. This check creates no new hardware evidence.
    """
    import torch

    validate_sha256(expected_arch_hash)
    validate_sha256(expected_q_hash)
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
    required = {"profile.json", "extrema.json", "numeric_graph.json", "integer_parameters.pt"}
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Incomplete PTQ artifact manifest")
    if set(files) - required - {"software_parity_witness.json"}:
        raise ValueError("Unexpected PTQ artifact manifest entries")
    for name, expected in files.items():
        validate_sha256(expected)
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise ValueError("PTQ artifact leaf resolves outside its bundle: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("PTQ artifact SHA256 drift: " + name)
    profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    genotype = Genotype.from_dict(profile["genotype"])
    plan = QuantizationPlan.from_dict(profile["quantization_plan"])
    if (genotype.arch_hash != expected_arch_hash or plan.arch_hash != expected_arch_hash
            or plan.q_hash != expected_q_hash or profile["precision_profile"] != plan.profile):
        raise ValueError("Saved PTQ profile belongs to a different architecture/plan")
    package = torch.load(folder / "integer_parameters.pt", map_location="cpu", weights_only=True)
    identity = dict(arch_hash=expected_arch_hash, q_hash=expected_q_hash,
                    genotype=genotype.to_dict(), quantization_plan=plan.to_dict())
    for payload in (manifest, profile, package):
        if any(payload.get(key) != value for key, value in identity.items()):
            raise ValueError("Saved PTQ architecture/plan identity drift")
    if manifest.get("formal_eligible") is not False or manifest.get("test_evaluated") is not False:
        raise ValueError("Saved PTQ bundle changed its evidence scope")
    graph = json.loads((folder / "numeric_graph.json").read_text(encoding="utf-8"))
    content = deepcopy(graph)
    declared = content.pop("numeric_graph_sha256", None)
    if (canonical_hash(content) != declared or profile["numeric_graph_sha256"] != declared
            or package["graph"] != graph or graph["architecture"] != genotype.legacy_architecture()
            or graph["num_classes"] != genotype.num_classes):
        raise ValueError("Saved PTQ numeric graph drift")
    convs = {node["id"]: node for node in graph["nodes"] if node["op"] == "conv"}
    if set(package["weights"]) != set(convs) or set(package["biases"]) != set(convs):
        raise ValueError("Saved integer tensor membership differs from graph")
    hashes = {}
    for name, node in sorted(convs.items()):
        weight, bias = package["weights"][name], package["biases"][name]
        bits = node.get("weight_bits", 8)
        dtype = {8: torch.int8, 16: torch.int16}.get(bits)
        shape = (node["Cout"], node["Cin"] // node["groups"], node["kernel"], node["kernel"])
        if (dtype is None or not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor)
                or weight.dtype != dtype or bias.dtype != torch.int32
                or tuple(weight.shape) != shape or tuple(bias.shape) != (node["Cout"],)):
            raise ValueError("Saved integer tensor dtype/geometry differs at " + name)
        hashes[name] = dict(weight_sha256=hashlib.sha256(weight.contiguous().numpy().tobytes()).hexdigest(),
                            bias_sha256=hashlib.sha256(bias.contiguous().numpy().tobytes()).hexdigest())
    if canonical_hash(hashes) != profile["integer_parameters_sha256"]:
        raise ValueError("Saved integer tensor source hash drift")
    if package["numeric_contract_hash"] != profile["numeric_contract_hash"]:
        raise ValueError("Saved integer numeric contract differs from profile")
    calibration = deepcopy(profile["calibration"])
    calibration_hash = calibration.pop("calibration_sha256", None)
    if (canonical_hash(calibration) != calibration_hash or calibration.get("split") != "train"
            or any(calibration.get(key) is not False for key in
                   ("augmentation", "shuffle", "valid_used_for_selection", "test_accessed", "test_evaluated"))):
        raise ValueError("Saved PTQ calibration is not the bound train-only manifest")
    if "software_parity_witness.json" in files:
        witness = json.loads((folder / "software_parity_witness.json").read_text(encoding="utf-8"))
        expected = dict(arch_hash=expected_arch_hash, q_hash=expected_q_hash,
                        numeric_graph_sha256=declared, calibration_sha256=calibration_hash,
                        original_model_state_sha256=profile["original_model_state_sha256"],
                        integer_parameters_sha256=profile["integer_parameters_sha256"])
        if any(witness.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved software witness differs from PTQ source identities")
        if (witness.get("formal_eligible") is not False or witness.get("board_parity") != "NOT_RUN"
                or witness.get("test_evaluated") is not False):
            raise ValueError("Software witness changed its evidence scope")
    return manifest


__all__ = ["TrainCalibration", "PreparedElasticPTQ", "model_state_hash", "prepare_ptq",
           "verify_fake_integer_parity", "save_ptq_artifacts", "verify_saved_artifacts", "prepare_qat"]
