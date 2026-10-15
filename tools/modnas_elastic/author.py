"""Bridge to pinned, unchanged MODNAS author primitives.

This is an engineering adaptation to one ZCU104 and several sampled objective
preferences.  It does not reproduce MODNAS's multi-device training protocol.
The NB201 preference table and device-conditioned expert mixture are retained;
their architecture output is changed to seven binary choices.  GDAS and the
minimum-norm solver execute the original source from the pinned snapshot.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


AUTHOR_COMMIT = "21ddf2fee3e34db8e5828ddf6749fe037704e181"
AUTHOR_URL = "https://github.com/automl/MODNAS"
SOURCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "reference" / "_local" / f"MODNAS-core-{AUTHOR_COMMIT}"
)
DEFAULT_SOURCE_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "results" / "modnas_elastic_adapter_20261007" / "source_manifest.json"
)
# These were captured once from fixed-commit raw downloads.  Verification must
# compare to these expectations, never bless the current bytes by rehashing them.
EXPECTED_AUTHOR_SHA256 = {
    "README.md": "15963d79e9edfaae50e99c268034b55fc22aabe15fb4a29f71484e23d96e6562",
    "requirements.txt": "73b3f3d976625615f2b8e8eeb9ace48a799d22062a556ea7ed9bc3d850811430",
    "hypernetworks/models/hpn_nb201.py": "d3f653477ea0b2b83c2efbb762e846539da554fef56d2fbb4bce87106a40406e",
    "optimizers/sampler/base_sampler.py": "8e86f366e809bc1f6da485d6efebb420b1b71b1767c5357ae3a5ce515ad5702a",
    "optimizers/sampler/gdas.py": "06275fb846cf9511cf7cb3a8fdd542ef0543e14c3f24a2cdd89f1eec27c6dc61",
    "optimizers/mgd/min_norm_solvers.py": "f48f309c1e5a0ba7157d0d8e5471da1d3d82bab7d755e57462e684353fc9d812",
    "optimizers/mgd/__init__.py": "d25b514544aaee66e1055c5c3dd1527e42abbcd277f85ff72a99eb53835ce6bb",
}


def verify_author_sources(manifest_path=None):
    """Fail closed on a missing card, changed expectations, or changed source."""
    card_path = Path(manifest_path) if manifest_path is not None else DEFAULT_SOURCE_MANIFEST
    if not card_path.is_file():
        raise FileNotFoundError(f"Pinned MODNAS source manifest missing: {card_path}")
    manifest = json.loads(card_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != AUTHOR_URL or manifest.get("commit") != AUTHOR_COMMIT:
        raise RuntimeError("MODNAS source manifest repository/commit mismatch")
    if manifest.get("snapshot_kind") != "unchanged_author_core_dependency_closure_not_full_repository":
        raise RuntimeError("Unexpected MODNAS source snapshot kind")
    records = manifest.get("files", [])
    card_expected = {record["path"]: record["sha256"] for record in records}
    if len(records) != len(card_expected) or card_expected != EXPECTED_AUTHOR_SHA256:
        raise RuntimeError("MODNAS source manifest hashes differ from fixed expectations")
    verified = {}
    for relative, expected in EXPECTED_AUTHOR_SHA256.items():
        path = SOURCE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned MODNAS author source missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"MODNAS author SHA256 mismatch: {relative}")
        verified[relative] = actual
    return {"status": "PASS", "repository": AUTHOR_URL, "commit": AUTHOR_COMMIT,
            "manifest_path": str(card_path.resolve()), "source_root": str(SOURCE_ROOT),
            "sha256": verified, "license": manifest.get("license", {"status": "UNKNOWN"}),
            "snapshot_kind": manifest["snapshot_kind"]}


def _load(relative_path: str, name: str):
    # Verify before exec_module executes any author Python, including first import.
    verify_author_sources()
    path = SOURCE_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Pinned MODNAS author source missing: {path}")
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise RuntimeError(f"Unexpected module already loaded as {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned author source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_AUTHOR_HPN = _load("hypernetworks/models/hpn_nb201.py", "_modnas_elastic_author_hpn")
_AUTHOR_MGD = _load("optimizers/mgd/min_norm_solvers.py", "_modnas_elastic_author_mgd")


def _load_gdas():
    # The author's GDAS file has one absolute import.  Temporarily satisfy it
    # without putting the entire upstream repository first on sys.path, and
    # without importing optim_factory (which eagerly requires REINMAX).
    aliases = ("optimizers", "optimizers.sampler", "optimizers.sampler.base_sampler")
    saved = {name: sys.modules.get(name) for name in aliases}
    base = _load("optimizers/sampler/base_sampler.py", "_modnas_elastic_author_sampler_base")
    try:
        for name in aliases[:2]:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        sys.modules[aliases[2]] = base
        return _load("optimizers/sampler/gdas.py", "_modnas_elastic_author_gdas")
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_AUTHOR_GDAS = _load_gdas()
MinNormSolver = _AUTHOR_MGD.MinNormSolver


class _ChoiceHypernetwork(_AUTHOR_HPN.HyperNetwork):
    """Author preference-table HPN with a different architecture output."""

    def __init__(self, num_intervals=101, num_choices=7):
        super().__init__(num_intervals=num_intervals)
        self.num_choices = int(num_choices)
        self.embedding = nn.Embedding(self.num_intervals, self.num_choices * 2)

    def forward(self, preference):
        # This retains the author's rounded, discrete preference-table lookup.
        index = (preference[0, 0] * (self.num_intervals - 1)).round().long().reshape(1)
        return self.embedding(index).reshape(self.num_choices, 2)


class ConditionalHypernetwork(_AUTHOR_HPN.MetaHyperNetwork):
    """Preference and hardware-conditioned logits for seven binary decisions.

    forward(preference[1,2], hardware[1,hw_embed_dim]) -> logits[num_choices,2].
    Choice ordering belongs to the common-space adapter, not the author code.
    No upstream pretrained table or multi-device checkpoint is reused.
    """

    def __init__(self, num_intervals=101, num_random_devices=50,
                 hw_embed_dim=10, num_choices=7):
        if num_intervals < 2 or num_random_devices < 1 or hw_embed_dim < 1 or num_choices < 1:
            raise ValueError("Hypernetwork dimensions must be positive; num_intervals >= 2")
        self.num_choices = int(num_choices)
        factory = partial(_ChoiceHypernetwork, num_choices=self.num_choices)
        super().__init__(num_random_devices=num_random_devices,
                         hw_embed_dim=hw_embed_dim, hpn_type=factory,
                         num_intervals=num_intervals)

    def forward(self, preference, hardware):
        if preference.shape != (1, 2):
            raise ValueError("preference must have shape [1,2]")
        if hardware.shape != (1, self.hw_embed_dim):
            raise ValueError(f"hardware must have shape [1,{self.hw_embed_dim}]")
        if preference.device != self.hardware_embedding.weight.device or hardware.device != preference.device:
            raise ValueError("Hypernetwork, preference and hardware must use the same device")
        if not torch.isfinite(preference).all() or not torch.isfinite(hardware).all():
            raise ValueError("Condition inputs must be finite")
        if (preference < 0).any() or not torch.allclose(preference.sum(), preference.new_tensor(1.0), atol=1e-6):
            raise ValueError("preference must be nonnegative and sum to one")
        # Author NB201 sin-softmax device-conditioned expert mixture, with the
        # fixed 6x5 output allocation replaced by a stack of 7x2 expert outputs.
        similarity = (hardware @ self.hardware_embedding.weight.T).reshape(-1)
        similarity = similarity / (self.hw_embed_dim ** 0.5)
        weights = _AUTHOR_HPN.sin_softmax(similarity, dim=-1)
        expert_logits = torch.stack([hpn(preference) for hpn in self.hpn_factory])
        return (weights[:, None, None] * expert_logits).sum(dim=0)


class AuthorSampler(_AUTHOR_GDAS.GDASSampler):
    """Unchanged author GDAS sampling with checked temperature setup.

    sample(logits) and sample_step(list_of_logits) are inherited from upstream.
    The initial temperature is ready immediately; before_epoch() follows the
    author's schedule.  One-epoch smoke runs hold the initial temperature.
    """

    def __init__(self, tau_min=1.0, tau_max=10.0, total_epochs=2):
        if not np.isfinite([tau_min, tau_max]).all() or not (0 < tau_min <= tau_max):
            raise ValueError("Require finite 0 < tau_min <= tau_max")
        if int(total_epochs) != total_epochs or total_epochs < 1:
            raise ValueError("total_epochs must be a positive integer")
        super().__init__()
        self.set_taus(float(tau_min), float(tau_max))
        self.set_total_epochs(int(total_epochs))
        self.tau_curr = self.tau_max.clone()

    def before_epoch(self):
        if self.epoch >= self.total_epochs:
            raise ValueError("Sampler schedule exhausted")
        if self.total_epochs == 1:
            self.tau_curr = self.tau_max.clone()
            self.epoch += 1
        else:
            super().before_epoch()


def combine_gradients(gradient_groups: Sequence[Sequence[torch.Tensor]],
                      normalize=True):
    """Aggregate scalarized-condition gradients with the author MGD solver.

    Returns (combined_parameter_gradients, convex_weights_numpy, squared_norm).
    Parameter lists must match exactly; use explicit zero tensors for unused
    parameters so different preferences cannot silently change list alignment.
    A single group bypasses the author's undefined one-vector initialization.
    A zero gradient group selects that stationary group directly.
    """
    if not gradient_groups or not gradient_groups[0]:
        raise ValueError("At least one nonempty parameter-gradient group is required")
    templates = gradient_groups[0]
    groups = []
    for group in gradient_groups:
        if len(group) != len(templates):
            raise ValueError("Gradient groups must have equal parameter counts")
        copied = []
        for gradient, template in zip(group, templates):
            if gradient.shape != template.shape or gradient.device != template.device or gradient.dtype != template.dtype:
                raise ValueError("Gradient parameter shape/device/dtype mismatch")
            if not gradient.is_floating_point() or not torch.isfinite(gradient).all():
                raise ValueError("Gradient tensors must be finite floating-point values")
            copied.append(gradient.detach().clone())
        groups.append(copied)
    norms = [torch.sqrt(sum(g.square().sum() for g in group)).item() for group in groups]
    zero = next((i for i, norm in enumerate(norms) if norm == 0.0), None)
    if zero is not None:
        weights = np.zeros(len(groups), dtype=np.float64)
        weights[zero] = 1.0
        return groups[zero], weights, 0.0
    if normalize:
        groups = [[gradient / norm for gradient in group] for group, norm in zip(groups, norms)]
    if len(groups) == 1:
        cost = float(sum(gradient.square().sum().item() for gradient in groups[0]))
        return groups[0], np.ones(1, dtype=np.float64), cost
    result = MinNormSolver.find_min_norm_element_FW(groups)
    if result is None:
        raise RuntimeError("Author minimum-norm solver returned no solution")
    weights, cost = result
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < -1e-8).any() or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("Author MGD returned invalid convex weights")
    combined = [sum(float(weight) * group[index] for weight, group in zip(weights, groups))
                for index in range(len(templates))]
    return combined, weights, float(cost)


def source_provenance():
    paths = ("hypernetworks/models/hpn_nb201.py", "optimizers/sampler/base_sampler.py",
             "optimizers/sampler/gdas.py", "optimizers/mgd/min_norm_solvers.py")
    verified = verify_author_sources()
    return {"repository": AUTHOR_URL, "commit": AUTHOR_COMMIT,
            "snapshot_kind": "unchanged_author_core_dependency_closure_not_full_repository",
            "source_root": str(SOURCE_ROOT),
            "sha256": {path: verified["sha256"][path] for path in paths},
            "source_manifest": verified["manifest_path"], "license": verified["license"],
            "adaptations": ["NB201 output 6x5 -> seven binary decisions",
                            "single ZCU104 with multiple preferences instead of multiple devices",
                            "GDAS selected instead of REINMAX",
                            "single-vector and zero-gradient MGD guards",
                            "one-epoch constant-temperature GDAS guard"]}
