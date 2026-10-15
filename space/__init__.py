"""Sx: immutable, versioned candidate identities; no ML/runtime imports."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import re

SPACE_VERSION = "elastic_neck72_v1"
MODEL_CONTRACT_VERSION = "elastic72_b1_gray640_u48_mb3e3_direct_head_v1"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_sha256(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class Genotype:
    pan: bool
    td_kernels: tuple[int, int, int]
    pan_kernels: tuple[int, ...] = ()
    num_classes: int = 2

    def __post_init__(self):
        if type(self.pan) is not bool or type(self.num_classes) is not int or self.num_classes < 1:
            raise ValueError("pan must be bool; num_classes must be a positive integer")
        for field, length in (("td_kernels", 3), ("pan_kernels", 3 if self.pan else 0)):
            values = getattr(self, field)
            if not isinstance(values, (tuple, list)) or len(values) != length:
                raise ValueError(f"{field} requires {length} active choices")
            if any(type(k) is not int or k not in (3, 5) for k in values):
                raise ValueError("Only integer kernel choices 3 and 5 are registered")
            object.__setattr__(self, field, tuple(values))

    def legacy_architecture(self):
        return dict(pan=self.pan, td_kernels=list(self.td_kernels), pan_kernels=list(self.pan_kernels))

    def to_dict(self):
        return dict(schema_version=1, space_version=SPACE_VERSION,
                    model_contract_version=MODEL_CONTRACT_VERSION,
                    input_chw=[1, 640, 640], num_classes=self.num_classes,
                    **self.legacy_architecture())

    def to_json(self):
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Genotype must be a mapping")
        expected = {"schema_version", "space_version", "model_contract_version", "input_chw",
                    "num_classes", "pan", "td_kernels", "pan_kernels"}
        if set(value) != expected:
            raise ValueError("Genotype fields must match the versioned schema exactly")
        if (type(value["schema_version"]) is not int or value["schema_version"] != 1
                or value["space_version"] != SPACE_VERSION
                or value["model_contract_version"] != MODEL_CONTRACT_VERSION
                or value["input_chw"] != [1, 640, 640]
                or any(type(i) is not int for i in value["input_chw"])):
            raise ValueError("Unsupported genotype version or fixed input contract")
        return cls(value["pan"], value["td_kernels"], value["pan_kernels"], value["num_classes"])

    @classmethod
    def from_json(cls, value):
        return cls.from_dict(json.loads(value))

    @property
    def arch_hash(self):
        return canonical_hash(self.to_dict())

    @property
    def legacy_arch_hash(self):
        """Mapping to historical choice-only hashes; never the new cache key."""
        return canonical_hash(self.legacy_architecture())

    @property
    def architecture_id(self):
        td = "_".join(f"{p}k{k}" for p, k in zip(("P2", "P3", "P4"), self.td_kernels))
        suffix = ("_PAN_" + "_".join(f"{p}k{k}" for p, k in zip(("P3", "P4", "P5"), self.pan_kernels))) if self.pan else ""
        return ("E1_" if self.pan else "E0_") + "TD_" + td + suffix


def enumerate_genotypes(num_classes=2):
    for pan in (False, True):
        for td in product((3, 5), repeat=3):
            for bottom in product((3, 5), repeat=3 if pan else 0):
                yield Genotype(pan, td, bottom, num_classes)


def encode(genotype):
    if not isinstance(genotype, Genotype):
        raise TypeError("Genotype required")
    return next(i for i, g in enumerate(enumerate_genotypes(genotype.num_classes)) if g == genotype)


def decode(index, num_classes=2):
    if type(index) is not int or not 0 <= index < 72:
        raise ValueError("Encoding must be an integer in [0,72)")
    return tuple(enumerate_genotypes(num_classes))[index]
