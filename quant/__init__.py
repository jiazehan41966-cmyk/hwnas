"""Sq: pure architecture-dependent precision planning, separate from PTQ."""
from dataclasses import dataclass
from space import Genotype, canonical_hash, validate_sha256

PROFILES = ("W8A8", "HEAD_W8A16", "HEAD_W16A16")
QUANT_CONTRACT_VERSION = "elastic72_h2_body_mixed_head_v1"


@dataclass(frozen=True)
class QuantizationPlan:
    arch_hash: str
    profile: str = "W8A8"

    def __post_init__(self):
        validate_sha256(self.arch_hash)
        if self.profile not in PROFILES:
            raise ValueError("Unsupported registered precision profile")

    def to_dict(self):
        head_weight_bits = 16 if self.profile == "HEAD_W16A16" else 8
        head_output_bits = 8 if self.profile == "W8A8" else 16
        return dict(schema_version=1, contract_version=QUANT_CONTRACT_VERSION,
                    arch_hash=self.arch_hash, profile=self.profile,
                    body=dict(weight_bits=8, activation_bits=8, accumulator_bits=32),
                    head=dict(weight_bits=head_weight_bits, input_bits=head_output_bits,
                              output_bits=head_output_bits, accumulator_bits=32),
                    calibration_policy=dict(split="train", images=64, augmentation=False),
                    status="PLAN_ONLY_NOT_CALIBRATED")

    @property
    def q_hash(self):
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Quantization plan must be a mapping")
        result = cls(value.get("arch_hash"), value.get("profile"))
        if value != result.to_dict():
            raise ValueError("Quantization plan differs from registered numeric contract")
        return result


def Q(genotype, profile="W8A8"):
    if not isinstance(genotype, Genotype):
        raise TypeError("Q requires a Genotype")
    return QuantizationPlan(genotype.arch_hash, profile)


def enumerate_plans(genotype):
    return tuple(Q(genotype, profile) for profile in PROFILES)
