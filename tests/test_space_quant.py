import dataclasses
import pytest
from space import Genotype, enumerate_genotypes, encode, decode
from quant import Q, QuantizationPlan


def test_complete_conditional_space_and_roundtrip():
    candidates = list(enumerate_genotypes())
    assert len(candidates) == len({g.arch_hash for g in candidates}) == 72
    assert sum(not g.pan for g in candidates) == 8
    for i, g in enumerate(candidates):
        assert encode(g) == i and decode(i) == g
        assert Genotype.from_json(g.to_json()) == g
        assert len(g.pan_kernels) == (3 if g.pan else 0)
        assert len({Q(g, p).q_hash for p in ("W8A8", "HEAD_W8A16", "HEAD_W16A16")}) == 3


def test_identity_covers_classes_and_input_contract():
    g = decode(0)
    assert dataclasses.replace(g, num_classes=10).arch_hash != g.arch_hash
    assert g.arch_hash != g.legacy_arch_hash
    value = g.to_dict()
    value["input_chw"] = [3, 640, 640]
    with pytest.raises(ValueError):
        Genotype.from_dict(value)
    value = g.to_dict()
    value["pan_kernels"] = [3, 3, 3]
    with pytest.raises(ValueError):
        Genotype.from_dict(value)


@pytest.mark.parametrize("pan,td,bottom", [(0,[3]*3,[]),(False,[True,3,3],[]),(True,[3]*3,[])])
def test_illegal_candidates_rejected(pan,td,bottom):
    with pytest.raises(ValueError):
        Genotype(pan,td,bottom)


def test_q_pure_and_plan_tamper_rejected():
    g = decode(8)
    assert Q(g) == Q(g)
    value = Q(g).to_dict()
    assert QuantizationPlan.from_dict(value) == Q(g)
    value["body"]["activation_bits"] = 16
    with pytest.raises(ValueError):
        QuantizationPlan.from_dict(value)
