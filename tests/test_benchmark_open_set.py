from pathlib import Path

import pytest

from hwnas_fpga.benchmarks.open_set import load_open_set_spec


CLASS_ORDER = (
    "big_propeller",
    "cylinder",
    "fishing_net",
    "floats",
    "iron_pipeline",
    "small_propeller",
    "soft_pipeline",
    "tire",
)


def test_frozen_open_set_manifest_has_five_known_three_unknown_per_fold():
    path = Path("configs/benchmarks/nksid_open_long_tail_v1.yaml")
    unknown_sets = []
    for fold in range(5):
        spec = load_open_set_spec(path, fold=fold, observed_classes=CLASS_ORDER)
        assert len(spec.known_class_ids) == 5
        assert len(spec.unknown_class_ids) == 3
        assert set(spec.known_class_ids).isdisjoint(spec.unknown_class_ids)
        unknown_sets.append(spec.unknown_class_ids)
    assert len(set(unknown_sets)) == 5


def test_open_set_manifest_rejects_dataset_class_order_drift():
    with pytest.raises(ValueError, match="class_order does not match"):
        load_open_set_spec(
            "configs/benchmarks/nksid_open_long_tail_v1.yaml",
            fold=0,
            observed_classes=tuple(reversed(CLASS_ORDER)),
        )
