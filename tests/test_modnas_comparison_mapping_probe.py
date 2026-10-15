"""Prepared PE changes preserve exact numeric witnesses and never invent costs."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_comparison.mapping_probe import (PARALLELISM, WITNESS_FILES, prepare_mapping_probe,
                                                   rewrite_parallelism)
from tools.modnas_elastic.cost_graph import canonical_hash


def test_all_three_prepared_realshape_witnesses_preserve_sources_and_unknown_costs(tmp_path):
    output = tmp_path / "mapping_probe"
    manifest = prepare_mapping_probe(output)
    assert manifest["case_count"] == 3
    assert manifest["baseline_numpy_reference_replay"]["status"] == "PASS_NUMPY_INTEGER_REFERENCE_REPLAY"
    assert manifest["baseline_numpy_reference_replay"]["samples"] == 32 * 320 * 320
    baseline = ROOT / manifest["baseline_exact_measurement"]["case_dir"]
    original = (baseline / "kernel.cpp").read_text(encoding="utf-8")
    for pe, case in zip(PARALLELISM, manifest["cases"]):
        folder = Path(case["case_dir"])
        key = json.loads((folder / "measurement_key.json").read_text(encoding="utf-8"))
        assert key["PE"] == key["output_tile"] == pe
        assert key["SIMD"] == key["input_tile"] == 1
        assert key["input_stream_bits"] == 8 and key["output_stream_bits"] == 256
        assert (folder / "kernel.cpp").read_text(encoding="utf-8") == original.replace(
            "static constexpr int OUT_TILE = 2;", f"static constexpr int OUT_TILE = {pe};")
        for filename in (*WITNESS_FILES, "numeric_v1.h", "tb.cpp", "run.tcl"):
            assert (folder / filename).read_bytes() == (baseline / filename).read_bytes()
        assert canonical_hash(key["source_files"]) == key["source_hash"]
        for filename, digest in {**key["source_files"], **key["witness_files"]}.items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest
        assert case["csim_status"] == case["csynth_status"] == "NOT_RUN"
        assert all(case[name] is None for name in ("latency_us", "LUT", "DSP", "BRAM18", "expected_speedup"))
    with pytest.raises(FileExistsError):
        prepare_mapping_probe(output)


@pytest.mark.parametrize("parallelism", [0, 2, 4, 7, 64, True])
def test_unregistered_parallelism_rejected(parallelism):
    with pytest.raises(ValueError):
        rewrite_parallelism("static constexpr int IN_TILE = 1;", parallelism)


def test_template_drift_blocks_preparation():
    with pytest.raises(ValueError, match="Baseline output"):
        rewrite_parallelism("static constexpr int IN_TILE = 1;\nstatic constexpr int OUT_TILE = 4;", 8)
