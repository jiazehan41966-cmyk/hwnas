"""Inline scheduling must preserve every original arithmetic and witness byte."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_comparison.mapping_inline_probe import (DEFAULT_PREVIOUS, INLINE_PRAGMA,
                                                         force_inline_numeric, prepare_inline_probe)
from tools.modnas_comparison.mapping_probe import WITNESS_FILES


def test_inline_numeric_only_adds_two_pragmas_and_rejects_drift():
    source = (DEFAULT_PREVIOUS / "cases/stem_c12eb274bd029575_pe8_simd1/numeric_v1.h").read_text(encoding="utf-8")
    changed = force_inline_numeric(source)
    assert changed.count(INLINE_PRAGMA) == 2
    assert changed.replace(INLINE_PRAGMA, "") == source
    assert "ap_int<64> product=ap_int<64>(a)*ap_int<64>(multiplier);" in changed
    with pytest.raises(ValueError):
        force_inline_numeric(changed)
    with pytest.raises(ValueError):
        force_inline_numeric(source.replace("inline ap_int<8> finish", "ap_int<8> finish"))


def test_one_case_preserves_parameters_port_protocol_and_all_witnesses(tmp_path):
    output = tmp_path / "inline_probe"
    manifest = prepare_inline_probe(output)
    assert manifest["case_count"] == 1
    record = manifest["cases"][0]
    folder = Path(record["case_dir"])
    original = DEFAULT_PREVIOUS / "cases/stem_c12eb274bd029575_pe8_simd1"
    key = json.loads((folder / "measurement_key.json").read_text(encoding="utf-8"))
    old = json.loads((original / "measurement_key.json").read_text(encoding="utf-8"))
    for field in ("numeric_contract_hash", "interface_contract_hash", "PE", "SIMD", "input_stream_bits",
                  "output_stream_bits", "target_clock_ns", "part", "witness_files"):
        assert key[field] == old[field]
    assert key["source_hash"] != old["source_hash"]
    assert {name for name in key["source_files"] if key["source_files"][name] != old["source_files"][name]} == {"numeric_v1.h"}
    original_header = (original / "numeric_v1.h").read_bytes()
    modified_header = (folder / "numeric_v1.h").read_bytes()
    assert b"\r\n" not in original_header
    assert b"\r\n" not in modified_header
    assert modified_header.count(INLINE_PRAGMA.encode("utf-8")) == 2
    assert modified_header.replace(INLINE_PRAGMA.encode("utf-8"), b"") == original_header
    for filename in (*WITNESS_FILES, "kernel.cpp", "run.tcl", "tb.cpp", "common_defs.h", "common_types.h"):
        assert (folder / filename).read_bytes() == (original / filename).read_bytes()
    assert record["csim_status"] == record["csynth_status"] == "NOT_RUN"
    assert record["expected_speedup"] is None
    assert manifest["no_budget_change"] and manifest["no_arithmetic_change"] and manifest["no_pixel_pipeline_change"]
    diagnosis = json.loads((output / "causal_diagnosis.json").read_text(encoding="utf-8"))
    assert [row["PE"] for row in diagnosis["measured"]] == [8, 16, 32]
    assert all(row["finish_helper"]["single_instance_in_top_report"] for row in diagnosis["measured"])
    assert all(row["finish_helper"]["latency_cycles"] == 6 for row in diagnosis["measured"])
