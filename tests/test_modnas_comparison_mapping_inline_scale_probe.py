"""Check byte-only scheduling changes and measured parent evidence for two probes."""
import json
from pathlib import Path

import pytest

from tools.modnas_comparison.mapping_inline_probe import INLINE_PRAGMA
from tools.modnas_comparison.mapping_inline_scale_probe import (
    CASE_FILES, DEFAULT_INLINE, DEFAULT_PREVIOUS, diagnose_inline_pe8, prepare_inline_scale_probe,
)
from tools.modnas_comparison.remote_worker import verify_case
from tools.modnas_elastic.cost_graph import canonical_hash


def test_two_probes_retain_raw_sources_witnesses_and_fixed_contract(tmp_path):
    output = tmp_path / "inline_scale"
    manifest = prepare_inline_scale_probe(output)
    assert manifest["case_count"] == 2
    assert [row["PE"] for row in manifest["cases"]] == [16, 32]
    for row in manifest["cases"]:
        folder = Path(row["case_dir"])
        key = verify_case(folder)
        original = DEFAULT_PREVIOUS / "cases" / f"stem_c12eb274bd029575_pe{row['PE']}_simd1"
        old = verify_case(original)
        assert key["source_hash"] == canonical_hash(key["source_files"]) == row["source_hash"]
        assert key["source_files"] == row["source_files"]
        assert key["witness_files"] == old["witness_files"] == row["witness_files"]
        for name in CASE_FILES:
            new_bytes = (folder / name).read_bytes()
            old_bytes = (original / name).read_bytes()
            if name == "numeric_v1.h":
                assert b"\r\n" not in new_bytes
                assert new_bytes.count(INLINE_PRAGMA.encode("utf-8")) == 2
                assert new_bytes.replace(INLINE_PRAGMA.encode("utf-8"), b"") == old_bytes
            else:
                assert new_bytes == old_bytes
        for field in ("numeric_contract_hash", "interface_contract_hash", "PE", "SIMD", "input_stream_bits",
                      "output_stream_bits", "target_clock_ns", "part", "tool_version"):
            assert key[field] == old[field]
        assert key["target_clock_ns"] == 5 and key["output_stream_bits"] == 256
        assert row["csim_status"] == row["csynth_status"] == "NOT_RUN"
        assert all(row[field] is None for field in ("latency_us", "LUT", "DSP", "BRAM18", "expected_speedup"))
    assert manifest["no_budget_change"] and manifest["no_arithmetic_change"] and manifest["no_pixel_pipeline_change"]
    with pytest.raises(FileExistsError):
        prepare_inline_scale_probe(output)


def test_measured_pe8_causality_matches_raw_xml_and_complete_output():
    diagnosis = diagnose_inline_pe8()
    assert diagnosis["metrics"]["latency_us"] == 197848.37
    assert diagnosis["metrics"]["LUT"] == 11173
    assert diagnosis["full_byte_csim_verified"] and diagnosis["output_bytes"] == 3276800
    assert not diagnosis["shared_finish_helper_present"]
    assert diagnosis["scan_loop"]["maximum_iteration_latency_cycles"] == 96
    assert diagnosis["output_group_loop"]["trip_count"] == 4
    assert diagnosis["output_group_loop"]["iteration_latency_cycles"] == 22
    assert diagnosis["baseline_exact_measurement"]["PE"] == 2
    assert diagnosis["counterfactual_latency_or_resources"] is None
    assert not diagnosis["board_measured"] and not diagnosis["full_network_measured"]


def test_rejects_parent_preparation_hash_drift_before_writing(tmp_path):
    inline = tmp_path / "invalid_parent"
    inline.mkdir()
    for name in ("preparation_manifest.json", "verified_results.json"):
        (inline / name).write_bytes((DEFAULT_INLINE / name).read_bytes())
    verified_path = inline / "verified_results.json"
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    verified["preparation_manifest_sha256"] = "0" * 64
    verified_path.write_bytes(json.dumps(verified).encode("utf-8"))
    output = tmp_path / "rejected_probe"
    with pytest.raises(ValueError, match="preparation provenance"):
        prepare_inline_scale_probe(output, inline=inline)
    assert not output.exists()
