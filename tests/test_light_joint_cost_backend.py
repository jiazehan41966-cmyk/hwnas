"""Temporary synthetic files exercise validation only; never research evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from tools.modnas_comparison import joint_cost_backend as cost


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def load(path):
    return cost.load_verified_joint_cost_backend(path, num_classes=2, profiles=cost.PROFILES,
        policies=cost.POLICIES, space_version=cost.SEARCH_SPACE_VERSION)


ARCH = dict(pan=False, td_kernels=[3, 3, 3], pan_kernels=[])
MAPPING = dict(PE=1, SIMD=1, buffer_profile="test_fixture", implementation_id="temporary_synthetic_test_only")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    base = tmp_path_factory.mktemp("light_cost")
    manifest = cost.empty_manifest()
    graph = cost.graph_requirements(ARCH, "W8A8")
    source = base / "synthetic.cpp"
    source.write_text("// SYNTHETIC VALIDATOR FIXTURE; NOT ACTUAL HLS EVIDENCE\n", encoding="utf-8")
    sources = {str(source): cost.sha(source)}
    proof = dict(status="REVIEWED_INTEGER_TOPOLOGY_CONTRACT", topology=graph,
                 implementation_sources=manifest["implementation_sources"],
                 source_files=sources, source_hash=cost.canonical_hash(sources))
    manifest["graph_contracts"] = [dict(architecture_hash=cost.architecture_hash(ARCH), profile="W8A8",
                                      evidence=cost._bind(write(base / "graph.json", proof)))]
    requirements = {row["primitive_requirement_id"]: row["requirement"] for row in graph["nodes"]}
    for index, (rid, req) in enumerate(requirements.items()):
        folder = base / str(index)
        folder.mkdir()
        task_id = cost.canonical_hash(dict(primitive_requirement_id=rid, mapping=MAPPING))
        plan = dict(status="FROZEN_CONCRETE_MAPPING", primitive_requirement_id=rid,
                    mapping_policy="reference_low", mapping=MAPPING)
        manifest["mapping_bindings"].append(dict(primitive_requirement_id=rid,
            mapping_policy="reference_low", mapping=MAPPING, evidence=cost._bind(write(folder / "mapping.json", plan))))
        shape, precision = req["primitive"], req["precision"]
        (folder / "input.bin").write_bytes(bytes(sum(math.prod(s) for s in shape["input_shapes"]) * precision["input_bitwidth"] // 8))
        (folder / "reference.bin").write_bytes(bytes(math.prod(shape["output_shape"]) * precision["output_bitwidth"] // 8))
        wcount = shape["Cout"] * (shape["Cin"] // shape["groups"]) * shape["kernel"] ** 2 if shape["op"] == "conv" else 1
        bcount = shape["Cout"] * 4 if shape["op"] == "conv" else 1
        (folder / "weights.bin").write_bytes(bytes(wcount))
        (folder / "biases.bin").write_bytes(bytes(bcount))
        witnesses = {name: cost.sha(folder / name) for name in ("input.bin", "reference.bin", "weights.bin", "biases.bin")}
        key = dict(task_id=task_id, requirement=req, mapping=MAPPING, csim_status="PASS_BIT_EXACT",
            source_files=sources, source_hash=cost.canonical_hash(sources), witness_files=witnesses,
            witness_hash=cost.canonical_hash(witnesses),
            numeric_contract=cost._bind(cost.ROOT / "reports/fpga_h2_contract/numeric_contract_v1.json"),
            interface_contract=cost._bind(cost.ROOT / "reports/fpga_h2_contract/interface_contract_v1.json"))
        keypath = write(folder / "key.json", key)
        xml = folder / "csynth.xml"
        xml.write_text(f"""<Report><ReportVersion><Version>2022.2</Version></ReportVersion>
<UserAssignments><Part>{cost.PART}</Part><TargetClockPeriod>5</TargetClockPeriod></UserAssignments>
<AreaEstimates><Resources><LUT>17</LUT><DSP>2</DSP><BRAM_18K>1</BRAM_18K></Resources></AreaEstimates>
<SummaryOfOverallLatency><Worst-caseLatency>100</Worst-caseLatency></SummaryOfOverallLatency>
<SummaryOfTimingAnalysis><EstimatedClockPeriod>4.5</EstimatedClockPeriod></SummaryOfTimingAnalysis></Report>""", encoding="utf-8")
        log = folder / "test.log"
        log.write_text("synthetic test fixture\n", encoding="utf-8")
        execution = dict(task_id=task_id, measurement_key_sha256=cost.sha(keypath), source_hash=key["source_hash"],
                         csynth_report_sha256=cost.sha(xml), csim_output_sha256=cost.sha(folder / "reference.bin"),
                         csim_log_sha256=cost.sha(log), hls_log_sha256=cost.sha(log), status="PASS_CSIM_AND_CSYNTH")
        receipt = dict(task_id=task_id, measurement_key=cost._bind(keypath),
                       csim_output=cost._bind(folder / "reference.bin"), csynth_report=cost._bind(xml),
                       csim_log=cost._bind(log), hls_log=cost._bind(log),
                       execution_record=cost._bind(write(folder / "execution.json", execution)))
        manifest["measurements"].append(receipt)
    path = write(base / "manifest.json", manifest)
    return base, manifest, path


def test_empty_unknown_is_not_zero_cost(tmp_path):
    backend = load(write(tmp_path / "empty.json", cost.empty_manifest()))
    row = backend.query(ARCH, "HEAD_W16A16", "reference_high")
    assert row["status"] == "NEEDS_MEASUREMENT" and row["Latency"] is None
    assert row["body8_to_head16_boundary_count"] == 4
    assert row["measured_node_count"] == 0


def test_exact_full_graph_aggregation_and_no_profile_mapping_alias(bundle):
    _, _, path = bundle
    backend = load(path)
    row = backend.query(ARCH, "W8A8", "reference_low")
    assert row["status"] == "LOOKED_UP"
    assert row["Latency"] == row["graph_node_count"] * .5
    assert row["LUT"] == 17 and row["DSP"] == 2 and row["BRAM"] == 1
    assert backend.query(ARCH, "HEAD_W8A16", "reference_low")["status"] == "NEEDS_MEASUREMENT"
    assert backend.query(ARCH, "W8A8", "reference_high")["status"] == "NEEDS_MEASUREMENT"


def test_mixed_heads_have_explicit_distinct_contracts():
    a = cost.graph_requirements(ARCH, "HEAD_W8A16")
    b = cost.graph_requirements(ARCH, "HEAD_W16A16")
    boundaries = [n for n in a["nodes"] if n["node_id"].endswith("input_requant")]
    assert len(boundaries) == 4
    for node in boundaries:
        assert node["requirement"]["precision"]["output_bitwidth"] == 16
        assert node["requirement"]["interface_contract_hash"] is None
    assert a["nodes"][-1]["primitive_requirement_id"] != b["nodes"][-1]["primitive_requirement_id"]
    assert a["nodes"][0]["primitive_requirement_id"] == b["nodes"][0]["primitive_requirement_id"]


def test_missing_one_node_cannot_claim_full_cost(bundle, tmp_path):
    _, manifest, _ = bundle
    modified = deepcopy(manifest)
    modified["measurements"].pop()
    row = load(write(tmp_path / "partial.json", modified)).query(ARCH, "W8A8", "reference_low")
    assert row["status"] == "NEEDS_MEASUREMENT" and row["Latency"] is None


def test_source_drift_and_duplicate_are_rejected(bundle, tmp_path):
    _, manifest, _ = bundle
    modified = deepcopy(manifest)
    modified["implementation_sources"][cost.SOURCES[0]] = "0" * 64
    with pytest.raises(ValueError, match="source drift"):
        load(write(tmp_path / "drift.json", modified))
    modified = deepcopy(manifest)
    modified["measurements"].append(modified["measurements"][0])
    with pytest.raises(ValueError, match="Duplicate measurement"):
        load(write(tmp_path / "duplicate.json", modified))


def test_raw_report_platform_and_hash_rejected(bundle, tmp_path):
    _, manifest, _ = bundle
    modified = deepcopy(manifest)
    receipt = modified["measurements"][0]
    report = tmp_path / "changed.xml"
    report.write_text(Path(receipt["csynth_report"]["path"]).read_text().replace(cost.PART, "wrong-part"))
    receipt["csynth_report"] = cost._bind(report)
    with pytest.raises(ValueError, match="platform/tool/clock"):
        load(write(tmp_path / "bad-platform.json", modified))
    receipt["csynth_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash-mismatched"):
        load(write(tmp_path / "bad-hash.json", modified))


def test_import_accepts_verified_results_and_rolls_back_conflict(bundle, tmp_path):
    _, manifest, _ = bundle
    original = deepcopy(manifest)
    original["measurements"] = []
    source = write(tmp_path / "source.json", original)
    receipts = write(tmp_path / "receipts.json", manifest["measurements"])
    output = tmp_path / "accepted.json"
    result = cost.import_measurement_results(source, receipts, output)
    assert result["accepted_tasks"] == len(manifest["measurements"])
    assert load(output).query(ARCH, "W8A8", "reference_low")["status"] == "LOOKED_UP"
    malformed = deepcopy(manifest["measurements"][:1])
    malformed[0]["csynth_report"]["sha256"] = "0" * 64
    bad = write(tmp_path / "bad-receipts.json", malformed)
    failed = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        cost.import_measurement_results(source, bad, failed)
    assert not failed.exists() and not failed.with_suffix(".json.pending").exists()


def test_queue_deduplicates_occurrences_without_measurement(bundle):
    _, _, path = bundle
    queue = cost.build_measurement_queue(load(path))
    assert queue["triple_count"] == 1296
    assert queue["deduplicated_tasks"] < sum(len(task["uses"]) for task in queue["tasks"])
    assert queue["complete_triples"] == 1
    assert queue["hls_launched"] is False
