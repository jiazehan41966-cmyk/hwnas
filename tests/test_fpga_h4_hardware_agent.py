"""Hardware Cost Agent regression: exact lookup, no FP32 fallback, full 144-architecture pass."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "fpga_h4"))

from accuracy_api import lookup_accuracy  # noqa: E402
from candidate_graphs import architectures  # noqa: E402
from common import ARCH, contracts  # noqa: E402
from cost_api import estimate_cost, get_database  # noqa: E402
from joint_search import run_joint_search  # noqa: E402
from pareto import PRIMARY_POLICIES, pareto_max_min  # noqa: E402
from quant_contracts import canonical_profile, quant_contract  # noqa: E402
from search import load_accuracy  # noqa: E402


@pytest.mark.parametrize('field,value', [('latency_us',0.001),('LUT',1),('cycles',1),('SIMD',99),('timing_status','ESTIMATED_TARGET_MISSED')])
def test_legacy_h3_report_and_mapping_are_always_verified(database,field,value):
    from search import validate_h3_row
    base=next(r for pool in database.options.values() for r in pool
        if not json.loads((ROOT/r['case_dir']/'measurement_key.json').read_text()).get('witness_files'))
    assert validate_h3_row(base,database.hashes)[1]==[]
    changed=dict(base,**{field:value})
    checked,errors=validate_h3_row(changed,database.hashes)
    assert errors and not checked['measurement_valid']


def test_accuracy_rejects_wrong_architecture_hash():
    row=lookup_accuracy(ARCH['B1'],'W8A8',architecture_hash='0'*64)
    assert row['status']=='CONTRACT_MISMATCH' and row['Accuracy'] is None


@pytest.mark.parametrize('field,value', [('quant_config_sha256','0'*64),('calibration_manifest_sha256',None),('checkpoint_sha256','0'*64),('seed',99),('int8_mAP50_95',float('nan'))])
def test_accuracy_verifies_retained_provenance(tmp_path,monkeypatch,field,value):
    import accuracy_api as api
    manifest=json.loads(api.PHASE5A_MANIFEST.read_text())
    if field in ('checkpoint_sha256','seed','int8_mAP50_95'):
        manifest['runs'][0][field]=value
    else:
        manifest[field]=value
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    monkeypatch.setattr(api,'PHASE5A_MANIFEST',path)
    row=api.lookup_accuracy(manifest['runs'][0]['architecture_id'],'W8A8')
    assert row['status']=='CONTRACT_MISMATCH' and row['Accuracy'] is None


def test_mixed_accuracy_rejects_profile_config_drift(tmp_path,monkeypatch):
    import csv
    import accuracy_api as api
    with api.PHASE5E_RESULTS.open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    rows[0]['quant_profile_config_sha256']='0'*64
    path=tmp_path/'mixed.csv'
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    monkeypatch.setattr(api,'PHASE5E_RESULTS',path)
    row=api.lookup_accuracy(rows[0]['architecture_id'],'HEAD_W8A16')
    assert row['status']=='CONTRACT_MISMATCH' and row['Accuracy'] is None


@pytest.mark.parametrize('mutation', [dict(network='D'),dict(profile='W8A8'),dict(quant_contract_hash='0'*64),dict(architecture_id=ARCH['D'])])
def test_q_gate_rejects_conflicting_candidate_identity(database,mutation):
    from pareto import eligible
    row=copy.deepcopy(database.estimate_cost('C','HEAD_W16A16','reference_min_latency'))
    row.update(mutation)
    assert not eligible(row) and row['q_status']=='Q_NOT_ADMITTED'


@pytest.mark.parametrize('mapping', [
    dict(policy='reference_min_latency',unknown_hardware_knob=999),
    dict(policy='reference_min_latency',by_node={'head.nonexistent':{'PE':999}}),
    dict(policy='reference_min_latency',by_shape={'unknown':{'PE':999}}),
    dict(policy='reference_min_latency',target_clock_ns=5.9),
    dict(policy='typo',PE=1),
])
def test_mapping_rejects_unconsumed_or_inexact_constraints(database,mapping):
    row=database.estimate_cost('B1','W8A8',mapping)
    assert row['status']=='NEEDS_MEASUREMENT' and row['Latency'] is None and not row['eligible']


def test_nonfinite_measurement_cannot_enter_pareto(database):
    row=copy.deepcopy(database.estimate_cost('B1','W8A8','reference_min_latency'))
    row['Accuracy']=float('nan')
    assert pareto_max_min([row],'Accuracy','Latency')==[]

MEASURED_NETWORKS = ("B1", "C", "D")


@pytest.fixture(scope="module")
def database():
    return get_database()


@pytest.fixture(scope="module")
def joint(database):
    return run_joint_search(database)


def test_quant_contracts_distinguish_head_bitwidths():
    body = contracts()
    uniform = quant_contract("W8A8", body)
    w8a16 = quant_contract("W8A16", body)
    w16a16 = quant_contract("W16A16", body)
    assert canonical_profile("W8A16") == "HEAD_W8A16"
    assert canonical_profile("W16A16") == "HEAD_W16A16"
    assert len({uniform["quant_contract_hash"], w8a16["quant_contract_hash"], w16a16["quant_contract_hash"]}) == 3
    assert w8a16["regions"]["backbone"] == {"weight_bitwidth": 8, "activation_bitwidth": 8}
    assert w8a16["regions"]["neck"] == {"weight_bitwidth": 8, "activation_bitwidth": 8}
    assert w8a16["regions"]["head"] == {"weight_bitwidth": 8, "activation_bitwidth": 16}
    assert w16a16["regions"]["head"] == {"weight_bitwidth": 16, "activation_bitwidth": 16}
    assert quant_contract("HEAD_W8A16", body)["quant_contract_hash"] == w8a16["quant_contract_hash"]


def test_b1_c_d_six_mappings_still_hit(database):
    hits = 0
    for network in MEASURED_NETWORKS:
        for policy in PRIMARY_POLICIES:
            row = estimate_cost(network, "NUMERIC_CONTRACT_V1", policy)
            assert row["status"] == "LOOKED_UP", (network, policy, row.get("reason"))
            assert row["hardware_status"] == "MEASURED"
            assert row["Latency"] is not None and row["DSP"] is not None
            assert row["measurement_scope"] == "OP_LEVEL_HLS"
            hits += 1
    assert hits == 18


def test_unknown_candidate_needs_measurement_and_is_not_predicted():
    row = estimate_cost("not_a_frozen_candidate", "W8A8", "reference_min_latency")
    assert row["status"] == "NEEDS_MEASUREMENT"
    assert row["hardware_status"] == "NEEDS_MEASUREMENT"
    assert row["Latency"] is None and row["DSP"] is None and row["LUT"] is None and row["BRAM"] is None
    assert row["eligible"] is False
    source = "\n".join((ROOT / "tools" / "fpga_h4" / name).read_text(encoding="utf-8")
                        for name in ("cost_api.py", "search.py", "joint_search.py", "accuracy_api.py"))
    for banned in ("def predict", "ignore_hash", "strict=False", "allow_stale", "use_default_cost", "FP32_REFERENCE_ONLY"):
        assert banned not in source


def test_modified_graph_does_not_hit(database):
    original = estimate_cost("B1", "W8A8", "reference_min_latency")
    broken = copy.deepcopy(database.graphs["B1"])
    broken["nodes"][0]["kernel"] = int(broken["nodes"][0]["kernel"]) + 2
    row = estimate_cost(broken, "W8A8", "reference_min_latency")
    assert row["status"] != "LOOKED_UP"
    assert row["hardware_status"] in {"NEEDS_MEASUREMENT", "CONTRACT_MISMATCH"}
    assert row["Latency"] is None
    assert row["Accuracy"] is None
    assert row.get("measurement_ids") != original.get("measurement_ids")


def test_modified_quant_contract_does_not_hit():
    original = estimate_cost("B1", "W8A8", "reference_min_latency")
    row = estimate_cost("B1", {"activation_bits": 4, "weight_bits": 4, "accumulator_bits": 32, "output_bits": 4}, "reference_min_latency")
    assert row["hardware_status"] == "NEEDS_MEASUREMENT"
    assert row["Latency"] is None
    assert row.get("quant_contract_hash") != original["quant_contract_hash"]
    mismatched = lookup_accuracy(original["architecture_id"], "W8A8", contracts(), quant_contract_hash="0" * 64)
    assert mismatched["status"] == "CONTRACT_MISMATCH"
    assert mismatched["Accuracy"] is None


def test_modified_mapping_does_not_hit():
    row = estimate_cost("B1", "W8A8", "not_a_measured_mapping")
    assert row["status"] == "NEEDS_MEASUREMENT"
    assert row["Latency"] is None and row["eligible"] is False
    explicit = estimate_cost("D", "NUMERIC_CONTRACT_V1", {"PE": 99, "SIMD": 99, "buffer_profile": "linebuffer_fifo4"})
    assert explicit["hardware_status"] == "NEEDS_MEASUREMENT"
    assert explicit["DSP"] is None


def test_graph_contract_hash_mismatch_is_not_a_hit(database):
    graph = copy.deepcopy(database.graphs["C"])
    graph["numeric_contract_hash"] = "0" * 64
    row = estimate_cost(graph, "NUMERIC_CONTRACT_V1", "reference_low")
    assert row["status"] == "CONTRACT_MISMATCH"
    assert row["hardware_status"] == "CONTRACT_MISMATCH"
    assert row["Latency"] is None and row["Accuracy"] is None


def test_full_144_search_calls_hardware_agent(joint):
    summary = joint["summary"]
    assert summary["architectures"] == 144
    assert summary["architectures_queried"] == 144
    assert summary["hardware_agent_calls"] == 144 * len(summary["profiles"]) * len(summary["mapping_policies"])
    assert summary["total_candidates"] == summary["hardware_agent_calls"]
    assert set(joint["queried_architecture_ids"]) == set(architectures())
    assert summary["hardware_agent_calls"] == 144 * 3 * 6


def test_unmeasured_hardware_cannot_enter_pareto(joint):
    front = {(row["architecture_id"], row.get("quant_config"), row.get("mapping_policy")) for row in joint["pareto"]}
    assert joint["summary"]["eligible_candidates"] == sum(1 for row in joint["rows"] if row["eligible"])
    assert joint["summary"]["pareto_candidates"] == len(joint["pareto"])
    for row in joint["rows"]:
        if row["hardware_status"] != "MEASURED":
            assert row["eligible"] is False
            assert (row["architecture_id"], row.get("quant_config"), row.get("mapping_policy")) not in front
            assert row["Latency"] is None
    for row in joint["pareto"]:
        assert row["hardware_status"] == "MEASURED"
        assert row["accuracy_status"] == "MEASURED"
        assert row["q_status"] == "Q_ADMITTED"


def test_missing_quantized_accuracy_does_not_use_fp32_or_enter_pareto(joint):
    fp32 = load_accuracy()
    known = {ARCH[name] for name in MEASURED_NETWORKS}
    missing = [aid for aid in architectures() if aid not in known]
    assert missing
    sample = next(row for row in joint["rows"] if row["architecture_id"] == missing[0] and row["profile"] == "W8A8")
    assert sample["accuracy_status"] == "NEEDS_ACCURACY"
    assert sample["Accuracy"] is None
    assert sample["eligible"] is False
    front_ids = {row["architecture_id"] for row in joint["pareto"]}
    assert missing[0] not in front_ids
    for network in MEASURED_NETWORKS:
        row = estimate_cost(network, "W8A8", "reference_min_latency")
        assert row["accuracy_precision"] == "W8A8"
        assert row["quantized_accuracy_bound"] is True
        assert row["Accuracy"] != fp32[network]["mean_mAP50_95"]
    held = [dict(row) for row in joint["pareto"][:1]]
    held.append(dict(architecture_id="synthetic", status="LOOKED_UP", hardware_status="MEASURED",
                      accuracy_status="NEEDS_ACCURACY", quantized_accuracy_bound=False,
                      Accuracy=fp32["B1"]["mean_mAP50_95"], Latency=1.0))
    assert pareto_max_min(held, "Accuracy", "Latency") == [] or all(
        item.get("accuracy_status") == "MEASURED" for item in pareto_max_min(held, "Accuracy", "Latency"))


def test_operator_metrics_are_not_board_or_whole_network(database):
    row = estimate_cost("B1", "W8A8", "reference_min_latency")
    assert row["latency_metric"] == "serial_op_estimate_us"
    assert row["resource_metric"] == "peak_operator"
    assert row["latency"]["board_measured_ms"] is None
    assert row["resource"]["integrated_post_synth"] is None
    assert row["resource"]["board_routed"] is None
    assert row["resource"]["peak_operator"]["dsp"] == row["DSP_peak"]
    assert "board_latency" not in row
    assert "end_to_end_latency" not in row
    assert "measured_network_latency" not in row
    assert "total_network_resource" not in row
    assert "post_route_resource" not in row


def _rows(joint, label, profile):
    found = [row for row in joint["rows"] if row["architecture_id"] == ARCH[label] and row.get("profile") == profile]
    assert len(found) == 6
    return found


def test_frozen_q_gates_final_eligibility(joint):
    summary = joint["summary"]
    assert summary["enumeration_scope"] == "FULL_SPACE_HARDWARE_COVERAGE"
    assert summary["pareto_domain"] == "FROZEN_FINAL_CODESIGN_DOMAIN"
    assert summary["hardware_measured"] == 2592
    assert summary["accuracy_measured"] == 42
    assert summary["final_joint_candidates"] == 30
    assert summary["eligible_candidates"] == 30
    assert len(joint["final_candidates"]) == 30
    assert all(row["q_status"] == "Q_ADMITTED" and row["hardware_status"] == "MEASURED" and row["accuracy_status"] == "MEASURED"
               for row in joint["final_candidates"])

    blocked_c = _rows(joint, "C", "HEAD_W16A16")
    assert all(row["hardware_status"] == "MEASURED" and row["accuracy_status"] == "MEASURED" for row in blocked_c)
    assert all(row["q_status"] == "Q_NOT_ADMITTED" and row["eligible"] is False for row in blocked_c)

    blocked_b1 = _rows(joint, "B1", "HEAD_W16A16")
    assert all(row["q_status"] == "Q_NOT_ADMITTED" and row["eligible"] is False for row in blocked_b1)

    assert all(row["eligible"] is True for row in _rows(joint, "B1", "HEAD_W8A16"))
    assert all(row["eligible"] is True for row in _rows(joint, "C", "W8A8"))
    assert all(row["eligible"] is True for row in _rows(joint, "D", "HEAD_W16A16"))

    front = {(row["architecture_id"], row.get("profile")) for row in joint["pareto"]}
    assert (ARCH["C"], "HEAD_W16A16") not in front
    assert (ARCH["B1"], "HEAD_W16A16") not in front
    for name, front_rows in joint["fronts"].items():
        assert (ARCH["C"], "HEAD_W16A16") not in {(row["architecture_id"], row.get("profile")) for row in front_rows}
        assert all(row["q_status"] == "Q_ADMITTED" for row in front_rows)


def test_w8a16_and_w16a16_are_independent_queries(joint):
    by_profile = {}
    for profile in ("HEAD_W8A16", "HEAD_W16A16"):
        rows = [row for row in joint["rows"] if row.get("profile") == profile and row["architecture_id"] == ARCH["B1"]]
        assert len(rows) == 6
        hashes = {row["quant_contract_hash"] for row in rows}
        assert len(hashes) == 1
        by_profile[profile] = rows
    assert by_profile["HEAD_W8A16"][0]["quant_contract_hash"] != by_profile["HEAD_W16A16"][0]["quant_contract_hash"]
    for left, right in zip(by_profile["HEAD_W8A16"], by_profile["HEAD_W16A16"]):
        if left["hardware_status"] == "MEASURED" and right["hardware_status"] == "MEASURED":
            assert left["measurement_ids"] != right["measurement_ids"]
        else:
            assert left["hardware_status"] == "NEEDS_MEASUREMENT" or right["hardware_status"] == "NEEDS_MEASUREMENT"
            assert left["Latency"] is None or right["Latency"] is None or left["measurement_ids"] != right["measurement_ids"]
