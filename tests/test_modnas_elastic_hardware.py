"""NEW-space cost topology, inactive probability and hardware evidence gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_elastic.cost_graph import (DEFAULT_PRECISION, build_cost_graph, canonical_hash,
                                            primitive_shape)
from tools.modnas_elastic.hardware import (BLOCKED, METRICS, PART, DifferentiableParamsTable,
                                           ExternalMeasuredCostTable, PrimitiveEvidenceCache,
                                           architecture_probabilities, enumerate_architectures)
from tools.modnas_elastic.model import (StaticElasticNeckDetector, architecture_hash,
                                        enumerate_architectures as model_architectures, one_hot_weights)


def test_all_72_cost_graphs_match_actual_640_meta_model_parameters_and_convolutions():
    assert enumerate_architectures() == list(model_architectures())
    for architecture in enumerate_architectures():
        graph = build_cost_graph(architecture)
        with torch.device("meta"):
            model = StaticElasticNeckDetector(architecture).eval()
        assert graph["architecture_sha256"] == architecture_hash(architecture)
        assert graph["Params"] == sum(parameter.numel() for parameter in model.parameters())
        observed, handles = {}, []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(
                    lambda conv, inputs, output, name=name: observed.__setitem__(
                        name, (list(inputs[0].shape[1:]), list(output.shape[1:]), conv.kernel_size[0], conv.groups))))
        outputs = model(torch.empty(1, 1, 640, 640, device="meta"))
        for handle in handles:
            handle.remove()
        convolutions = {node["id"]: node for node in graph["nodes"] if node["op"] == "conv"}
        assert set(convolutions) == set(observed)
        for name, (input_shape, output_shape, kernel, groups) in observed.items():
            node = convolutions[name]
            assert node["input_shapes"] == [input_shape]
            assert node["output_shape"] == output_shape
            assert node["kernel"] == kernel and node["groups"] == groups
        for level, output in zip(("P2", "P3", "P4", "P5"), outputs["predictions"]):
            assert graph["edges"][graph["outputs"][level]] == list(output.shape[1:])
        counts = {op: sum(node["op"] == op for node in graph["nodes"])
                  for op in ("nearest", "add", "residual_add", "maxpool")}
        assert counts == dict(nearest=3, add=6 if architecture["pan"] else 3,
                              residual_add=6 if architecture["pan"] else 3,
                              maxpool=3 if architecture["pan"] else 0)


def test_class_count_changes_full_prediction_params_and_head_shape():
    architecture = enumerate_architectures()[0]
    two, five = build_cost_graph(architecture), build_cost_graph(architecture, num_classes=5)
    with torch.device("meta"):
        model = StaticElasticNeckDetector(architecture, num_classes=5)
    assert five["Params"] == sum(parameter.numel() for parameter in model.parameters())
    assert five["Params"] - two["Params"] == 4 * 49 * 3 * 3
    assert all(five["edges"][edge][0] == 30 for edge in five["outputs"].values())
    assert five["cost_graph_sha256"] != two["cost_graph_sha256"]


def test_onehot_exact_params_and_mass_inactive_pan_merge_for_every_candidate():
    table = DifferentiableParamsTable()
    assert table.hardware_search_status == BLOCKED
    assert table.measurement_scope == "SOFTWARE_PARAMETER_COUNT_NOT_HARDWARE"
    for index, architecture in enumerate(enumerate_architectures()):
        weights = one_hot_weights(architecture, dtype=torch.float64)
        probabilities = architecture_probabilities(weights)
        assert probabilities[index] == 1 and probabilities.sum() == 1
        assert torch.count_nonzero(probabilities) == 1
        assert table(weights).item() == table.values[index].item()
    weights = one_hot_weights(enumerate_architectures()[0], dtype=torch.float64)
    reference = architecture_probabilities(weights)
    weights["pan_kernel_weights"] = torch.tensor([[.1, .9], [.9, .1], [.5, .5]], dtype=torch.float64, requires_grad=True)
    assert torch.equal(reference, architecture_probabilities(weights))
    table(weights).backward()
    assert torch.equal(weights["pan_kernel_weights"].grad, torch.zeros(3, 2, dtype=torch.float64))


def test_differentiable_params_gradcheck_through_preferences_architecture_weights():
    table = DifferentiableParamsTable()
    logits = torch.tensor([.2, -.4, .3, -.1, .5, -.2, .1, .7, -.3, .4, -.1, .6, -.2, .8],
                          dtype=torch.float64, requires_grad=True)
    def objective(value):
        return table(dict(pan_weights=value[:2].softmax(-1),
                          td_kernel_weights=value[2:8].reshape(3, 2).softmax(-1),
                          pan_kernel_weights=value[8:].reshape(3, 2).softmax(-1))) / 1e6
    assert torch.autograd.gradcheck(objective, (logits,), eps=1e-6, atol=1e-5)
    objective(logits).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("mutation", ["negative", "nan", "unnormalized"])
def test_probabilities_reject_invalid_mass(mutation):
    weights = one_hot_weights(enumerate_architectures()[0])
    weights["pan_weights"] = {"negative": torch.tensor([1.1, -.1]), "nan": torch.tensor([float("nan"), 0.]),
                              "unnormalized": torch.tensor([.2, .2])}[mutation]
    with pytest.raises(ValueError):
        architecture_probabilities(weights)


def test_old_h3_exact_primitive_cache_never_fills_new_neck_costs():
    database = PrimitiveEvidenceCache()
    graph = build_cost_graph(enumerate_architectures()[0])
    stem = graph["nodes"][0]
    found = database.query(stem)
    assert found["status"] == "EXACT_PRIMITIVE_HLS_EVIDENCE"
    assert len(found["measurement"]["source_report_sha256"]) == 64
    assert database.query(stem) == found
    missing = next(node for node in graph["nodes"] if node["id"] == "neck.lateral_convs.P2.0")
    assert database.query(missing)["status"] == "MISSING"
    nearest = next(node for node in graph["nodes"] if node["op"] == "nearest")
    assert database.query(nearest)["measurement"] is None
    assert database.counts["total_node_queries"] == 4
    assert database.counts["distinct_primitive_queries"] == 3
    assert database.counts["cache_hits"] == 1


def test_hardware_constructor_cannot_bypass_source_gate():
    with pytest.raises(ValueError, match="from_json"):
        ExternalMeasuredCostTable(torch.zeros(72, 4), {})


def _write_fixture_table(tmp_path):
    """Synthetic TEST FILES validate loader behavior, never experiment evidence."""
    def write(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    report_sha = write("TEST_csynth.xml", b"<profile><ReportVersion><Version>2022.2</Version></ReportVersion>"
                       b"<UserAssignments><Part>xczu7ev-ffvc1156-2-e</Part><TargetClockPeriod>5</TargetClockPeriod></UserAssignments>"
                       b"<SummaryOfTimingAnalysis><EstimatedClockPeriod>4</EstimatedClockPeriod></SummaryOfTimingAnalysis>"
                       b"<SummaryOfOverallLatency><Worst-caseLatency>200</Worst-caseLatency></SummaryOfOverallLatency>"
                       b"<AreaEstimates><Resources><LUT>10</LUT><DSP>2</DSP><BRAM_18K>1</BRAM_18K></Resources></AreaEstimates></profile>")
    witness_sha = write("TEST_witness.bin", b"synthetic_test_fixture_not_experiment")
    source_sha = write("TEST_kernel.cpp", b"// synthetic TEST fixture only\n")
    source_files = {"TEST_kernel.cpp": source_sha}
    architectures = enumerate_architectures()
    contract = dict(status="NEW_ELASTIC_INTEGER_CONTRACT_VERIFIED", space_sha256=canonical_hash(architectures),
                    precision=DEFAULT_PRECISION, num_classes=2,
                    numeric_contract_hash="1" * 64, interface_contract_hash="2" * 64)
    contract_sha = write("TEST_contract.json", json.dumps(contract).encode())
    table = dict(schema_version=1, space_sha256=canonical_hash(architectures), num_classes=2,
                 precision=DEFAULT_PRECISION, mapping_policy="reference_min_latency", measurement_scope="OP_LEVEL_HLS",
                 quant_contract=dict(path="TEST_contract.json", sha256=contract_sha), rows=[])
    for architecture in architectures:
        graph = build_cost_graph(architecture)
        measurements = []
        for node in graph["nodes"]:
            measurements.append(dict(node_id=node["id"], primitive=primitive_shape(node),
                                     primitive_shape_sha256=node["primitive_shape_sha256"], precision=DEFAULT_PRECISION,
                                     part=PART, tool_version="2022.2", target_clock_ns=5, PE=1, SIMD=1,
                                     buffer_profile="TEST_fixture", csim_status="PASS_BIT_EXACT", csynth_status="PASS",
                                     numeric_contract_hash=contract["numeric_contract_hash"],
                                     interface_contract_hash=contract["interface_contract_hash"], source_files=source_files,
                                     source_hash=canonical_hash(source_files), source_report_path="TEST_csynth.xml",
                                     source_report_sha256=report_sha, csim_reference_path="TEST_witness.bin",
                                     csim_reference_sha256=witness_sha, csim_output_path="TEST_witness.bin",
                                     csim_output_sha256=witness_sha, Latency_us=1., LUT=10, DSP=2, BRAM18=1))
        table["rows"].append(dict(architecture_sha256=graph["architecture_sha256"],
                                  cost_graph_sha256=graph["cost_graph_sha256"], measurements=measurements,
                                  Latency_us=float(len(measurements)), LUT=10, DSP=2, BRAM18=1))
    path = tmp_path / "TEST_table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path, table


def test_complete_hardware_loader_exact_expectation_and_budget_units(tmp_path):
    path, fixture = _write_fixture_table(tmp_path)
    table = ExternalMeasuredCostTable.from_json(path)
    weights = one_hot_weights(enumerate_architectures()[-1], dtype=torch.float64)
    expectation = table(weights)
    assert {name: value.item() for name, value in expectation.items()} == table.exact(enumerate_architectures()[-1])
    budgets = dict(Latency_us=100., LUT=20., DSP=4., BRAM18=2.)
    for name, value in table(weights, budgets).items():
        assert value.item() == expectation[name].item() / budgets[name]
    assert table.metadata["measurement_scope"] == "OP_LEVEL_HLS"
    assert table.metadata["board_deployment"] == "NOT_CLAIMED"
    assert "replacement_not_author" in table.adaptation


@pytest.mark.parametrize("mutation", ["partial72", "aggregate", "shape", "source_hash", "arch_hash", "missing_residual", "report_platform"])
def test_external_table_rejects_incomplete_or_unbound_evidence(tmp_path, mutation):
    path, table = _write_fixture_table(tmp_path)
    if mutation == "partial72":
        table["rows"].pop()
    elif mutation == "aggregate":
        table["rows"][0]["LUT"] += 1
    elif mutation == "shape":
        table["rows"][0]["measurements"][0]["primitive"]["Cin"] += 1
    elif mutation == "source_hash":
        table["rows"][0]["measurements"][0]["source_hash"] = "0" * 64
    elif mutation == "arch_hash":
        table["rows"][0]["architecture_sha256"] = "0" * 64
    elif mutation == "missing_residual":
        table["rows"][0]["measurements"] = [item for item in table["rows"][0]["measurements"]
                                                 if "residual_add" not in item["node_id"]]
    elif mutation == "report_platform":
        report = tmp_path / "TEST_csynth.xml"
        report.write_bytes(report.read_bytes().replace(b"xczu7ev-ffvc1156-2-e", b"wrongpart"))
        digest = hashlib.sha256(report.read_bytes()).hexdigest()
        for row in table["rows"]:
            for measurement in row["measurements"]:
                measurement["source_report_sha256"] = digest
    path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(ValueError):
        ExternalMeasuredCostTable.from_json(path)
