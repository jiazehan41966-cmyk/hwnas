"""Read-only graph enumeration and a separate, non-admitted FPGA pilot matrix.

No model weights, dataset, torch, HLS, or board runtime are loaded.  The current
72-candidate graph and the existing three region-specific precision profiles
are the source of the full-library inventory.  Additional operator/precision
probes are explicitly distinguished from primitives in those actual graphs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/hardware_measurement_pilot_20261012/inventory.json"
ACTIVE_PROFILES = ("W8A8", "HEAD_W8A16")
SOURCE_WORKBOOK = Path("E:/1paper汇报/声呐_HWNAS_全链条分阶段梳理_v2_算子上板前置_20261010.xlsx")
SOURCE_WORKBOOK_REVIEWED_SHA256 = "95323753df58cfc09d295659a64cd0abe47d48c6df294805fd99178e0604f4fb"
SOURCE_PATHS = (
    "tools/modnas_elastic/cost_graph.py",
    "tools/modnas_elastic/model.py",
    "tools/fpga_h4/quant_contracts.py",
    "tools/fpga_h2/numeric.py",
    "tools/modnas_comparison/quantization_mixed.py",
    "tools/fpga_h4/search.py",
    "configs/search_space_sonar_v0_1.yaml",
    "configs/search/sonar_joint_v2_elastic_v1.yaml",
    "tools/hardware_measurement_pilot/inventory.py",
)


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def architectures():
    """Same complete 8 + 64 registry as model.enumerate_architectures, without torch."""
    for pan in (False, True):
        for td in itertools.product((3, 5), repeat=3):
            for bottom in itertools.product((3, 5), repeat=3 if pan else 0):
                yield dict(pan=pan, td_kernels=list(td), pan_kernels=list(bottom))


def architecture_id(arch):
    td = "_".join(f"{level}k{kernel}" for level, kernel in
                  zip(("P2", "P3", "P4"), arch["td_kernels"]))
    suffix = "_PAN_" + "_".join(f"{level}k{kernel}" for level, kernel in
                               zip(("P3", "P4", "P5"), arch["pan_kernels"])) if arch["pan"] else ""
    return ("E1_" if arch["pan"] else "E0_") + "TD_" + td + suffix


def family(node):
    if node["op"] == "conv":
        return f"DW{node['kernel']}" if node["groups"] != 1 else f"conv{node['kernel']}x{node['kernel']}"
    return "requant_add" if node["op"] in ("add", "residual_add") else node["op"]


def build_inventory():
    graph_module = _load("tools/modnas_elastic/cost_graph.py", "pilot_cost_graph")
    quant_module = _load("tools/fpga_h4/quant_contracts.py", "pilot_quant_contracts")
    rows = {}
    candidate_graphs = {}
    source_graphs = []
    occurrence_counts = Counter()

    def record(node, profile, aid, precision):
        shape = graph_module.primitive_shape(node)
        # Residual and lateral addition have identical integer arithmetic when
        # shape and explicit input/output numeric contracts agree.
        if node["op"] in ("add", "residual_add"):
            shape.update(op="requant_add", operator_type="requant_add")
        if node["op"] != "conv":
            precision["weight_bitwidth"] = None
        identity = dict(shape=shape, precision=precision)
        key = digest(identity)
        group = family(node)
        if key not in rows:
            rows[key] = dict(primitive_id=key, family=group, **identity,
                             source_nodes=set(), architecture_ids=set(), profiles=set(), profile_occurrences=Counter(),
                             occurrences=0, status="NEEDS_MEASUREMENT", formal_cost_eligible=False)
        row = rows[key]
        row["source_nodes"].add(node["id"])
        row["architecture_ids"].add(aid)
        row["profiles"].add(profile)
        row["occurrences"] += 1
        row["profile_occurrences"][profile] += 1
        occurrence_counts[(group, profile)] += 1

    for arch in architectures():
        aid = architecture_id(arch)
        graph = graph_module.build_cost_graph(arch, num_classes=2)
        candidate_graphs[aid] = graph
        source_graphs.append(dict(architecture_id=aid, architecture=arch,
                                 cost_graph_sha256=graph["cost_graph_sha256"], nodes=len(graph["nodes"])))
        for profile in quant_module.PROFILES:
            for node in graph["nodes"]:
                precision = quant_module.node_precision(profile, node["id"])
                record(node, profile, aid, precision)
                if node["id"].startswith("head.") and precision["input_bitwidth"] == 16:
                    shape = node["input_shapes"][0]
                    boundary = dict(id=node["id"] + ".input_requant", op="body8_to_head16",
                                    operator_type="body8_to_head16", input_shapes=[shape],
                                    output_shape=shape, activation="linear")
                    record(boundary, profile, aid, dict(input_bitwidth=8, weight_bitwidth=None,
                                                       accumulator_bitwidth=32, output_bitwidth=16))

    primitive_rows, deferred_rows = [], []
    for _, value in sorted(rows.items()):
        row = deepcopy(value)
        for field in ("source_nodes", "architecture_ids", "profiles"):
            row[field] = sorted(row[field])
        row["profile_occurrences"] = dict(row["profile_occurrences"])
        if any(profile in ACTIVE_PROFILES for profile in row["profiles"]):
            row["registered_profiles"] = list(row["profiles"])
            row["profiles"] = [profile for profile in row["profiles"] if profile in ACTIVE_PROFILES]
            row["profile_occurrences"] = {profile: count for profile, count in row["profile_occurrences"].items()
                                           if profile in ACTIVE_PROFILES}
            row["occurrences"] = sum(row["profile_occurrences"].values())
            primitive_rows.append(row)
        else:
            row["status"] = "DEFERRED_D06_NOT_ENABLED_FOR_PILOT"
            deferred_rows.append(row)
    counts = Counter(row["family"] for row in primitive_rows)
    counts_precision = Counter((row["family"], row["precision"]["input_bitwidth"],
                                row["precision"]["weight_bitwidth"], row["precision"]["output_bitwidth"])
                               for row in primitive_rows)
    aid3 = "E0_TD_P2k3_P3k3_P4k3"
    aid5 = "E0_TD_P2k3_P3k3_P4k5"
    selections = (
        ("pw1x1", aid3, "head.towers.2.0"),
        ("dw3", aid3, "neck.top_down_blocks.P4.dw_conv"),
        ("dw5", aid5, "neck.top_down_blocks.P4.dw_conv"),
        ("nearest2x", aid3, "neck.top_down.P4.nearest"),
        ("requant_add", aid3, "neck.top_down.P4.add"),
    )
    representatives = []
    for name, aid, path in selections:
        node = next(n for n in candidate_graphs[aid]["nodes"] if n["id"] == path)
        pw = name == "pw1x1"
        dw = name.startswith("dw")
        mapping = {
            "reference_low": dict(PE=1 if pw else 2, SIMD=2 if pw else 1),
            "reference_high": dict(PE=3 if pw else 4, SIMD=4 if pw else 1),
        }
        for spec in mapping.values():
            spec.update(input_tile=spec["SIMD"], output_tile=spec["PE"],
                        status="PROPOSED_UNMEASURED", admitted_existing_mapping=False)
        representatives.append(dict(name=name, source_architecture_id=aid, source_id=path,
                                    real_layer=True, shape=graph_module.primitive_shape(node),
                                    mappings=mapping, structural_graph_sha256=candidate_graphs[aid]["cost_graph_sha256"]))
    head = representatives[0]
    boundary_shape = dict(op="body8_to_head16", operator_type="body8_to_head16",
                          input_shapes=deepcopy(head["shape"]["input_shapes"]),
                          output_shape=deepcopy(head["shape"]["input_shapes"][0]), activation="linear")
    representatives.append(dict(
        name="body8_to_head16", source_architecture_id=aid3, source_id="head.towers.2.0.input_requant",
        real_layer=True, shape=boundary_shape, source_edge_shape_witness="head.towers.2.0",
        reason="Explicit INT8 body to HEAD_W8A16 input conversion in the mixed integer graph",
        mappings=deepcopy(representatives[-1]["mappings"])))
    cases = []
    for representative in representatives:
        if representative["name"] == "body8_to_head16":
            precisions = [("i8o16", 8, None, 16, "HEAD_W8A16")]
        elif representative["name"] == "pw1x1":
            precisions = [("int8", 8, 8, 8, "W8A8"), ("w8a16", 16, 8, 16, "HEAD_W8A16")]
        else:
            precisions = [("int8", 8, 8 if representative["name"].startswith("dw") else None, 8, "W8A8")]
        for precision_name, input_bits, weight_bits, output_bits, profile in precisions:
            for mapping, spec in representative["mappings"].items():
                case_id = (f"{representative['name']}_{mapping}" if representative["name"] == "body8_to_head16"
                           else f"{representative['name']}_{precision_name}_{mapping}")
                cases.append(dict(case_id=case_id,
                                  operator=representative["name"], shape=representative["shape"],
                                  source_id=representative["source_id"],
                                  source_architecture_id=representative["source_architecture_id"],
                                  precision_profile=profile,
                                  input_bits=input_bits, output_bits=output_bits, weight_bits=weight_bits,
                                  accumulator_bits=32, requant_product_bits=64,
                                  mapping_policy=mapping, mapping=deepcopy(spec),
                                  actual_registered_graph_primitive=True,
                                  evidence_scope="PILOT_ONLY_NEVER_FORMAL_COST_TABLE",
                                  status="NOT_RUN", formal_cost_eligible=False))
    assert len(source_graphs) == 72 and len(cases) == 14
    assert not any(row["family"] == "requant_concat" for row in primitive_rows)
    return dict(
        schema_version="hardware_measurement_pilot_inventory_v2", space_version="elastic_neck72_v1",
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS},
        architecture_count=72, profiles=list(ACTIVE_PROFILES), registered_profiles=list(quant_module.PROFILES),
        deferred_profiles=["HEAD_W16A16"], num_classes=2,
        source_workbook_review=dict(path=SOURCE_WORKBOOK.as_posix(), reviewed_sha256=SOURCE_WORKBOOK_REVIEWED_SHA256,
            sheet="先确定事项", D06_row=11, D07_row=12, D09_row=14,
            final_values_F_G_H="EMPTY_AT_REVIEW", decision_status_I="待冻结", original_workbook_modified=False,
            source_of_pilot_precision_choice="current user instruction: W8A8 body and W8A16 head+boundary; W16 deferred pending D06"),
        input_chw=[1, 640, 640], unique_primitive_count=len(primitive_rows),
        counts_by_family=dict(sorted(counts.items())),
        counts_by_family_precision=[dict(family=k[0], input_bits=k[1], weight_bits=k[2], output_bits=k[3], count=v)
                                    for k, v in sorted(counts_precision.items(), key=lambda pair: str(pair[0]))],
        occurrences_by_family_profile=[dict(family=k[0], profile=k[1], count=v)
                                       for k, v in sorted(occurrence_counts.items()) if k[1] in ACTIVE_PROFILES],
        conditional_all_registered_profiles_unique_primitive_count=len(primitive_rows)+len(deferred_rows),
        deferred_primitive_inventory=deferred_rows,
        nominal_full_measurement_items_by_physical_configurations_per_primitive={str(n): n * len(primitive_rows) for n in (2, 3, 6)},
        physical_measurement_item_count=None,
        physical_measurement_count_status="PENDING_EXACT_PHYSICAL_CONFIGURATION_BINDING_AND_DEDUPLICATION",
        counting_scope="deduplicated exact structural primitives and bitwidths across 72 architectures x active W8A8/HEAD_W8A16 profiles; no topology multiplication; HEAD_W16A16 conditional on D06",
        counting_limits=["Different numeric scales, source implementations, interfaces, clocks, or tiles may create additional measurement identities",
                         "Boundary shape-identical HEAD_W8A16/HEAD_W16A16 conversions deduplicate only when generated boundary implementation/interface are identical",
                         "DMA, repack, shell, CPU decode, whole-network route and integration measurements are not included"],
        mapping_note="Existing reference_low/high and four min-resource/latency names are selectors and can reuse the same measured row; unique primitives x 2/3/6 is nominal only if every primitive has that many distinct physical configurations. Exact count requires binding and deduplicating PE/SIMD/tiles/buffer/interface/clock/source/precision. Pilot values here are newly proposed, unmeasured configurations",
        int16_note="Current pilot uses W8A16 head only plus explicit I8-to-O16 boundary; DW3/DW5/nearest/add remain INT8. W16 weights are deferred until D06 confirmation. No all-body INT16 probe is active",
        concat_note="Concatenation is absent from the registered network and is not enabled by the reviewed workbook; excluded from default pilot pending explicit scope decision",
        supersedes="attempt01 24-case local software probes; preserve their native evidence without promoting W16/INT16-body/concat to current hardware scope",
        pilot_case_count=len(cases), representatives=representatives, pilot_cases=cases,
        primitive_inventory=primitive_rows, source_graphs=source_graphs,
        status="INVENTORY_ONLY_NOT_HARDWARE_EVIDENCE", formal_cost_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inventory = build_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: inventory[k] for k in ("architecture_count", "unique_primitive_count", "counts_by_family", "pilot_case_count")}, indent=2))


if __name__ == "__main__":
    main()
