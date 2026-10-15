"""Immutable-source inventory and bounded-tile coverage contract for T6.

The 65 IDs are the already-enumerated W8A8 / HEAD_W8A16 structural keys.
This module does not enumerate a new D01 search domain, admit cost rows,
modify the original workbook, synthesize hardware, or operate a board.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INVENTORY = ROOT / "results/hardware_measurement_pilot_20261012/attempt05/inventory.json"
SOURCE_COVERAGE = ROOT / "results/hardware_measurement_pilot_20261012/buffer_coverage.json"
DEFAULT_OUTPUT = ROOT / "results/hardware_template_coverage_20261013"
TILE_H = 2
TILE_W = 32
OPERATORS = {"conv1x1": "pw1x1", "conv3x3": "conv3x3", "DW3": "dw3", "DW5": "dw5",
             "nearest": "nearest2x", "requant_add": "requant_add",
             "body8_to_head16": "body8_to_head16", "maxpool": "maxpool"}
HISTORICAL_EXCEED = "EXCEEDS_DEVICE_BRAM_UNDER_CURRENT_FULL_FRAME_TEMPLATE"
HISTORICAL_MISSING = "NO_PILOT_OPERATOR_TEMPLATE"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources():
    inventory, coverage = load(SOURCE_INVENTORY), load(SOURCE_COVERAGE)
    rows = inventory["primitive_inventory"]
    if len(rows) != 65 or inventory["unique_primitive_count"] != 65:
        raise ValueError("Current source must have exactly 65 active primitive IDs")
    if inventory["space_version"] != "elastic_neck72_v1" or inventory["architecture_count"] != 72:
        raise ValueError("T6 source must be the current 72-architecture network")
    by_id = {row["primitive_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {r["primitive_id"] for r in coverage["rows"]}:
        raise ValueError("Inventory and historical buffer-coverage IDs differ")
    for row in rows:
        if digest(dict(shape=row["shape"], precision=row["precision"])) != row["primitive_id"]:
            raise ValueError("Source structural/precision identity hash differs")
        if row["family"] not in OPERATORS:
            raise ValueError("Unimplemented family in active source inventory")
        if row["precision"]["input_bitwidth"] not in (8, 16):
            raise ValueError("Only current INT8/INT16 activation profiles are in scope")
    historical_exceeds = sum(r["assessment"] == HISTORICAL_EXCEED for r in coverage["rows"])
    historical_missing = sum(r["assessment"] == HISTORICAL_MISSING for r in coverage["rows"])
    if (historical_exceeds, historical_missing) != (18, 4):
        raise ValueError("Historical 18 + 4 inventory changed")
    return inventory, coverage


def _case(row):
    family, shape, precision = row["family"], deepcopy(row["shape"]), row["precision"]
    regular = family in ("conv1x1", "conv3x3")
    simd = 4 if family == "conv1x1" else 1
    profile = "HEAD_W8A16" if precision["output_bitwidth"] == 16 else "W8A8"
    if profile not in row["profiles"]:
        raise ValueError("Representative precision profile is absent from the primitive")
    return dict(case_id=f"t6_{family.lower()}_{row['primitive_id'][:12]}_tiled",
                primitive_id=row["primitive_id"], family=family, operator=OPERATORS[family], shape=shape,
                source_id=row["source_nodes"][0], source_nodes=list(row["source_nodes"]),
                source_architecture_id=row["architecture_ids"][0],
                precision_profile=profile, profile=profile,
                input_bits=precision["input_bitwidth"], output_bits=precision["output_bitwidth"],
                weight_bits=precision["weight_bitwidth"], accumulator_bits=32, requant_product_bits=64,
                mapping_policy="t6_bounded_spatial_tile_v1",
                mapping=dict(PE=4, SIMD=simd, effective_SIMD=simd if regular else 1,
                             input_banks=simd if regular else 4, input_tile=simd, output_tile=4,
                             TILE_H=TILE_H, TILE_W=TILE_W,
                             buffer_profile="bounded_spatial_tiles_ddr_full_frame_v1",
                             layout="CHW_contiguous_little_endian", status="PROPOSED_UNMEASURED",
                             admitted_existing_mapping=False),
                actual_registered_graph_primitive=True,
                evidence_scope="TEMPLATE_COVERAGE_VALIDATION_NOT_FORMAL_COST_TABLE",
                status="NOT_RUN", formal_cost_eligible=False)


def _representatives(rows, cases):
    def largest(family, predicate=lambda row: True):
        choices = [r for r in rows if r["family"] == family and predicate(r)]
        if not choices:
            raise ValueError(f"No required representative for {family}")
        return max(choices, key=lambda r: (sum(c*h*w for c,h,w in r["shape"]["input_shapes"])
                                           + _elements(r["shape"]["output_shape"]), r["primitive_id"]))

    selections = [
        ("pw_largest_full_frame", largest("conv1x1", lambda r: r["precision"]["input_bitwidth"] == 8)),
        ("pw_mixed_head_largest", largest("conv1x1", lambda r: r["precision"]["input_bitwidth"] == 16)),
        ("dw3_stride2_largest", largest("DW3", lambda r: r["shape"]["stride"] == 2)),
        ("dw3_stride1_largest", largest("DW3", lambda r: r["shape"]["stride"] == 1)),
        ("dw5_stride2_largest", largest("DW5", lambda r: r["shape"]["stride"] == 2)),
        ("dw5_stride1_largest", largest("DW5", lambda r: r["shape"]["stride"] == 1)),
        ("nearest_largest", largest("nearest")),
        ("requant_add_largest", largest("requant_add")),
        ("body8_to_head16_largest", largest("body8_to_head16")),
        ("stem_conv3x3", largest("conv3x3")),
    ]
    for row in sorted((r for r in rows if r["family"] == "maxpool"),
                      key=lambda r: r["shape"]["input_shapes"][0][1], reverse=True):
        selections.append((f"maxpool_input{row['shape']['input_shapes'][0][1]}", row))
    case_by_id = {case["primitive_id"]: case for case in cases}
    result = [dict(role=role, primitive_id=row["primitive_id"],
                   case_id=case_by_id[row["primitive_id"]]["case_id"],
                   family=row["family"], shape=deepcopy(row["shape"]),
                   precision=deepcopy(row["precision"])) for role, row in selections]
    if len(result) != 13 or len({r["case_id"] for r in result}) != 13:
        raise ValueError("Required independent-board representative set must be 13 unique cases")
    return result


def _elements(shape):
    c,h,w = shape
    return c*h*w


def build_inventory():
    """Return old-compatible inventory plus 65 proposed tile cases and 13 IDs."""
    source, coverage = _sources()
    result = deepcopy(source)
    cases = [_case(row) for row in result["primitive_inventory"]]
    representatives = _representatives(result["primitive_inventory"], cases)
    result.update(schema_version="hardware_template_coverage_inventory_v1",
                  counting_scope="current_W8A8_HEAD_W8A16_structural_precision_keys_only",
                  source_inventory_path=str(SOURCE_INVENTORY), source_inventory_sha256=sha(SOURCE_INVENTORY),
                  source_buffer_coverage_path=str(SOURCE_COVERAGE),
                  source_buffer_coverage_sha256=sha(SOURCE_COVERAGE),
                  supersedes="new_separate_T6_implementation_campaign_preserves_pilot_evidence",
                  pilot_case_count=len(cases), pilot_cases=cases, representatives=representatives,
                  representative_case_ids=[r["case_id"] for r in representatives],
                  status="INVENTORY_ONLY_NOT_HARDWARE_EVIDENCE", formal_cost_eligible=False,
                  d01_scope="D01 new-domain definition remains pending; 65 applies only to current domain")
    return result


def _storage(row):
    s,p,f = row["shape"], row["precision"], row["family"]
    ci = s["input_shapes"][0][0]
    co = s["output_shape"][0]
    k,stride = s.get("kernel",1),s.get("stride",1)
    regular = f in ("conv1x1", "conv3x3")
    if f == "nearest":
        ph,pw = (TILE_H+1)//2,(TILE_W+1)//2
    else:
        ph,pw = (TILE_H-1)*stride+k,(TILE_W-1)*stride+k
    input_channels, output_channels = (ci,co) if regular else (4,4)
    input_bits = input_channels*ph*pw*p["input_bitwidth"]
    output_bits = output_channels*TILE_H*TILE_W*p["output_bitwidth"]
    second_bits = 4*TILE_H*TILE_W*p["input_bitwidth"] if f == "requant_add" else 0
    weight_elements = co*ci*k*k if regular else co*k*k if f in ("DW3","DW5") else 0
    weight_bits = weight_elements*(p["weight_bitwidth"] or 0)
    bias_bits = co*32 if weight_elements else 0
    accumulator_bits = 4*32 if regular else 0
    return dict(tile_output_hw=[TILE_H,TILE_W], input_patch_hw=[ph,pw],
                input_tile_channels=input_channels, output_tile_channels=output_channels,
                declared_input_tile_bits=input_bits, declared_output_tile_bits=output_bits,
                declared_second_tile_bits=second_bits, declared_weight_bits=weight_bits,
                declared_bias_bits=bias_bits, declared_parameter_bits=8*32,
                declared_accumulator_register_bits=accumulator_bits,
                total_declared_array_bits=input_bits+output_bits+second_bits+weight_bits+bias_bits+8*32+accumulator_bits,
                evidence_layer="ANALYTIC_DECLARED_ARRAY_BITS_NOT_SYNTHESIS_OR_ROUTE",
                excludes="memory banking fragmentation, FIFOs, AXI shell, scheduler registers and implementation overhead")


def _workbook_d01(path):
    """Read OOXML with stdlib only; preserve the original workbook byte-for-byte."""
    ns = {"x":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    with ZipFile(path) as archive:
        if "xl/sharedStrings.xml" in archive.namelist():
            strings=["".join(s.itertext()) for s in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        else:
            strings=[]
        for member in archive.namelist():
            if not member.startswith("xl/worksheets/sheet") or not member.endswith(".xml"):
                continue
            for row in ET.fromstring(archive.read(member)).findall(".//x:row",ns):
                cells={}
                for cell in row.findall("x:c",ns):
                    typ,value=cell.attrib.get("t",""),cell.find("x:v",ns)
                    if typ=="s": text=strings[int(value.text)] if value is not None else None
                    elif typ=="inlineStr": text="".join(cell.find("x:is",ns).itertext())
                    else: text=value.text if value is not None else None
                    cells[cell.attrib["r"]]=text
                if "D01" in cells.values():
                    rows.append(dict(worksheet_xml=member,row_number=int(row.attrib["r"]),cells=cells))
    return dict(path=str(path),sha256=sha(path),read_only=True,d01_rows=rows,
                original_workbook_modified=False)


def build_contract():
    inv = build_inventory()
    source, old = _sources()
    old_by_id = {r["primitive_id"]:r for r in old["rows"]}
    cases = {r["primitive_id"]:r for r in inv["pilot_cases"]}
    rows=[]
    for row in inv["primitive_inventory"]:
        historical=old_by_id[row["primitive_id"]]
        rows.append(dict(primitive_id=row["primitive_id"],case_id=cases[row["primitive_id"]]["case_id"],
                         family=row["family"],shape=deepcopy(row["shape"]),precision=deepcopy(row["precision"]),
                         source_nodes=list(row["source_nodes"]),profiles=list(row["profiles"]),
                         historical_assessment=historical["assessment"],
                         historical_full_frame_bram18_lower_bound=historical["theoretical_bram18_lower_bound"],
                         historical_missing_template=historical["assessment"]==HISTORICAL_MISSING,
                         historical_existing_template_exceeds=historical["assessment"]==HISTORICAL_EXCEED,
                         full_frame_capacity_exceeds=historical["theoretical_bram18_lower_bound"]>old["available_bram18"],
                         proposed_tiled_storage=_storage(row),
                         template_status="PROPOSED_PENDING_INTEGER_CSIM_HLS_ROUTE_BOARD_VERIFICATION",
                         formal_cost_eligible=False))
    existing_exceeds=[r for r in rows if r["historical_existing_template_exceeds"]]
    missing=[r for r in rows if r["historical_missing_template"]]
    exceeds=[r for r in rows if r["full_frame_capacity_exceeds"]]
    wb_path=Path(source["source_workbook_review"]["path"])
    contract=dict(schema="hardware_template_coverage_contract_v1",space_version=inv["space_version"],
                  architecture_count=72,active_profiles=inv["profiles"],deferred_profiles=inv["deferred_profiles"],
                  primitive_count=65,counts_by_family=dict(Counter(r["family"] for r in rows)),
                  available_bram18=old["available_bram18"],
                  historical_existing_template_exceeds_count=len(existing_exceeds),
                  historical_absent_template_count=len(missing),historical_assessment_set_overlap_count=0,
                  physical_full_frame_capacity_exceeds_count=len(exceeds),
                  absent_and_physical_capacity_overlap_count=sum(r["full_frame_capacity_exceeds"] for r in missing),
                  targeted_primitive_union_count=len({r["primitive_id"] for r in existing_exceeds+missing}),
                  source_inventory=dict(path=str(SOURCE_INVENTORY),sha256=sha(SOURCE_INVENTORY)),
                  source_historical_coverage=dict(path=str(SOURCE_COVERAGE),sha256=sha(SOURCE_COVERAGE)),
                  generator_source=dict(path=str(Path(__file__)),sha256=sha(__file__)),
                  workbook_review=_workbook_d01(wb_path),
                  d01_boundary=dict(status="PENDING_SCOPE_DECISION",
                                    independent_work="Implement generic tiled templates for current shapes; retain configurable CI/CO/H/W/K/STRIDE/PAD/PE/SIMD/TILE_H/TILE_W",
                                    dependent_work="If D01 changes searchable backbone/width/kernel/stride/connections/head, regenerate structural inventory and validate newly introduced shapes/operators",
                                    no_new_domain_claim=True),
                  missing_template_semantics=dict(conv3x3="stem dense 3x3 stride2 padding1 groups1, signed INT8 input/weight/output, ReLU; bias false, BN folded in graph",
                                                  maxpool="three registered signed INT8 2x2 stride2 padding0 max pools, linear output; no requantization or weights"),
                  proposed_template_contract=dict(tile_output_hw=[TILE_H,TILE_W],PE=4,
                                                   SIMD="4 for pointwise; 1 for stem/DW/pool/nearest/add/boundary",
                                                   activation_frames="full CHW tensors in DDR; bounded patches and output tiles in BRAM",
                                                   numeric_rule="existing signed INT8/INT16, INT32 modulo accumulator, INT64 multiply, nearest-even, saturation; fixture parameters are explicit validation values",
                                                   activation_rule="registered relu/relu6/linear applied after requantization; ReLU6 integer clamp supplied explicitly by fixture params",
                                                   trained_weights_or_calibration=False),
                  representative_count=13,representative_case_ids=inv["representative_case_ids"],
                  representatives=inv["representatives"],rows=rows,
                  required_gates=["65 current primitive identities supported by configurable template and full-shape integer/native comparison",
                                  "HLS C simulation and synthesis reports for all 65 configurations",
                                  "13 representative route/bitstream/board bit-exact comparisons and separated tile load/compute/store timing",
                                  "independent verification of source bindings and evidence boundaries"],
                  status="IMPLEMENTATION_CONTRACT_NOT_COMPLETION_EVIDENCE",formal_cost_eligible=False)
    return contract


def _chw(shapes):
    return " + ".join("×".join(map(str,s)) for s in shapes)


def write_contract(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    contract=build_contract()
    json_path,md_path=output/"coverage_contract.json",output/"coverage_contract.md"
    json_path.write_text(json.dumps(contract,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# 待办6：当前65种原语的分块实现合同", "",
           "本文件记录实施范围与待验收条件，不代表综合、布线或上板已经通过。原试做、原工作簿和正式成本表保持原证据边界。", "",
           "当前范围：`elastic_neck72_v1` 的72个结构候选，启用 W8A8 与 HEAD_W8A16，共65个结构/精度原语；HEAD_W16A16仍为D06待定域。", "",
           "原报告的18种是**已有模板且整帧BRAM下界超容量**，4种是**缺模板**，两种assessment互斥；缺模板中stem卷积和最大maxpool也超容量，因此按容量单独统计为20种，缺模板与超容量重叠2种，并集仍为22种。", "",
           "65种全部使用新的有界空间块模板，输出块默认2×32，PE（并行输出通道）为4；PW的SIMD（并行输入通道）为4，其余为1。输入与输出整帧保存在DDR；片上只保存patch（带邻域的局部输入块）、输出块、权重与bias。", "",
           "声明数组位数只核对源码分配规模，未包含分银行碎片、FIFO、AXI外壳及实现开销，不能代替HLS/布局布线资源报告。", "",
           "## D01边界", "",
           "原工作簿D01仍为待冻结：待确认固定骨干或骨干+颈部范围，及允许算子、通道/宽度、kernel（卷积核）、stride（步长）、连接、head（预测头）和版本。当前模板开发可独立进行；D01若改变域，须重新导出原语清单并验证新增形状/算子，65不外推为未知新域全覆盖。", "",
           "## 18种已有模板的整帧容量缺口", "",
           "| 精确primitive_id | 类型 | 输入CHW | 输出CHW | 原整帧BRAM18下界 |", "|---|---|---|---|---:|"]
    for row in contract["rows"]:
        if row["historical_existing_template_exceeds"]:
            lines.append(f"| `{row['primitive_id']}` | {row['family']} | {_chw(row['shape']['input_shapes'])} | {_chw([row['shape']['output_shape']])} | {row['historical_full_frame_bram18_lower_bound']} |")
    lines += ["", "## 4种缺模板", "", "| 精确primitive_id | 类型 | 输入CHW | 输出CHW | 语义 |", "|---|---|---|---|---|"]
    for row in contract["rows"]:
        if row["historical_missing_template"]:
            s=row["shape"]
            meaning=f"K{s['kernel']} / stride{s['stride']} / pad{s['padding']} / {s['activation']}"
            lines.append(f"| `{row['primitive_id']}` | {row['family']} | {_chw(s['input_shapes'])} | {_chw([s['output_shape']])} | {meaning} |")
    lines += ["", "stem为普通卷积，groups=1，图中bias=false、BN已折叠；验证fixture显式提供bias与整数激活上限，不代表训练权重已导出。三个maxpool是有符号INT8窗口最大值，保持原值，不附加重定标。", "", "## 13项完整流程代表", "", "| case_id | 选择理由 | 输入CHW | 输出CHW |", "|---|---|---|---|"]
    for r in contract["representatives"]:
        lines.append(f"| `{r['case_id']}` | {r['role']} | {_chw(r['shape']['input_shapes'])} | {_chw([r['shape']['output_shape']])} |")
    lines += ["", "## 验收门", ""]
    lines += [f"{i}. {gate}" for i,gate in enumerate(contract["required_gates"],1)]
    lines += ["", "全部记录保持`formal_cost_eligible=false`，代表板测不能替代其余原语的正式成本测量。执行结果另存，不回写本合同为已通过。", "", "来源绑定见`coverage_contract.json`；原工作簿仅以OOXML只读读取并记录当前SHA256。", ""]
    md_path.write_text("\n".join(lines),encoding="utf-8")
    return dict(status=contract["status"],primitive_count=65,representative_count=13,
                output=str(output),json_sha256=sha(json_path),markdown_sha256=sha(md_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    print(json.dumps(write_contract(args.output),ensure_ascii=False))


if __name__ == "__main__":
    main()
