"""Summarize the 65 source-bound HLS resource reports; no hardware actions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/hardware_template_coverage_20261013"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    attempt = BASE / "attempt01"
    pointer = json.loads((attempt / "remote_evidence/latest_fetch.json").read_text())
    snapshot = Path(pointer["directory"]).resolve()
    if not snapshot.is_relative_to((attempt / "remote_evidence").resolve()):
        raise ValueError("snapshot escapes this attempt")
    archive = snapshot / Path(pointer["archive"]).name
    if sha(archive) != pointer["sha256"]:
        raise ValueError("snapshot archive changed")
    manifest = json.loads((snapshot / "fetch_manifest.json").read_text())["files"]
    contract = json.loads((BASE / "coverage_contract.json").read_text(encoding="utf-8"))
    expected = {row["case_id"]: row for row in contract["rows"]}
    rows = []
    for path in sorted(snapshot.glob("results/*/hls/run001/pilot_kernel_csynth.xml")):
        name = path.relative_to(snapshot).as_posix()
        if sha(path) != manifest[name]:
            raise ValueError("HLS report changed: " + name)
        case_id = path.parents[2].name
        source = expected[case_id]
        xml = ET.parse(path).getroot()
        if xml.findtext("./UserAssignments/Part") != "xczu7ev-ffvc1156-2-e":
            raise ValueError("unexpected HLS target part")
        used = {node.tag: int(node.text) for node in xml.findall("./AreaEstimates/Resources/*")}
        available = {node.tag: int(node.text) for node in xml.findall("./AreaEstimates/AvailableResources/*")}
        if available != dict(BRAM_18K=624, DSP=1728, FF=460800, LUT=230400, URAM=96):
            raise ValueError("unexpected available resources")
        if set(used) != set(available) or any(used[key] > available[key] for key in available):
            raise ValueError("resource gate failed: " + case_id)
        rows.append(dict(case_id=case_id, family=source["family"],
            historical_full_frame_bram18_lower_bound=source["historical_full_frame_bram18_lower_bound"],
            historical_existing_template_exceeds=source["historical_existing_template_exceeds"],
            historical_missing_template=source["historical_missing_template"],
            report_path=name, report_sha256=sha(path), used=used, available=available))
    if len(rows) != 65 or {row["case_id"] for row in rows} != set(expected):
        raise ValueError("exactly 65 unique current primitive reports required")
    maxima = {key: max(row["used"][key] for row in rows) for key in rows[0]["available"]}
    now = datetime.now(timezone.utc)
    output = attempt / "resource_summary" / ("run_" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="t6_actual_hls_resources_v1", created_utc=now.isoformat(),
        status="PASS_65_EXACT_HLS_RESOURCE_REPORTS", evidence_layer="HLS_ESTIMATE_NOT_ROUTE_OR_BOARD",
        source_manifest_sha256=sha(attempt / "source_manifest.json"),
        contract_sha256=sha(BASE / "coverage_contract.json"),
        fetch_manifest_sha256=sha(snapshot / "fetch_manifest.json"),
        snapshot_directory=str(snapshot), archive_sha256=pointer["sha256"],
        generator_sha256=sha(__file__), primitive_count=65, maxima=maxima, rows=rows,
        formal_cost_eligible=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 当前65项的实际HLS资源核对", "",
        "65项均读取Vitis HLS生成的核心综合报告，并核对报告哈希、精确case集合及目标器件。全部满足HLS容量门。",
        "HLS资源估计不能替代整套外壳的布局布线资源或板上验证；本表不建立性能改善或正式成本资格。", "",
        f"全65项最大BRAM18={maxima['BRAM_18K']}/624，DSP={maxima['DSP']}/1728，LUT={maxima['LUT']}/230400，FF={maxima['FF']}/460800，URAM={maxima['URAM']}/96。各列最大值可来自不同配置。", "",
        "下表仅列原18项整帧超容量和4项缺模板的并集。旧列是整帧存储的容量下界，新列是实际HLS报告，证据层不同。", "",
        "| case_id | 类型 | 原缺口 | 原整帧BRAM18下界 | 新HLS BRAM18 | 新HLS DSP | 新HLS LUT |", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["historical_existing_template_exceeds"] or row["historical_missing_template"]:
            reason = "缺模板" if row["historical_missing_template"] else "已有模板整帧超容量"
            used = row["used"]
            lines.append(f"| `{row['case_id']}` | {row['family']} | {reason} | {row['historical_full_frame_bram18_lower_bound']} | {used['BRAM_18K']} | {used['DSP']} | {used['LUT']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), primitive_count=65, maxima=maxima)))


if __name__ == "__main__":
    main()
