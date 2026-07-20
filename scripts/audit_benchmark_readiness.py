#!/usr/bin/env python3
"""Audit every formal acceptance item for the paper-benchmark campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.archive import CampaignPaths, sha256_file, write_json
from hwnas_fpga.benchmarks.readiness import build_readiness_report, make_requirement


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _resolve_evidence_path(project_root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (project_root / path).resolve()


def _tracked_patch_integrity(project_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    code = dict((summary.get("provenance") or {}).get("code") or {})
    dirty = bool(code.get("dirty"))
    patch = dict(code.get("tracked_patch") or {})
    patch_path = _resolve_evidence_path(project_root, patch.get("path"))
    expected_sha = str(patch.get("sha256") or "")
    observed_sha = sha256_file(patch_path) if patch_path and patch_path.is_file() else None
    valid = bool(
        (not dirty and not patch)
        or (
            dirty
            and patch_path is not None
            and patch_path.is_file()
            and len(expected_sha) == 64
            and observed_sha == expected_sha
        )
    )
    return {
        "dirty": dirty,
        "path": str(patch_path) if patch_path else None,
        "expected_sha256": expected_sha or None,
        "observed_sha256": observed_sha,
        "valid": valid,
    }


def _source_freeze_integrity(
    project_root: Path,
    source_freeze: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = _resolve_evidence_path(project_root, source_freeze.get("path"))
    archive_path = _resolve_evidence_path(project_root, source_freeze.get("archive_path"))
    expected_manifest_sha = str(source_freeze.get("manifest_sha256") or "")
    expected_archive_sha = str(source_freeze.get("archive_sha256") or "")
    observed_manifest_sha = (
        sha256_file(manifest_path) if manifest_path and manifest_path.is_file() else None
    )
    observed_archive_sha = (
        sha256_file(archive_path) if archive_path and archive_path.is_file() else None
    )
    valid = bool(
        source_freeze.get("verification_status") == "PASS"
        and len(expected_manifest_sha) == 64
        and len(expected_archive_sha) == 64
        and observed_manifest_sha == expected_manifest_sha
        and observed_archive_sha == expected_archive_sha
    )
    return {
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_expected_sha256": expected_manifest_sha or None,
        "manifest_observed_sha256": observed_manifest_sha,
        "archive_path": str(archive_path) if archive_path else None,
        "archive_expected_sha256": expected_archive_sha or None,
        "archive_observed_sha256": observed_archive_sha,
        "valid": valid,
    }


def _formal_classification_runs(
    project_root: Path,
    *,
    task: str,
    methods: list[dict[str, Any]],
    expected_pairs: set[tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    rows = {
        str(method["method_id"]): {
            "method_id": str(method["method_id"]),
            "result_dir": str((project_root / str(method["result_dir"])).resolve()),
            "claimable": False,
            "source_freeze_verified": False,
            "source_freeze_integrity": False,
            "tracked_patch_integrity": False,
            "observed_pairs": [],
            "prediction_files": 0,
            "run_paths": [],
        }
        for method in methods
    }
    for method in methods:
        method_id = str(method["method_id"])
        run_dir = (project_root / str(method["result_dir"])).resolve()
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        config = dict(manifest.get("immutable_config") or {})
        if str(config.get("method_id") or "") != method_id or str(config.get("task")) != task:
            continue
        summary_path = manifest_path.with_name("protocol_summary.json")
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        claimability = dict(summary.get("claimability") or {})
        pairs = {
            (int(item["fold"]), int(item["seed"]))
            for item in manifest.get("completed_pairs") or []
        }
        prediction_files = list(manifest_path.parent.glob("outer_predictions_*.jsonl"))
        row = rows[method_id]
        row["observed_pairs"] = [
            {"fold": fold, "seed": seed} for fold, seed in sorted(pairs)
        ]
        row["prediction_files"] = len(prediction_files)
        row["run_paths"].append(str(manifest_path.parent.resolve()))
        source_freeze = dict(config.get("source_freeze") or {})
        freeze_integrity = _source_freeze_integrity(project_root, source_freeze)
        patch_integrity = _tracked_patch_integrity(project_root, summary)
        row["source_freeze_verified"] = freeze_integrity["valid"]
        row["source_freeze_integrity"] = freeze_integrity["valid"]
        row["tracked_patch_integrity"] = patch_integrity["valid"]
        row["source_freeze_evidence"] = freeze_integrity
        row["tracked_patch_evidence"] = patch_integrity
        row["claimable"] = bool(
            claimability.get("claimable")
            and claimability.get("protocol_complete")
            and claimability.get("source_freeze_verified")
            and row["source_freeze_verified"]
            and row["tracked_patch_integrity"]
            and pairs == expected_pairs
            and len(prediction_files) == len(expected_pairs)
        )
    return rows


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Benchmark Formal Readiness",
        "",
        f"- Campaign: `{report['campaign_id']}`",
        f"- Status: `{report['status']}`",
        f"- Passed: `{report['passed_count']}/{report['requirement_count']}`",
        "",
        "| requirement | status | observed | required | blockers |",
        "|---|---|---|---|---|",
    ]
    for row in report["requirements"]:
        observed = json.dumps(row["observed"], ensure_ascii=False, separators=(",", ":"))
        required = json.dumps(row["required"], ensure_ascii=False, separators=(",", ":"))
        blockers = "; ".join(row["blockers"]) or "-"
        lines.append(
            f"| `{row['requirement_id']}` | `{row['status']}` | {observed} | "
            f"{required} | {blockers} |"
        )
    lines.extend(["", f"> {report['boundary']}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_zh(path: Path, report: dict[str, Any], source_path: Path) -> None:
    labels = {
        "R1_sources": "论文、代码版本与许可证/对应关系",
        "R2_environments": "论文专用隔离环境",
        "R3_closed_set": "闭集四方法统一协议",
        "R4_open_long_tail": "开放集与长尾三方法统一协议",
        "R5_nas_formal": "四种 NAS 策略正式比较",
        "R6_hls_proxy": "HLS/route 代理可靠性",
        "R7_board_power": "AV7K325 板级与功耗",
        "R8_gates": "测量优先门禁",
        "R9_archive": "T1-T9 与 F1-F12 可重建归档",
    }
    blocker_zh = {
        "main-paper source audit cards are incomplete": "五篇主论文的来源审计卡不完整",
        "six paper-specific environments are not locked and verified": "六个论文专用环境尚未全部锁定并验证",
        "closed-set 15-unit formal runs are incomplete": "闭集四方法的 15 单元正式结果不完整",
        "open-set 15-unit formal runs are incomplete": "开放集三方法的 15 单元正式结果不完整",
        "formal matched NAS comparison is absent or incomplete": "等预算 NAS 正式比较缺失或不完整",
        "formal HLS/route sample threshold and G2 are not satisfied": "HLS/route 正式样本阈值和 G2 尚未满足",
        "AV7K325 three-candidate board/power evidence is incomplete": "AV7K325 三候选板级/功耗证据不完整",
        "measurement-first gates do not permit the full formal campaign": "测量优先门禁尚不允许完整正式 campaign",
        "formal tables/figures are still pending": "正式表格和图片仍未全部完成",
    }
    lines = [
        "# 对标 campaign 正式就绪状态（中文伴随档案）",
        "",
        "## 来源与绑定",
        "",
        f"- Campaign：`{report['campaign_id']}`。",
        f"- 英文机器报告：`{source_path.name}`。",
        f"- 英文报告 SHA256：`{sha256_file(source_path)}`。",
        f"- 当前状态：`{report['status']}`；通过 `{report['passed_count']}/{report['requirement_count']}`。",
        "- 本文件只解释当前证据，不增加正式实验结果。",
        "",
        "## 九项要求",
        "",
        "| 要求 | 状态 | 阻塞项 |",
        "|---|---|---|",
    ]
    for row in report["requirements"]:
        blockers = "；".join(
            blocker_zh.get(str(value), str(value)) for value in row.get("blockers") or []
        ) or "无"
        lines.append(
            f"| {labels.get(row['requirement_id'], row['requirement_id'])} | "
            f"`{row['status']}` | {blockers} |"
        )
    lines.extend([
        "",
        "## 关键解释",
        "",
        "R3 只接受逐样本预测齐全、source freeze 文件与归档哈希可复核、且 tracked patch 内容哈希一致的 15 个 fold-seed 单元。旧 scratch 的 patch provenance 失效后，不得继续被 readiness 计为正式证据；当前映射使用 scratch-v2。SURE 仍为 0/15，正式执行开关保持关闭。",
        "",
        "## 当前边界",
        "",
        "smoke、代码存在、作者原始数值、跨 FPGA 平台结果和历史代理值都不能替代本项目统一协议结果。SURE、HLS、板卡或功耗实验仍须单独授权。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="ccf_ab_nksid_av7k325_v1")
    parser.add_argument("--config", default="configs/benchmarks/ccf_ab_campaign_v1.yaml")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    paths = CampaignPaths.from_repo(PROJECT_ROOT, args.campaign_id).create()
    manifests = paths.artifact_root / "manifests"

    source_path = manifests / "source_audit.json"
    source = _read_json(source_path) if source_path.is_file() else {"audits": []}
    main_audits = [
        row for row in source.get("audits") or []
        if (row.get("paper") or {}).get("registry_role", "main") == "main"
    ]
    source_pass = len(main_audits) == 5 and all(
        row.get("source_audit_pass") is True and row.get("commit_matches_pin") is True
        for row in main_audits
    )

    environment_index_path = paths.artifact_root / "environment" / "index.json"
    environment_index = (
        _read_json(environment_index_path) if environment_index_path.is_file() else {"cards": []}
    )
    cards = []
    for value in environment_index.get("cards") or []:
        card_path = Path(value)
        if card_path.is_file():
            cards.append(_read_json(card_path))
    environments_ready = len(cards) == 6 and all(
        row.get("isolation_status") == "READY_DEDICATED_ENVIRONMENT"
        for row in cards
    )

    classification = dict(config.get("classification") or {})
    folds = [int(value) for value in classification.get("outer_folds") or []]
    seeds = [int(value) for value in classification.get("seeds") or []]
    expected_pairs = {(fold, seed) for fold in folds for seed in seeds}
    closed_methods = [dict(row) for row in classification.get("formal_methods") or []]
    closed_ids = [str(row["method_id"]) for row in closed_methods]
    open_config = dict(config.get("open_long_tail") or {})
    open_methods = [dict(row) for row in open_config.get("formal_methods") or []]
    open_ids = [str(row["method_id"]) for row in open_methods]
    closed_runs = _formal_classification_runs(
        PROJECT_ROOT,
        task="closed_set",
        methods=closed_methods,
        expected_pairs=expected_pairs,
    )
    open_runs = _formal_classification_runs(
        PROJECT_ROOT,
        task="open_long_tail",
        methods=open_methods,
        expected_pairs=expected_pairs,
    )

    search_cfg = dict(config.get("search") or {})
    formal_search = dict(search_cfg.get("formal") or {})
    search_path = paths.raw_root / "formal" / "search_comparison" / "comparison.json"
    search_report = _read_json(search_path) if search_path.is_file() else {}
    required_search_methods = sorted(str(value) for value in search_cfg.get("methods") or [])
    required_search_seeds = sorted(int(value) for value in formal_search.get("seeds") or [])
    observed_search_methods = sorted(str(value) for value in search_report.get("seed_sets") or {})
    search_ready = bool(
        search_report.get("comparison_ready")
        and search_report.get("formal_exact_hv_ready")
        and observed_search_methods == required_search_methods
        and all(
            sorted(int(value) for value in (search_report.get("seed_sets") or {}).get(method, []))
            == required_search_seeds
            for method in required_search_methods
        )
        and all(
            int(row.get("candidate_count") or 0) == int(formal_search.get("evaluator_calls") or 0)
            for row in search_report.get("runs") or []
        )
    )

    calibration_path = PROJECT_ROOT / "artifacts/hw_surrogate_calibration_v2/calibration_v2.json"
    calibration = _read_json(calibration_path) if calibration_path.is_file() else {}
    hls_count = int(
        (calibration.get("evidence_counts") or {}).get("mainline_network_unique_rows") or 0
    )
    required_hls = int((config.get("hls_proxy") or {}).get("formal_minimum_samples") or 100)

    gate_path = PROJECT_ROOT / "artifacts/measurement_first_rebuild/status.json"
    gate = _read_json(gate_path) if gate_path.is_file() else {"gates": {}}
    gate_rows = dict(gate.get("gates") or {})
    required_gate_ids = ["G0_protocol", "G1_accuracy_baselines", "G2_hardware_measurement", "G4_int8_board"]
    required_gates_pass = all((gate_rows.get(key) or {}).get("pass") is True for key in required_gate_ids)
    search_gate = dict(gate_rows.get("G3_search") or {})
    search_reentry = bool(search_gate.get("may_start_new_claimable_search"))
    power_gate = dict(gate_rows.get("power") or {})

    artifact_path = manifests / "artifact_status.json"
    artifact = _read_json(artifact_path) if artifact_path.is_file() else {}
    table_states = {key: row.get("status") for key, row in (artifact.get("tables") or {}).items()}
    figure_states = {key: row.get("status") for key, row in (artifact.get("figures") or {}).items()}
    archive_ready = (
        len(table_states) == 9
        and len(figure_states) == 12
        and all(value == "AVAILABLE" for value in [*table_states.values(), *figure_states.values()])
    )

    requirements = [
        make_requirement(
            "R1_sources",
            "Five main paper source, version and license/correspondence audit cards",
            passed=source_pass,
            observed={"main_cards": len(main_audits), "passing": sum(bool(row.get("source_audit_pass")) for row in main_audits)},
            required={"main_cards": 5, "pinned_commit_match": True},
            evidence=[_evidence(source_path)],
            blockers=[] if source_pass else ["main-paper source audit cards are incomplete"],
        ),
        make_requirement(
            "R2_environments",
            "One locked dedicated runtime per registered main/supplementary paper",
            passed=environments_ready,
            observed={"cards": len(cards), "ready": sum(row.get("isolation_status") == "READY_DEDICATED_ENVIRONMENT" for row in cards)},
            required={"cards": 6, "status": "READY_DEDICATED_ENVIRONMENT"},
            evidence=[_evidence(environment_index_path)],
            blockers=[] if environments_ready else ["six paper-specific environments are not locked and verified"],
        ),
        make_requirement(
            "R3_closed_set",
            "Four closed-set methods with complete claimable 5-fold x 3-seed predictions",
            passed=bool(closed_runs) and all(row["claimable"] for row in closed_runs.values()),
            observed=closed_runs,
            required={"methods": closed_ids, "pairs_per_method": len(expected_pairs)},
            blockers=[] if closed_runs and all(row["claimable"] for row in closed_runs.values()) else ["closed-set 15-unit formal runs are incomplete"],
        ),
        make_requirement(
            "R4_open_long_tail",
            "CE+MSP, DMCL and PLUD with complete claimable 5-fold x 3-seed predictions",
            passed=bool(open_runs) and all(row["claimable"] for row in open_runs.values()),
            observed=open_runs,
            required={"methods": open_ids, "pairs_per_method": len(expected_pairs)},
            blockers=[] if open_runs and all(row["claimable"] for row in open_runs.values()) else ["open-set 15-unit formal runs are incomplete"],
        ),
        make_requirement(
            "R5_nas_formal",
            "Four NAS methods at equal 300-evaluator budgets over ten paired seeds with exact HV",
            passed=search_ready,
            observed={"comparison_exists": search_path.is_file(), "methods": observed_search_methods, "formal_exact_hv_ready": bool(search_report.get("formal_exact_hv_ready"))},
            required={"methods": required_search_methods, "seeds": required_search_seeds, "evaluator_calls": formal_search.get("evaluator_calls")},
            evidence=[_evidence(search_path)],
            blockers=[] if search_ready else ["formal matched NAS comparison is absent or incomplete"],
        ),
        make_requirement(
            "R6_hls_proxy",
            "At least 100 semantic-safe full-network HLS/route samples for grouped 5-fold comparison",
            passed=hls_count >= required_hls and bool((gate_rows.get("G2_hardware_measurement") or {}).get("pass")),
            observed={"mainline_network_unique_rows": hls_count, "g2_pass": bool((gate_rows.get("G2_hardware_measurement") or {}).get("pass"))},
            required={"minimum_samples": required_hls, "validation": (config.get("hls_proxy") or {}).get("validation"), "g2_pass": True},
            evidence=[_evidence(calibration_path)],
            blockers=[] if hls_count >= required_hls and bool((gate_rows.get("G2_hardware_measurement") or {}).get("pass")) else ["formal HLS/route sample threshold and G2 are not satisfied"],
        ),
        make_requirement(
            "R7_board_power",
            "Three fixed AV7K325 candidates measured with one route and external power protocol",
            passed=bool((gate_rows.get("G4_int8_board") or {}).get("pass")) and bool(power_gate.get("pass")),
            observed={"g4": (gate_rows.get("G4_int8_board") or {}).get("status"), "power": power_gate.get("status")},
            required={"candidate_roles": (config.get("power") or {}).get("candidate_roles"), "g4_pass": True, "power_pass": True},
            evidence=[_evidence(gate_path)],
            blockers=[] if bool((gate_rows.get("G4_int8_board") or {}).get("pass")) and bool(power_gate.get("pass")) else ["AV7K325 three-candidate board/power evidence is incomplete"],
        ),
        make_requirement(
            "R8_gates",
            "Measurement-first gates and claimable-search re-entry",
            passed=required_gates_pass and search_reentry,
            observed={key: (gate_rows.get(key) or {}).get("status") for key in [*required_gate_ids, "G3_search", "G5_sonar_ablation", "power"]},
            required={"pass": required_gate_ids, "G3_may_start_new_claimable_search": True},
            evidence=[_evidence(gate_path)],
            blockers=[] if required_gates_pass and search_reentry else ["measurement-first gates do not permit the full formal campaign"],
        ),
        make_requirement(
            "R9_archive",
            "T1-T9 and F1-F12 available in every required archival format",
            passed=archive_ready,
            observed={"tables": table_states, "figures": figure_states},
            required={"tables_available": 9, "figures_available": 12},
            evidence=[_evidence(artifact_path)],
            blockers=[] if archive_ready else ["formal tables/figures are still pending"],
        ),
    ]
    report = build_readiness_report(args.campaign_id, requirements)
    report["language"] = "zh-CN"
    report["boundary_zh"] = (
        "只有九项要求全部由对应证据层证明后，campaign 才能标记 READY。"
    )
    report["campaign_config"] = _evidence(config_path)
    json_path = write_json(manifests / "formal_readiness.json", report)
    md_path = manifests / "formal_readiness.md"
    _write_markdown(md_path, report)
    _write_markdown_zh(manifests / "formal_readiness.zh-CN.md", report, md_path)
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
