"""Campaign directory, prediction schema, table, and figure metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
from typing import Any, Mapping, Sequence

from .adapters import CLAIMABILITY_STATES


PREDICTION_REQUIRED_FIELDS = (
    "campaign_id",
    "paper_id",
    "method",
    "fold",
    "seed",
    "sample_id",
    "target",
    "prediction",
    "confidence",
    "checkpoint_sha",
    "config_sha",
    "data_sha",
    "split_sha",
    "code_commit",
    "code_state_sha",
    "claimability_status",
)


TABLE_TITLES = {
    "T1": "Open-source Baseline Eligibility and Reproducibility Audit / 开源基线资格、许可证与可复现性审计",
    "T2": "Closed-set NKSID Classification, Calibration and Cost / NKSID 闭集分类、校准与计算成本",
    "T3": "Open-set Long-tailed Sonar Recognition / NKSID 开放集长尾识别",
    "T4": "Sonar Corruption and SNR Robustness / 声呐噪声、SNR 与 corruption 鲁棒性",
    "T5": "NAS Search Efficiency and Pareto Quality / NAS 搜索效率与 Pareto 质量",
    "T6": "HLS and Route Proxy Reliability / HLS-Route 代理可靠性",
    "T7": "AV7K325 Synthesis, Route and Board Latency / AV7K325 综合、布线与板级延迟",
    "T8": "AV7K325 Power and Dynamic Energy Efficiency / AV7K325 功耗与动态能效",
    "T9": "Ablations, Effect Sizes and Significance / 消融实验、效应量与显著性",
}


FIGURE_TITLES = {
    "F1": "Evidence-aligned Benchmark Workflow / 分层证据与对标实验工作流",
    "F2": "Anytime Exact Hypervolume by NAS Method / 各 NAS 方法的 anytime exact HV",
    "F3": "Feasible NAS Pareto Front / NAS 可行 Pareto front",
    "F4": "Proxy Predictions versus HLS-Route Measurements / 代理预测值与 HLS-route 实测值",
    "F5": "Closed- and Open-set Confusion Matrices / 闭集及开放集混淆矩阵",
    "F6": "Reliability and Risk-Coverage Curves / 可靠性图与 risk-coverage curve",
    "F7": "Macro-F1 versus Image-domain SNR / macro-F1-SNR 曲线",
    "F8": "Clean, Corrupted and Processed Sonar Examples / clean-corrupted-processed 声呐样例",
    "F9": "FPGA Resource Utilization / LUT-DSP-BRAM-FF 资源利用率",
    "F10": "Board Latency Distribution and ECDF / 板级延迟分布与 ECDF",
    "F11": "Idle-Active Power Traces and Dynamic Energy / 功率时间序列与动态能量",
    "F12": "Accuracy-Latency-Energy Pareto Front / macro-F1、延迟和动态能耗 Pareto 图",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in PREDICTION_REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"prediction record missing fields: {missing}")
    if not 0.0 <= float(record["confidence"]) <= 1.0:
        raise ValueError("prediction confidence must be in [0, 1]")
    for field in ("checkpoint_sha", "config_sha", "data_sha", "split_sha", "code_state_sha"):
        if len(str(record[field])) != 64:
            raise ValueError(f"{field} must be a SHA256 hex digest")
    if str(record["claimability_status"]) not in CLAIMABILITY_STATES:
        raise ValueError("invalid claimability_status")


@dataclass(frozen=True)
class CampaignPaths:
    campaign_id: str
    raw_root: Path
    artifact_root: Path
    document_path: Path

    @classmethod
    def from_repo(cls, repo_root: str | Path, campaign_id: str) -> "CampaignPaths":
        root = Path(repo_root).resolve()
        return cls(
            campaign_id=campaign_id,
            raw_root=root / "results" / "benchmarks" / campaign_id,
            artifact_root=root / "artifacts" / "benchmarks" / campaign_id,
            document_path=root / "docs" / "benchmark_campaigns" / f"{campaign_id}.md",
        )

    def create(self) -> "CampaignPaths":
        self.raw_root.mkdir(parents=True, exist_ok=True)
        for directory in (
            "tables",
            "figures",
            "source_data",
            "manifests",
            "logs",
            "environment",
        ):
            (self.artifact_root / directory).mkdir(parents=True, exist_ok=True)
        self.document_path.parent.mkdir(parents=True, exist_ok=True)
        return self


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def write_table_bundle(
    paths: CampaignPaths,
    table_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if table_id not in TABLE_TITLES:
        raise ValueError(f"unknown table_id: {table_id}")
    if not rows:
        raise ValueError("table rows must not be empty")
    fields = list(rows[0].keys())
    if any(list(row.keys()) != fields for row in rows):
        raise ValueError("all table rows must use identical ordered fields")
    table_dir = paths.artifact_root / "tables"
    base = table_dir / table_id.lower()
    table_dir.mkdir(parents=True, exist_ok=True)
    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def text(value: Any) -> str:
        return "" if value is None else str(value).replace("|", "\\|")

    markdown = [f"# {table_id}: {TABLE_TITLES[table_id]}", "", "| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    markdown.extend("| " + " | ".join(text(row[field]) for field in fields) + " |" for row in rows)
    base.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    def latex(value: Any) -> str:
        result = "" if value is None else str(value)
        for old, new in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&")):
            result = result.replace(old, new)
        return result

    latex_rows = [" & ".join(latex(field) for field in fields) + r" \\", r"\hline"]
    latex_rows.extend(" & ".join(latex(row[field]) for field in fields) + r" \\" for row in rows)
    base.with_suffix(".tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    return {
        "csv": str(base.with_suffix(".csv")),
        "markdown": str(base.with_suffix(".md")),
        "latex": str(base.with_suffix(".tex")),
    }


def write_figure_metadata(
    paths: CampaignPaths,
    figure_id: str,
    *,
    caption: str,
    supported_claim: str,
    limitations: Sequence[str],
    generator_sha: str,
    input_files: Sequence[str | Path],
    source_data_path: str | Path,
) -> Path:
    if figure_id not in FIGURE_TITLES:
        raise ValueError(f"unknown figure_id: {figure_id}")
    for digest in (generator_sha,):
        if len(digest) != 64:
            raise ValueError("generator_sha must be a SHA256 hex digest")
    inputs = [
        {"path": str(Path(path).resolve()), "sha256": sha256_file(path)} for path in input_files
    ]
    source_path = Path(source_data_path).resolve()
    payload = {
        "figure_id": figure_id,
        "title": FIGURE_TITLES[figure_id],
        "caption": caption,
        "supported_claim": supported_claim,
        "limitations": list(limitations),
        "generator_sha256": generator_sha,
        "inputs": inputs,
        "source_data": {"path": str(source_path), "sha256": sha256_file(source_path)},
    }
    return write_json(paths.artifact_root / "figures" / f"{figure_id.lower()}_meta.json", payload)
