#!/usr/bin/env python3
"""Generate a review-oriented inventory for results/ and artifacts/."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Entry:
    root: str
    name: str
    files: int
    size_mb: float
    tag: str
    action: str


def classify_results(name: str) -> tuple[str, str]:
    normalized = name.lower()
    if "formal" in normalized or "baseline" in normalized:
        return "formal-or-baseline", "keep_candidate"
    if "paper" in normalized or "comparison" in normalized:
        return "reporting", "keep_candidate"
    if "smoke" in normalized or "debug" in normalized or "test" in normalized:
        return "smoke-debug-test", "archive_candidate"
    if "viz" in normalized or "plot" in normalized:
        return "visualization", "archive_candidate"
    if "restart" in normalized or "runtime_probe" in normalized:
        return "restart-or-probe", "archive_candidate"
    if "live" in normalized or "quick" in normalized:
        return "interactive-or-quick", "review_first"
    if normalized.startswith("_"):
        return "helper", "archive_candidate"
    return "uncategorized", "review_first"


def classify_artifacts(name: str) -> tuple[str, str]:
    normalized = name.lower()
    if "server_snapshots" in normalized:
        return "snapshot", "archive_candidate"
    if "formal" in normalized or "structured" in normalized:
        return "formal-export", "keep_candidate"
    if "smoke" in normalized or "tmp" in normalized:
        return "smoke-temp", "archive_candidate"
    return "artifact", "review_first"


def iter_entries(root_path: Path) -> Iterable[Entry]:
    classifier = classify_results if root_path.name == "results" else classify_artifacts
    for child in sorted(root_path.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        files = list(child.rglob("*"))
        file_count = sum(1 for item in files if item.is_file())
        total_bytes = sum(item.stat().st_size for item in files if item.is_file())
        tag, action = classifier(child.name)
        yield Entry(
            root=root_path.name,
            name=child.name,
            files=file_count,
            size_mb=round(total_bytes / (1024 * 1024), 2),
            tag=tag,
            action=action,
        )


def render_markdown(entries: list[Entry], repo_root: Path) -> str:
    lines = [
        "# Run Storage Audit",
        "",
        "This file is an inventory only. No directories were deleted by this audit.",
        "",
        "Review policy:",
        "",
        "- `reference/` is retained and excluded from cleanup.",
        "- `results/` should be reviewed manually before deletion or archiving.",
        "- `Suggested Action` is heuristic only; reviewed overrides belong in "
        "  `docs/RESULT_RETENTION_DECISIONS.md` when present.",
        "- `artifacts/server_snapshots/` is usually the safest large archive target.",
        "",
        f"Repository root: `{repo_root}`",
        "",
        "| Root | Name | Files | Size (MB) | Tag | Suggested Action |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for entry in sorted(entries, key=lambda item: (item.root, -item.size_mb, item.name.lower())):
        lines.append(
            f"| {entry.root} | `{entry.name}` | {entry.files} | {entry.size_mb:.2f} | "
            f"{entry.tag} | {entry.action} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit top-level results/ and artifacts/ directories")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional markdown output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    entries: list[Entry] = []
    for dirname in ("results", "artifacts"):
        target = repo_root / dirname
        if target.exists():
            entries.extend(iter_entries(target))

    markdown = render_markdown(entries, repo_root)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
