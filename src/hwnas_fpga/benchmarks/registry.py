"""Paper registry and third-party source audit helpers.

The registry separates bibliographic suitability from executable readiness.
An open GitHub URL is not treated as a redistribution license, and a checkout
that merely exists is not treated as a completed reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import subprocess
from typing import Any, Mapping

import yaml


COMPARABILITY_CLASSES = {"A", "B", "C"}
LICENSE_STATES = {"verified", "unverified", "missing", "incompatible"}
CODE_CORRESPONDENCE_STATES = {"verified", "partial", "unverified"}
REGISTRY_ROLES = {"main", "supplementary"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkPaper:
    paper_id: str
    title: str
    year: int
    venue: str
    direction: str
    comparability_class: str
    paper_url: str
    repo_url: str
    checkout_path: str
    pinned_commit: str
    license_spdx: str | None
    license_state: str
    official_code: bool
    execution_role: str
    numerical_comparison_rule: str
    registry_role: str
    paper_code_correspondence: str
    correspondence_note: str | None
    smoke: Mapping[str, Any]

    def validate(self) -> None:
        if not self.paper_id or any(char.isspace() for char in self.paper_id):
            raise ValueError(f"invalid paper_id: {self.paper_id!r}")
        if self.comparability_class not in COMPARABILITY_CLASSES:
            raise ValueError(
                f"{self.paper_id}: comparability_class must be A/B/C, "
                f"got {self.comparability_class!r}"
            )
        if self.license_state not in LICENSE_STATES:
            raise ValueError(f"{self.paper_id}: invalid license_state={self.license_state!r}")
        if self.paper_code_correspondence not in CODE_CORRESPONDENCE_STATES:
            raise ValueError(
                f"{self.paper_id}: invalid paper_code_correspondence="
                f"{self.paper_code_correspondence!r}"
            )
        if self.paper_code_correspondence != "verified" and not self.correspondence_note:
            raise ValueError(f"{self.paper_id}: non-verified code correspondence needs a note")
        if self.registry_role not in REGISTRY_ROLES:
            raise ValueError(
                f"{self.paper_id}: registry_role must be main/supplementary"
            )
        if len(self.pinned_commit) != 40:
            raise ValueError(f"{self.paper_id}: pinned_commit must be a full 40-char SHA")
        if not self.paper_url.startswith("https://") or not self.repo_url.startswith("https://"):
            raise ValueError(f"{self.paper_id}: paper/repo URLs must use https")
        if self.license_state == "verified" and not self.license_spdx:
            raise ValueError(f"{self.paper_id}: verified license requires license_spdx")
        if self.comparability_class == "C" and "no_cross_platform_ranking" not in str(
            self.numerical_comparison_rule
        ):
            raise ValueError(
                f"{self.paper_id}: class C must explicitly prohibit cross-platform ranking"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _paper_from_mapping(payload: Mapping[str, Any]) -> BenchmarkPaper:
    paper = BenchmarkPaper(
        paper_id=str(payload["paper_id"]),
        title=str(payload["title"]),
        year=int(payload["year"]),
        venue=str(payload["venue"]),
        direction=str(payload["direction"]),
        comparability_class=str(payload["comparability_class"]).upper(),
        paper_url=str(payload["paper_url"]),
        repo_url=str(payload["repo_url"]),
        checkout_path=str(payload["checkout_path"]),
        pinned_commit=str(payload["pinned_commit"]),
        license_spdx=(
            None if payload.get("license_spdx") in (None, "") else str(payload["license_spdx"])
        ),
        license_state=str(payload["license_state"]),
        official_code=bool(payload.get("official_code", True)),
        execution_role=str(payload["execution_role"]),
        numerical_comparison_rule=str(payload["numerical_comparison_rule"]),
        registry_role=str(payload.get("registry_role", "main")),
        paper_code_correspondence=str(
            payload.get("paper_code_correspondence", "verified")
        ),
        correspondence_note=(
            None
            if payload.get("correspondence_note") in (None, "")
            else str(payload["correspondence_note"])
        ),
        smoke=dict(payload.get("smoke") or {}),
    )
    paper.validate()
    return paper


def load_paper_registry(path: str | Path) -> list[BenchmarkPaper]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("paper registry schema_version must be 1")
    papers = [_paper_from_mapping(item) for item in payload.get("papers", [])]
    if not papers:
        raise ValueError("paper registry contains no papers")
    ids = [paper.paper_id for paper in papers]
    if len(ids) != len(set(ids)):
        raise ValueError("paper registry contains duplicate paper_id values")
    return papers


def _git(path: Path, *args: str) -> tuple[int, str, str]:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def audit_source_checkout(paper: BenchmarkPaper, repo_root: str | Path) -> dict[str, Any]:
    checkout = (Path(repo_root) / paper.checkout_path).resolve()
    audit: dict[str, Any] = {
        "paper": paper.to_dict(),
        "checkout_path": str(checkout),
        "checkout_exists": checkout.is_dir(),
        "commit_matches_pin": False,
        "remote_matches_registry": False,
        "license_file": None,
        "license_file_sha256": None,
        "source_shape": "missing",
        "formal_eligible": False,
        "blockers": [],
    }
    if not checkout.is_dir():
        audit["blockers"].append("checkout_missing")
        return audit

    rc, commit, error = _git(checkout, "rev-parse", "HEAD")
    audit["observed_commit"] = commit if rc == 0 else None
    if rc != 0:
        audit["blockers"].append(f"git_commit_unavailable:{error}")
    else:
        audit["commit_matches_pin"] = commit == paper.pinned_commit
        if not audit["commit_matches_pin"]:
            audit["blockers"].append("commit_pin_mismatch")

    rc, remote, error = _git(checkout, "remote", "get-url", "origin")
    audit["observed_remote"] = remote if rc == 0 else None
    expected_remote = paper.repo_url.removesuffix(".git").lower()
    observed_remote = remote.removesuffix(".git").lower() if rc == 0 else ""
    audit["remote_matches_registry"] = observed_remote == expected_remote
    if not audit["remote_matches_registry"]:
        audit["blockers"].append(f"remote_mismatch:{error or remote}")

    license_candidates = sorted(
        path
        for path in checkout.iterdir()
        if path.is_file()
        and path.name.upper().split(".")[0] in {"LICENSE", "COPYING", "NOTICE"}
    )
    if license_candidates:
        license_path = license_candidates[0]
        audit["license_file"] = str(license_path)
        audit["license_file_sha256"] = sha256_file(license_path)
    if paper.license_state == "verified" and not license_candidates:
        audit["blockers"].append("verified_license_file_missing")
    if paper.license_state != "verified":
        audit["blockers"].append("redistribution_license_unverified")
    if paper.paper_code_correspondence != "verified":
        audit["blockers"].append(
            f"paper_code_correspondence_{paper.paper_code_correspondence}"
        )

    rar_files = sorted(checkout.glob("*.rar"))
    python_files = sorted(checkout.rglob("*.py"))
    if rar_files and not python_files:
        audit["source_shape"] = "archive_only"
        audit["archives"] = [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in rar_files
        ]
        audit["blockers"].append("archive_requires_isolated_extraction_and_audit")
    elif python_files:
        audit["source_shape"] = "python_source"
        audit["python_file_count"] = len(python_files)
    else:
        audit["source_shape"] = "non_python_or_empty"

    audit["source_audit_pass"] = not any(
        blocker.startswith(("checkout_", "git_", "commit_", "remote_"))
        for blocker in audit["blockers"]
    )
    # Source presence never makes a paper result formally eligible. Formal
    # eligibility is granted only after a local unified-protocol run exists.
    audit["formal_eligible"] = False
    audit["claimability_status"] = "PENDING_LOCAL_REEVALUATION"
    return audit
