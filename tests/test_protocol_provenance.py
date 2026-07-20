import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import run_eval_protocol
from scripts.audit_measurement_first_gates import g1_status, run_patch_integrity


def _mock_git(monkeypatch: pytest.MonkeyPatch, diff_payload: list[bytes]) -> None:
    monkeypatch.setattr(run_eval_protocol, "git_commit", lambda: "a" * 40)

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess:
        command = args[1]
        if command == "status":
            return subprocess.CompletedProcess(args, 0, stdout=" M tracked.py\n", stderr="")
        if command == "diff":
            return subprocess.CompletedProcess(args, 0, stdout=diff_payload[0], stderr=b"")
        if command == "ls-files":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(run_eval_protocol.subprocess, "run", fake_run)


def test_different_git_diffs_never_overwrite_prior_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = [b"first tracked diff\n"]
    _mock_git(monkeypatch, payload)
    run_dir = tmp_path / "run"

    first = run_eval_protocol.git_code_provenance(run_dir)
    first_path = Path(first["tracked_patch"]["path"])
    first_bytes = first_path.read_bytes()
    payload[0] = b"second tracked diff\n"
    second = run_eval_protocol.git_code_provenance(run_dir)
    second_path = Path(second["tracked_patch"]["path"])

    assert first_path != second_path
    assert first_path.read_bytes() == first_bytes == b"first tracked diff\n"
    assert second_path.read_bytes() == b"second tracked diff\n"
    assert first["tracked_patch"]["sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert not (run_dir / "code_patch.diff").exists()


def test_same_git_diff_reuses_content_addressed_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = [b"stable tracked diff\n"]
    _mock_git(monkeypatch, payload)
    run_dir = tmp_path / "run"

    first = run_eval_protocol.git_code_provenance(run_dir)
    first_path = Path(first["tracked_patch"]["path"])
    first_mtime_ns = first_path.stat().st_mtime_ns
    second = run_eval_protocol.git_code_provenance(run_dir)

    assert first["tracked_patch"] == second["tracked_patch"]
    assert first_path.read_bytes() == payload[0]
    assert first_path.stat().st_mtime_ns == first_mtime_ns


def test_corrupted_content_addressed_patch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = [b"expected tracked diff\n"]
    _mock_git(monkeypatch, payload)
    expected_sha256 = hashlib.sha256(payload[0]).hexdigest()
    patch_path = (
        tmp_path
        / "run"
        / "provenance"
        / f"code_patch_{expected_sha256}.diff"
    )
    patch_path.parent.mkdir(parents=True)
    patch_path.write_bytes(b"corrupted bytes")

    with pytest.raises(RuntimeError, match="Refusing to overwrite corrupted"):
        run_eval_protocol.git_code_provenance(tmp_path / "run")

    assert patch_path.read_bytes() == b"corrupted bytes"


def _write_g1_method(
    root: Path,
    relative_run_dir: str,
    *,
    pretrained_loaded: bool,
    corrupt_patch: bool = False,
) -> Path:
    run_dir = root / relative_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "protocol_summary.json"
    summary = {
        "claimability": {"claimable": True},
        "runs": [{"fold": index // 3, "seed": 42 + index % 3} for index in range(15)],
        "run_fingerprints": ["b" * 64],
        "protocol_context_sha256": "c" * 64,
        "normalization": {"mean": [0.5], "std": [0.5]},
        "model": {"pretrained_loaded": pretrained_loaded},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    patch_path = run_dir / "provenance" / "code_patch_test.diff"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_bytes(b"recorded patch")
    expected_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    if corrupt_patch:
        expected_sha256 = "0" * 64
    manifest = {
        "code_provenance": {
            "tracked_patch": {
                "path": str(patch_path),
                "sha256": expected_sha256,
            }
        }
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return summary_path


def test_g1_ledger_rejects_manifest_patch_hash_mismatch(tmp_path: Path) -> None:
    scratch_summary = _write_g1_method(
        tmp_path,
        "results/protocol/g1_clean_20260718/g1_mobilenet_v2_scratch_v2",
        pretrained_loaded=False,
        corrupt_patch=True,
    )
    _write_g1_method(
        tmp_path,
        "results/protocol/g1_clean_20260711/g1_mobilenet_v2_grayscale_imagenet",
        pretrained_loaded=True,
    )
    _write_g1_method(
        tmp_path,
        "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected",
        pretrained_loaded=False,
    )

    assert run_patch_integrity(scratch_summary) is False
    result = g1_status(tmp_path)

    assert result["completed_tasks"] == 45
    assert result["gates"]["scratch_tracked_patch_integrity"] is False
    assert result["status"] == "PENDING"
    assert result["pass"] is False


def test_g1_ledger_accepts_three_intact_manifest_bound_patches(tmp_path: Path) -> None:
    _write_g1_method(
        tmp_path,
        "results/protocol/g1_clean_20260718/g1_mobilenet_v2_scratch_v2",
        pretrained_loaded=False,
    )
    _write_g1_method(
        tmp_path,
        "results/protocol/g1_clean_20260711/g1_mobilenet_v2_grayscale_imagenet",
        pretrained_loaded=True,
    )
    _write_g1_method(
        tmp_path,
        "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected",
        pretrained_loaded=False,
    )

    result = g1_status(tmp_path)

    assert result["completed_tasks"] == 45
    assert result["status"] == "PASS"
    assert result["pass"] is True


def test_staged_new_unit_limit_counts_only_new_training() -> None:
    assert run_eval_protocol.staged_limit_reached(None, 100) is False
    assert run_eval_protocol.staged_limit_reached(1, 0) is False
    assert run_eval_protocol.staged_limit_reached(1, 1) is True
    assert run_eval_protocol.staged_limit_reached(3, 2) is False
    assert run_eval_protocol.staged_limit_reached(3, 3) is True
    with pytest.raises(ValueError, match="positive integer"):
        run_eval_protocol.staged_limit_reached(0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        run_eval_protocol.staged_limit_reached(1, -1)


def test_cpu_unit_runtime_measurement_is_finite_and_has_no_gpu_peak() -> None:
    started_at = run_eval_protocol.begin_unit_runtime_measurement("cpu")
    measurement = run_eval_protocol.finish_unit_runtime_measurement(started_at, "cpu")

    assert measurement["wall_clock_seconds"] >= 0.0
    assert measurement["peak_gpu_memory_mb"] is None
