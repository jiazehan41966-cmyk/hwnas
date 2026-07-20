from pathlib import Path
import json
from types import SimpleNamespace
import sys

import pytest
import yaml

from run_eval_protocol import environment_card_provenance, runtime_provenance
from hwnas_fpga.training.protocol_reporting import sha256_file
from hwnas_fpga.training.protocol_reporting import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_environment_config_covers_registry_once() -> None:
    registry = yaml.safe_load(
        (ROOT / "configs/benchmarks/paper_registry_v1.yaml").read_text(encoding="utf-8")
    )
    runtime = yaml.safe_load(
        (ROOT / "configs/benchmarks/runtime_environments_v1.yaml").read_text(encoding="utf-8")
    )
    paper_ids = {row["paper_id"] for row in registry["papers"]}
    assert set(runtime["papers"]) == paper_ids
    assert len(runtime["papers"]) == 6
    assert runtime["runtime_role"] == "unified_protocol_adapter_not_author_original_environment"


def test_every_runtime_path_is_ignored_and_distinct() -> None:
    runtime = yaml.safe_load(
        (ROOT / "configs/benchmarks/runtime_environments_v1.yaml").read_text(encoding="utf-8")
    )
    paths = [row["environment_path"] for row in runtime["papers"].values()]
    assert len(paths) == len(set(paths))
    assert all(path.startswith(".venv_benchmarks/") for path in paths)


def test_host_capture_preserves_verified_dedicated_environment() -> None:
    source = (ROOT / "scripts/capture_benchmark_environments.py").read_text(
        encoding="utf-8"
    )
    assert 'previous.get("dedicated_environment")' in source
    assert 'previous.get("isolation_status")' in source


def test_external_adapter_environment_card_binds_interpreter_and_lock(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "runtime.lock.txt"
    lock.write_text("torch==test\n", encoding="utf-8")
    card = tmp_path / "sure.json"
    freeze = {"path": str(lock), "sha256": sha256_file(lock)}
    dedicated = {
        "path": sys.prefix,
        "interpreter": sys.executable,
        "probe": {"status": "PASS"},
        "freeze": freeze,
    }
    dedicated["verification_fingerprint"] = canonical_sha256(
        {
            "runtime_role": "test",
            "path": dedicated["path"],
            "interpreter": dedicated["interpreter"],
            "probe": dedicated["probe"],
            "freeze": dedicated["freeze"],
        }
    )
    card.write_text(
        json.dumps(
            {
                "isolation_status": "READY_DEDICATED_ENVIRONMENT",
                "runtime_role": "test",
                "dedicated_environment": dedicated,
            }
        ),
        encoding="utf-8",
    )
    provenance = environment_card_provenance(
        SimpleNamespace(
            environment_card=str(card),
            adapter_id="sure_author_recipe",
            campaign_id="campaign",
            paper_id="sure_2024",
        )
    )
    assert provenance is not None
    assert provenance["freeze_sha256"] == sha256_file(lock)
    assert provenance["verification_fingerprint"] == dedicated["verification_fingerprint"]
    assert runtime_provenance()["interpreter"] == str(Path(sys.executable).resolve())


def test_external_adapter_rejects_mismatched_environment(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.lock.txt"
    lock.write_text("torch==test\n", encoding="utf-8")
    card = tmp_path / "sure.json"
    freeze = {"path": str(lock), "sha256": sha256_file(lock)}
    dedicated = {
        "path": str(tmp_path / "different_env"),
        "interpreter": sys.executable,
        "probe": {"status": "PASS"},
        "freeze": freeze,
    }
    dedicated["verification_fingerprint"] = canonical_sha256(
        {
            "runtime_role": None,
            "path": dedicated["path"],
            "interpreter": dedicated["interpreter"],
            "probe": dedicated["probe"],
            "freeze": dedicated["freeze"],
        }
    )
    card.write_text(
        json.dumps(
            {
                "isolation_status": "READY_DEDICATED_ENVIRONMENT",
                "dedicated_environment": dedicated,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        environment_card_provenance(
            SimpleNamespace(
                environment_card=str(card),
                adapter_id="sure_author_recipe",
                campaign_id="campaign",
                paper_id="sure_2024",
            )
        )
