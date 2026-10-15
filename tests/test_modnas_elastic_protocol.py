"""Prevent test-set selection, source drift, and mismatched execution claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from tools.modnas_elastic import protocol


@pytest.mark.parametrize("change", ["test_split", "old_space", "formal_smoke"])
def test_unregistered_protocol_is_rejected(tmp_path, change):
    cfg, _ = protocol.load_config()
    cfg = deepcopy(cfg)
    if change == "test_split":
        cfg["dataset"]["architecture_split"] = "test"
    elif change == "old_space":
        cfg["space"]["unique_candidates"] = 144
    else:
        cfg["smoke"]["formal_eligible"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        protocol.load_config(path)


def test_source_closure_binds_verified_author_card_and_all_primitives():
    from tools.modnas_elastic.author import DEFAULT_SOURCE_MANIFEST, SOURCE_ROOT, EXPECTED_AUTHOR_SHA256
    sources = protocol.source_binding()
    assert DEFAULT_SOURCE_MANIFEST.relative_to(protocol.ROOT).as_posix() in sources
    for relative, expected in EXPECTED_AUTHOR_SHA256.items():
        assert sources[(SOURCE_ROOT / relative).relative_to(protocol.ROOT).as_posix()] == expected


def test_tampered_author_card_is_rejected(tmp_path):
    from tools.modnas_elastic.author import DEFAULT_SOURCE_MANIFEST, verify_author_sources
    card = json.loads(DEFAULT_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    card["files"][0]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hashes differ"):
        verify_author_sources(path)


@pytest.mark.parametrize("kind", ["source", "dataset", "environment"])
def test_execution_drift_requires_fresh_output(tmp_path, monkeypatch, kind):
    sources, data, environment = {"model.py": "a"}, {"train": "a", "valid": "b"}, {"device": "cpu"}
    monkeypatch.setattr(protocol, "source_binding", lambda: deepcopy(sources))
    monkeypatch.setattr(protocol, "dataset_binding", lambda baseline: deepcopy(data))
    monkeypatch.setattr(protocol, "runtime", lambda device: deepcopy(environment))
    monkeypatch.setattr(protocol, "git_identity", lambda: {"commit": "diagnostic"})
    protocol.prepare(tmp_path)
    if kind == "source":
        sources["model.py"] = "changed"
    elif kind == "dataset":
        data["valid"] = "changed"
    else:
        environment["device"] = "cuda"
    with pytest.raises(ValueError, match="manifest drift"):
        protocol.prepare(tmp_path)

