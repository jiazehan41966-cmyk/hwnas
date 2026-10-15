"""Synthetic temporary fixtures test contracts only, never research metrics."""
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from hw import HardwareContext, METRIC_UNITS, SCHEMA_VERSION, evaluate
from hw.backends.hls_report import parse_hls_xml
from quant import Q
from space import decode


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def xml(*, estimated="4.5", dsp="2", cycles="100"):
    return f'''<Report><ReportVersion><Version>2022.2</Version></ReportVersion>
<UserAssignments><Part>xczu7ev-ffvc1156-2-e</Part><TargetClockPeriod>5</TargetClockPeriod></UserAssignments>
<AreaEstimates><Resources><LUT>17</LUT><DSP>{dsp}</DSP><BRAM_18K>1</BRAM_18K></Resources>
<AvailableResources><LUT>999999</LUT><DSP>9999</DSP><BRAM_18K>9999</BRAM_18K></AvailableResources></AreaEstimates>
<SummaryOfOverallLatency><Worst-caseLatency>{cycles}</Worst-caseLatency></SummaryOfOverallLatency>
<SummaryOfTimingAnalysis><EstimatedClockPeriod>{estimated}</EstimatedClockPeriod></SummaryOfTimingAnalysis></Report>'''


def context(**kwargs):
    return HardwareContext("xczu7ev-ffvc1156-2-e", 200, "synthetic_test_only", "hls_report", "2022.2", "full_hls", **kwargs)


def write_bundle(tmp_path, *, scope="full_hls", backend="hls_report", estimated="4.5"):
    genotype = decode(0)
    plan = Q(genotype)
    implementation = tmp_path / "implementation.txt"
    implementation.write_text("SYNTHETIC TEST FIXTURE ONLY", encoding="utf-8")
    kwargs = {"implementation_sha256": digest(implementation)}
    sources = [{"role": "implementation", "path": implementation.name, "sha256": digest(implementation)}]
    if backend in {"board", "power"}:
        for role in ("checkpoint", "quant_artifact", "bitstream"):
            artifact = tmp_path / f"{role}.bin"
            artifact.write_bytes(f"SYNTHETIC {role}".encode())
            kwargs[role + "_sha256"] = digest(artifact)
            sources.append({"role": role, "path": artifact.name, "sha256": digest(artifact)})
    ctx = HardwareContext("xczu7ev-ffvc1156-2-e", 200, "synthetic_test_only", backend, "2022.2", scope, **kwargs)
    report = tmp_path / ("report.xml" if backend == "hls_report" else "report.txt")
    report.write_text(xml(estimated=estimated) if backend == "hls_report" else "SYNTHETIC TEST REPORT ONLY", encoding="utf-8")
    sources.append({"role": "report", "path": report.name, "sha256": digest(report)})
    metrics = {"latency_us": .5, "lut": 17, "dsp": 2, "bram18": 1} if scope != "power" else {"power_w": 1.5}
    payload = dict(schema_version=SCHEMA_VERSION, arch_hash=genotype.arch_hash, q_hash=plan.q_hash,
                   context=ctx.to_dict(), context_hash=ctx.context_hash, evidence_level=scope,
                   timing_status="PASS" if float(estimated) <= 5 else "FAIL",
                   metrics={k: dict(value=v, unit=METRIC_UNITS[k]) for k, v in metrics.items()},
                   provenance=dict(kind=scope, producer="synthetic_test_fixture", sources=sources))
    envelope = tmp_path / "report.xml.json"
    envelope.write_text(json.dumps(payload), encoding="utf-8")
    return genotype, plan, ctx, envelope, payload, report


def save(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_measurement_stays_empty(tmp_path):
    g = decode(0)
    result = evaluate(g, Q(g), context(), cache_path=tmp_path / "cache.db")
    assert result.status == "NEEDS_MEASUREMENT" and result.metrics == {}
    assert not result.pareto_eligible


def test_verified_hls_and_cache_repeat(tmp_path):
    g, q, ctx, path, _, report = write_bundle(tmp_path)
    cache = tmp_path / "cache.db"
    result = evaluate(g, q, ctx, evidence_path=report, cache_path=cache)
    assert result.status == "READY" and result.pareto_eligible
    assert result.metrics == {"latency_us": .5, "lut": 17, "dsp": 2, "bram18": 1}
    assert evaluate(g, q, ctx, cache_path=cache).to_dict() == result.to_dict()
    assert evaluate(g, q, ctx, evidence_path=path).status == "READY"
    with pytest.raises(TypeError):
        result.metrics["dsp"] = 20


@pytest.mark.parametrize("field,value", [
    ("device", "different_device"), ("clock_mhz", 100), ("mapping_policy", "other_policy"),
    ("tool_version", "2023.1"), ("scope", "operator_hls"),
    ("implementation_sha256", "a" * 64), ("checkpoint_sha256", "b" * 64),
    ("quant_artifact_sha256", "c" * 64), ("bitstream_sha256", "d" * 64),
])
def test_cache_context_isolation(tmp_path, field, value):
    g, q, ctx, path, _, _ = write_bundle(tmp_path)
    cache = tmp_path / "cache.db"
    assert evaluate(g, q, ctx, evidence_path=path, cache_path=cache).status == "READY"
    changed = replace(ctx, **{field: value})
    assert changed.context_hash != ctx.context_hash
    assert evaluate(g, q, changed, cache_path=cache).status == "NEEDS_MEASUREMENT"
    assert evaluate(g, q, changed, evidence_path=path).status == "INVALID_EVIDENCE"


def test_cache_never_aliases_precision_or_architecture(tmp_path):
    g, q, ctx, path, _, _ = write_bundle(tmp_path)
    cache = tmp_path / "cache.db"
    assert evaluate(g, q, ctx, evidence_path=path, cache_path=cache).status == "READY"
    assert evaluate(g, Q(g, "HEAD_W16A16"), ctx, cache_path=cache).status == "NEEDS_MEASUREMENT"
    other = decode(1)
    assert evaluate(other, Q(other), ctx, cache_path=cache).status == "NEEDS_MEASUREMENT"
    with pytest.raises(ValueError, match="different architecture"):
        evaluate(other, q, ctx)


@pytest.mark.parametrize("target", ["report", "implementation", "envelope"])
def test_cache_rehashes_every_source(tmp_path, target):
    g, q, ctx, path, _, report = write_bundle(tmp_path)
    cache = tmp_path / "cache.db"
    assert evaluate(g, q, ctx, evidence_path=path, cache_path=cache).status == "READY"
    changed = {"report": report, "implementation": tmp_path / "implementation.txt", "envelope": path}[target]
    changed.write_bytes(changed.read_bytes() + b" modified")
    result = evaluate(g, q, ctx, cache_path=cache)
    assert result.status == "INVALID_EVIDENCE" and result.metrics == {}
    assert evaluate(g, q, ctx, cache_path=cache).status == "NEEDS_MEASUREMENT"


def test_cache_missing_source_rejected(tmp_path):
    g, q, ctx, path, _, report = write_bundle(tmp_path)
    cache = tmp_path / "cache.db"
    assert evaluate(g, q, ctx, evidence_path=path, cache_path=cache).status == "READY"
    report.unlink()
    assert evaluate(g, q, ctx, cache_path=cache).status == "INVALID_EVIDENCE"


@pytest.mark.parametrize("metric,value,unit", [
    ("lut", -1, "count"), ("dsp", float("nan"), "count"),
    ("bram18", float("inf"), "18k_blocks"), ("lut", True, "count"),
    ("dsp", 2.0, "count"), ("latency_us", 0, "us"),
    ("latency_us", 1, "ms"), ("bram18", 1, "36k_blocks"),
    ("power_w", 5, "W"),
])
def test_bad_metrics_and_units_rejected(tmp_path, metric, value, unit):
    g, q, ctx, path, payload, _ = write_bundle(tmp_path)
    payload["metrics"][metric] = dict(value=value, unit=unit)
    save(path, payload)
    result = evaluate(g, q, ctx, evidence_path=path)
    assert result.status == "INVALID_EVIDENCE" and result.metrics == {}


@pytest.mark.parametrize("field,value", [("arch_hash", "a" * 64), ("q_hash", "b" * 64),
    ("context_hash", "c" * 64), ("evidence_level", "board"), ("schema_version", 1)])
def test_identity_forgery_rejected(tmp_path, field, value):
    g, q, ctx, path, payload, _ = write_bundle(tmp_path)
    payload[field] = value
    save(path, payload)
    assert evaluate(g, q, ctx, evidence_path=path).status == "INVALID_EVIDENCE"


def test_sidecar_cannot_forge_xml_resource_or_timing(tmp_path):
    g, q, ctx, path, payload, _ = write_bundle(tmp_path)
    payload["metrics"]["lut"]["value"] = 1
    save(path, payload)
    assert evaluate(g, q, ctx, evidence_path=path).status == "INVALID_EVIDENCE"
    payload["metrics"]["lut"]["value"] = 17
    payload["timing_status"] = "UNKNOWN"
    save(path, payload)
    assert evaluate(g, q, ctx, evidence_path=path).status == "INVALID_EVIDENCE"


def test_failed_hls_timing_never_ready(tmp_path):
    g, q, ctx, path, _, _ = write_bundle(tmp_path, estimated="5.5")
    result = evaluate(g, q, ctx, evidence_path=path)
    assert result.status == "TIMING_FAILED" and not result.pareto_eligible
    assert result.metrics["latency_us"] == .5


def test_operator_scope_cannot_claim_full_graph(tmp_path):
    g, q, ctx, path, _, _ = write_bundle(tmp_path, scope="operator_hls")
    result = evaluate(g, q, ctx, evidence_path=path)
    assert result.status == "READY" and result.evidence_level == "operator_hls"
    assert not result.pareto_eligible
    with pytest.raises(ValueError):
        replace(ctx, backend="board")
    with pytest.raises(ValueError):
        parse_hls_xml(xml(), scope="board")


def test_incomplete_route_and_unknown_timing_not_pareto(tmp_path):
    g, q, ctx, path, payload, _ = write_bundle(tmp_path, scope="route", backend="route")
    del payload["metrics"]["bram18"]
    save(path, payload)
    result = evaluate(g, q, ctx, evidence_path=path)
    assert result.status == "INCOMPLETE" and not result.pareto_eligible
    payload["metrics"]["bram18"] = dict(value=1, unit="18k_blocks")
    payload["timing_status"] = "UNKNOWN"
    save(path, payload)
    assert evaluate(g, q, ctx, evidence_path=path).status == "INCOMPLETE"


@pytest.mark.parametrize("backend,scope", [("board", "board"), ("power", "power")])
def test_board_power_artifacts_and_separate_scope(tmp_path, backend, scope):
    unknown = HardwareContext("part", 200, "policy", backend, "UNMEASURED", scope)
    g = decode(0)
    assert evaluate(g, Q(g), unknown).status == "NEEDS_MEASUREMENT"
    g, q, ctx, path, _, _ = write_bundle(tmp_path, scope=scope, backend=backend)
    result = evaluate(g, q, ctx, evidence_path=path)
    assert result.status == "READY" and result.evidence_level == scope
    assert result.pareto_eligible == (scope == "board")
    (tmp_path / "checkpoint.bin").write_bytes(b"changed checkpoint")
    assert evaluate(g, q, ctx, evidence_path=path).status == "INVALID_EVIDENCE"


def test_pure_hls_parser_unknown_resource_and_wrong_unit():
    parsed = parse_hls_xml(xml(dsp="undef", cycles="undef"), scope="operator_hls")
    assert parsed.metrics == {"lut": 17, "bram18": 1}
    assert parsed.evidence_level == "operator_hls"
    assert parse_hls_xml(xml(dsp="0"), scope="full_hls").metrics["dsp"] == 0
    with pytest.raises(ValueError):
        parse_hls_xml(xml(dsp="-1"), scope="full_hls")
    with pytest.raises(ValueError):
        parse_hls_xml("<!DOCTYPE x><Report/>", scope="full_hls")
    with pytest.raises(ValueError):
        parse_hls_xml("<Report>", scope="full_hls")


def test_duplicate_keys_and_missing_provenance_rejected(tmp_path):
    g, q, ctx, path, payload, _ = write_bundle(tmp_path)
    path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    assert evaluate(g, q, ctx, evidence_path=path).status == "INVALID_EVIDENCE"
    payload["provenance"]["sources"] = []
    save(path, payload)
    assert evaluate(g, q, ctx, evidence_path=path).status == "INVALID_EVIDENCE"


def test_invalid_raw_xml_units_rejected():
    malformed_units = xml().replace("<Part>", "<unit>us</unit><Part>")
    with pytest.raises(ValueError, match="Unsupported HLS unit"):
        parse_hls_xml(malformed_units, scope="full_hls")


def test_board_import_requires_actual_artifact_identities(tmp_path):
    g, q, ctx, path, payload, _ = write_bundle(tmp_path, scope="board", backend="board")
    ctx = replace(ctx, checkpoint_sha256=None)
    payload["context"], payload["context_hash"] = ctx.to_dict(), ctx.context_hash
    save(path, payload)
    result = evaluate(g, q, ctx, evidence_path=path)
    assert result.status == "INVALID_EVIDENCE" and "artifact identities" in result.reason


def test_hls_import_requires_implementation_and_real_tool(tmp_path):
    g, q, ctx, path, payload, _ = write_bundle(tmp_path)
    ctx = replace(ctx, implementation_sha256=None)
    payload["context"], payload["context_hash"] = ctx.to_dict(), ctx.context_hash
    save(path, payload)
    assert "artifact identities" in evaluate(g, q, ctx, evidence_path=path).reason
    ctx = replace(ctx, tool_version="UNMEASURED")
    payload["context"], payload["context_hash"] = ctx.to_dict(), ctx.context_hash
    save(path, payload)
    assert "actual tool version" in evaluate(g, q, ctx, evidence_path=path).reason


def test_malformed_raw_xml_returns_invalid_evidence(tmp_path):
    g, q, ctx, path, payload, report = write_bundle(tmp_path)
    report.write_text("<Report>", encoding="utf-8")
    payload["provenance"]["sources"][-1]["sha256"] = digest(report)
    save(path, payload)
    assert evaluate(g, q, ctx, evidence_path=path).status == "INVALID_EVIDENCE"
