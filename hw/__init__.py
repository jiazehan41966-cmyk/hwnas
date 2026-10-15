"""Shw: verified read-only evidence, context-aware caching, explicit unknowns."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from types import MappingProxyType

from space import Genotype, canonical_hash, validate_sha256
from quant import QuantizationPlan
from .backends.hls_report import parse_hls_xml

SCHEMA_VERSION = "sonar_hwnas.hardware.v1"
METRIC_UNITS = {"latency_us": "us", "lut": "count", "dsp": "count", "bram18": "18k_blocks", "power_w": "W"}
RESOURCE_METRICS = frozenset({"lut", "dsp", "bram18"})
CORE_METRICS = frozenset({"latency_us", "lut", "dsp", "bram18"})
BACKEND_SCOPES = {"hls_report": {"operator_hls", "full_hls"}, "route": {"route"}, "board": {"board"}, "power": {"power"}}
ARTIFACT_FIELDS = ("checkpoint_sha256", "quant_artifact_sha256", "bitstream_sha256", "implementation_sha256")
STATUS_VALUES = {"READY", "NEEDS_MEASUREMENT", "INVALID_EVIDENCE", "INCOMPLETE", "TIMING_FAILED"}


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, trimmed string")


@dataclass(frozen=True)
class HardwareContext:
    device: str
    clock_mhz: float
    mapping_policy: str
    backend: str
    tool_version: str
    scope: str
    checkpoint_sha256: str | None = None
    quant_artifact_sha256: str | None = None
    bitstream_sha256: str | None = None
    implementation_sha256: str | None = None

    def __post_init__(self):
        for name in ("device", "mapping_policy", "backend", "tool_version", "scope"):
            _nonempty(getattr(self, name), name)
        if self.backend not in BACKEND_SCOPES or self.scope not in BACKEND_SCOPES[self.backend]:
            raise ValueError("Hardware backend and evidence scope must agree")
        if type(self.clock_mhz) not in (int, float) or not math.isfinite(self.clock_mhz) or self.clock_mhz <= 0:
            raise ValueError("clock_mhz must be finite and positive")
        object.__setattr__(self, "clock_mhz", float(self.clock_mhz))
        for name in ARTIFACT_FIELDS:
            if getattr(self, name) is not None:
                validate_sha256(getattr(self, name))
        # Unmeasured contexts may be queried without inventing artifact hashes.
        # Actual imports require all backend-specific identities below.

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Hardware context must match its schema exactly")
        return cls(**value)

    @property
    def context_hash(self):
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class HardwareResult:
    status: str
    arch_hash: str
    q_hash: str
    context_hash: str
    metrics: dict
    evidence_level: str
    source_sha256: str | None = None
    reason: str = ""

    def __post_init__(self):
        if self.status not in STATUS_VALUES:
            raise ValueError("Unknown hardware result status")
        if self.evidence_level not in {"operator_hls", "full_hls", "route", "board", "power"}:
            raise ValueError("Unknown hardware evidence level")
        for value in (self.arch_hash, self.q_hash, self.context_hash):
            validate_sha256(value)
        if self.source_sha256 is not None:
            validate_sha256(self.source_sha256)
        _validate_metrics(self.metrics, normalized=False)
        required = {"power_w"} if self.evidence_level == "power" else CORE_METRICS
        if self.status == "READY" and (not required <= self.metrics.keys() or self.source_sha256 is None):
            raise ValueError("READY results require complete metrics and a source identity")
        if self.status in {"NEEDS_MEASUREMENT", "INVALID_EVIDENCE"} and self.metrics:
            raise ValueError("Unknown or invalid results cannot carry metrics")
        if ("power_w" in self.metrics and self.evidence_level != "power") or (self.evidence_level == "power" and set(self.metrics) - {"power_w"}):
            raise ValueError("Power metrics require their separate evidence scope")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def pareto_eligible(self):
        return (self.status == "READY" and self.evidence_level in {"full_hls", "route", "board"}
                and CORE_METRICS <= self.metrics.keys())

    def to_dict(self):
        return {"status": self.status, "arch_hash": self.arch_hash, "q_hash": self.q_hash,
                "context_hash": self.context_hash, "metrics": dict(self.metrics),
                "evidence_level": self.evidence_level, "source_sha256": self.source_sha256, "reason": self.reason}


def _validate_metrics(value, *, normalized):
    if not isinstance(value, (dict, MappingProxyType)):
        raise ValueError("Hardware metrics must be a mapping")
    if set(value) - METRIC_UNITS.keys():
        raise ValueError("Unknown hardware metric or noncanonical unit name")
    metrics = {}
    for name, raw in value.items():
        if normalized:
            if not isinstance(raw, dict) or set(raw) != {"value", "unit"} or raw["unit"] != METRIC_UNITS[name]:
                raise ValueError(f"{name} requires unit {METRIC_UNITS[name]}")
            raw = raw["value"]
        if type(raw) not in (int, float) or not math.isfinite(raw) or raw < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        if name in RESOURCE_METRICS and type(raw) is not int:
            raise ValueError(f"{name} must be an integer resource count")
        if name in {"latency_us", "power_w"} and raw <= 0:
            raise ValueError(f"{name} must be positive")
        metrics[name] = raw
    return metrics


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(raw):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs, parse_constant=invalid_constant)


def _file(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Evidence source must be a regular file")
    raw = path.read_bytes()
    return path, raw, sha256(raw).hexdigest()


def _import_evidence(path, genotype, plan, context):
    primary, primary_raw, primary_hash = _file(path)
    envelope = primary if primary.suffix.lower() != ".xml" else primary.with_suffix(primary.suffix + ".json")
    envelope, envelope_raw, envelope_hash = _file(envelope)
    value = _read_json(envelope_raw)
    expected = {"schema_version", "arch_hash", "q_hash", "context", "context_hash", "evidence_level", "timing_status", "metrics", "provenance"}
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported normalized hardware evidence schema")
    declared_context = HardwareContext.from_dict(value["context"])
    if (value["arch_hash"] != genotype.arch_hash or value["q_hash"] != plan.q_hash
            or declared_context != context or value["context_hash"] != context.context_hash
            or value["evidence_level"] != context.scope):
        raise ValueError("Evidence candidate, quantization, context or scope identity mismatch")
    if context.tool_version.upper() in {"UNMEASURED", "UNKNOWN", "N/A"}:
        raise ValueError("Imported evidence requires an actual tool version")
    required_artifacts = ("checkpoint_sha256", "quant_artifact_sha256", "bitstream_sha256") if context.backend in {"board", "power"} else ("implementation_sha256",)
    if any(getattr(context, name) is None for name in required_artifacts):
        raise ValueError(f"Imported {context.backend} evidence requires artifact identities: {', '.join(required_artifacts)}")
    metrics = _validate_metrics(value["metrics"], normalized=True)
    if "power_w" in metrics and context.scope != "power":
        raise ValueError("Power requires its separate power evidence scope")
    if context.scope == "power" and set(metrics) - {"power_w"}:
        raise ValueError("Power evidence cannot attest to unrelated resource or latency metrics")
    timing = value["timing_status"]
    if timing not in {"PASS", "FAIL", "UNKNOWN"}:
        raise ValueError("timing_status must be PASS, FAIL or UNKNOWN")
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"kind", "producer", "sources"} or provenance["kind"] != context.scope:
        raise ValueError("Provenance must declare its exact evidence kind")
    _nonempty(provenance["producer"], "provenance producer")
    if not isinstance(provenance["sources"], list) or not provenance["sources"]:
        raise ValueError("Provenance sources are required")
    checks = {str(primary): primary_hash, str(envelope): envelope_hash}
    sources = {}
    for entry in provenance["sources"]:
        if not isinstance(entry, dict) or set(entry) != {"role", "path", "sha256"}:
            raise ValueError("Each provenance source needs role, path and sha256")
        _nonempty(entry["role"], "source role")
        _nonempty(entry["path"], "source path")
        validate_sha256(entry["sha256"])
        if entry["role"] in sources:
            raise ValueError("Provenance source roles must be unique")
        source = Path(entry["path"])
        source = source if source.is_absolute() else envelope.parent / source
        source, source_raw, digest = _file(source)
        if digest != entry["sha256"]:
            raise ValueError(f"Source content hash mismatch: {entry['role']}")
        if source == envelope:
            raise ValueError("Normalized evidence cannot attest to itself")
        sources[entry["role"]] = (source, source_raw, digest)
        checks[str(source)] = digest
    if "report" not in sources:
        raise ValueError("A raw report source is required")
    if context.backend in {"hls_report", "route"} and context.implementation_sha256 is None:
        raise ValueError("Imported HLS/route evidence requires implementation_sha256")
    for name in ARTIFACT_FIELDS:
        digest = getattr(context, name)
        if digest is not None:
            role = name.removesuffix("_sha256")
            if role not in sources or sources[role][2] != digest:
                raise ValueError(f"Context artifact {role} is missing or mismatched")
    if primary.suffix.lower() == ".xml" and sources["report"][0] != primary:
        raise ValueError("XML sidecar is bound to a different report")
    if context.backend == "hls_report":
        if sources["report"][0].suffix.lower() != ".xml":
            raise ValueError("HLS normalization requires its raw XML report")
        parsed = parse_hls_xml(sources["report"][1], scope=context.scope)
        if (parsed.device != context.device or parsed.tool_version != context.tool_version
                or parsed.clock_mhz is None or not math.isclose(parsed.clock_mhz, context.clock_mhz, rel_tol=1e-9)):
            raise ValueError("HLS XML device, target clock or tool identity is absent/mismatched")
        if parsed.metrics != metrics:
            raise ValueError("Normalized metrics differ from raw HLS XML")
        if parsed.timing_status != timing:
            raise ValueError("Normalized timing differs from raw HLS XML")
    required = {"power_w"} if context.scope == "power" else CORE_METRICS
    if timing == "FAIL":
        status, reason = "TIMING_FAILED", "Evidence does not meet the declared target clock"
    elif timing != "PASS" or not required <= metrics.keys():
        status, reason = "INCOMPLETE", "Missing required metrics or verified timing status"
    else:
        status, reason = "READY", "Verified evidence at the declared scope only"
    return HardwareResult(status, genotype.arch_hash, plan.q_hash, context.context_hash,
                          metrics, context.scope, primary_hash, reason), checks


def _cache(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS hardware_v1 (arch_hash TEXT NOT NULL, q_hash TEXT NOT NULL, context_hash TEXT NOT NULL, evidence_path TEXT NOT NULL, checks_json TEXT NOT NULL, PRIMARY KEY (arch_hash, q_hash, context_hash))")
    return connection


def evaluate(genotype, plan, context, *, cache_path=None, evidence_path=None):
    """Import verified evidence or return unknown; never measure or estimate a gap.

    The cache stores references, not trusted metrics. Every hit rehashes every
    source and revalidates the envelope, so deleted/changed evidence is rejected.
    """
    if not isinstance(genotype, Genotype) or not isinstance(plan, QuantizationPlan) or not isinstance(context, HardwareContext):
        raise TypeError("evaluate requires Genotype, QuantizationPlan and HardwareContext")
    if plan.arch_hash != genotype.arch_hash:
        raise ValueError("Quantization plan belongs to a different architecture")
    key = (genotype.arch_hash, plan.q_hash, context.context_hash)
    connection = _cache(cache_path) if cache_path is not None else None
    try:
        checks = None
        if evidence_path is None and connection is not None:
            cached = connection.execute("SELECT evidence_path, checks_json FROM hardware_v1 WHERE arch_hash=? AND q_hash=? AND context_hash=?", key).fetchone()
            if cached is not None:
                evidence_path, checks_raw = cached
                checks = _read_json(checks_raw.encode("utf-8"))
        if evidence_path is None:
            return HardwareResult("NEEDS_MEASUREMENT", *key, {}, context.scope, reason="No evidence is registered for this exact context")
        try:
            if checks is not None:
                if not isinstance(checks, dict) or not checks:
                    raise ValueError("Invalid cache provenance")
                for source, digest in checks.items():
                    validate_sha256(digest)
                    if _file(source)[2] != digest:
                        raise ValueError("Cached evidence source content hash mismatch")
            result, verified = _import_evidence(evidence_path, genotype, plan, context)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            if connection is not None:
                connection.execute("DELETE FROM hardware_v1 WHERE arch_hash=? AND q_hash=? AND context_hash=?", key)
                connection.commit()
            return HardwareResult("INVALID_EVIDENCE", *key, {}, context.scope, reason=str(exc))
        if connection is not None and result.status == "READY":
            connection.execute("INSERT OR REPLACE INTO hardware_v1 VALUES (?,?,?,?,?)", (*key, str(Path(evidence_path).resolve()), json.dumps(verified, sort_keys=True)))
            connection.commit()
        return result
    finally:
        if connection is not None:
            connection.close()
