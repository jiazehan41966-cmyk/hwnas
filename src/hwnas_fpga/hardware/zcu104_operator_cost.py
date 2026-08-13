"""Strict operator-cost adapter for the frozen ZCU104 CFEM20 search space.

The fresh campaign measures the two searchable positions only.  Fixed network
layers remain part of parameter/MAC accounting, but their historical or
analytic hardware costs must not leak into this campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from hwnas_fpga.hardware.cost import CostEstimate, FPGACostEstimator
from hwnas_fpga.search_space import ArchitectureSpec, SearchSpace


SEARCH_SPACE_VERSION = "fourstage_cfem20_v1"
FORMAL_DATA_KIND = "FORMAL_ZCU104_MEASURED"
TEST_DATA_KIND = "TEST_ONLY_SYNTHETIC"
EXPECTED_TARGET = {
    "board": "ZCU104",
    "part": "xczu7ev-ffvc1156-2-e",
    "clock_period_ns": 5.0,
    "toolchain": "2022.2",
}
REQUIRED_METRICS = (
    "latency_board_ms",
    "lut",
    "dsp",
    "bram36",
    "p_dynamic_w",
    "energy_dynamic_mj_per_op",
    "wns_ns",
    "tns_ns",
)
REQUIRED_PROVENANCE = (
    "source_commit",
    "source_freeze_sha256",
    "hls_config_sha256",
    "bitstream_sha256",
    "runtime_sha256",
)
EXPECTED_ROWS = {
    "stage2__k3_e3__16x112x112_to_24x56x56__s2": ("stage2", "k3_e3", [16, 112, 112], [24, 56, 56], 2),
    "stage2__k3_e6__16x112x112_to_24x56x56__s2": ("stage2", "k3_e6", [16, 112, 112], [24, 56, 56], 2),
    "stage2__k5_e3__16x112x112_to_24x56x56__s2": ("stage2", "k5_e3", [16, 112, 112], [24, 56, 56], 2),
    "stage2__k5_e6__16x112x112_to_24x56x56__s2": ("stage2", "k5_e6", [16, 112, 112], [24, 56, 56], 2),
    "stage4__skip__32x28x28_to_32x28x28__s1": ("stage4", "skip", [32, 28, 28], [32, 28, 28], 1),
    "stage4__mbconv_k3_e3__32x28x28_to_32x28x28__s1": ("stage4", "mbconv_k3_e3", [32, 28, 28], [32, 28, 28], 1),
    "stage4__mbconv_k5_e3__32x28x28_to_32x28x28__s1": ("stage4", "mbconv_k5_e3", [32, 28, 28], [32, 28, 28], 1),
    "stage4__dmdc_conv_sonar__32x28x28_to_32x28x28__s1": ("stage4", "dmdc_conv_sonar", [32, 28, 28], [32, 28, 28], 1),
    "stage4__cfem_sonar__32x28x28_to_32x28x28__s1": ("stage4", "cfem_sonar", [32, 28, 28], [32, 28, 28], 1),
}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_hashed_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "formal ZCU104 operator-cost input is missing; search remains "
            f"WAITING_FOR_FORMAL_OPERATOR_COST_LUT: {resolved}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    expected = str(payload.get("payload_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    actual = canonical_sha256(unsigned)
    if expected != actual:
        raise ValueError(
            f"operator-cost payload hash mismatch for {resolved}: {expected} != {actual}"
        )
    return resolved, payload


def _require_exact_target(row: Mapping[str, Any]) -> None:
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"cost row {row.get('cost_key')} has no target binding")
    for key, expected in EXPECTED_TARGET.items():
        observed = target.get(key)
        if key == "clock_period_ns":
            matches = observed is not None and float(observed) == float(expected)
        else:
            matches = str(observed) == str(expected)
        if not matches:
            raise ValueError(
                f"cost row {row.get('cost_key')} has wrong target {key}: "
                f"{observed!r} != {expected!r}"
            )


class ZCU104CFEM20OperatorCostEstimator(FPGACostEstimator):
    """Drop-in estimator that replaces variable-position cost with fresh rows."""

    def __init__(
        self,
        *args: Any,
        operator_cost_path: str | Path,
        coverage_path: str | Path,
        allow_test_data: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(self.hardware_spec.clock_mhz) != 200 or "ZCU104" not in self.hardware_spec.name:
            raise ValueError("CFEM20 operator costs require the 200 MHz ZCU104 hardware profile")
        if self.lut_query_engine is not None:
            raise ValueError("CFEM20 fresh operator costs forbid a legacy/general LUT engine")
        if any(float(value) != 1.0 for value in self.analytic_calibration.values()):
            raise ValueError("CFEM20 fresh operator costs forbid analytic calibration reuse")
        self.operator_cost_path, lut = _load_hashed_json(operator_cost_path)
        self.coverage_path, coverage = _load_hashed_json(coverage_path)
        self.operator_cost_file_sha256 = sha256_file(self.operator_cost_path)
        self.coverage_file_sha256 = sha256_file(self.coverage_path)
        self.data_kind = str(lut.get("data_kind") or "")
        if self.data_kind == TEST_DATA_KIND:
            if not allow_test_data:
                raise ValueError(
                    "TEST_ONLY_SYNTHETIC operator costs require explicit allow_test_data=true"
                )
        elif self.data_kind != FORMAL_DATA_KIND:
            raise ValueError(
                "operator-cost table must declare data_kind=FORMAL_ZCU104_MEASURED; "
                "scaffolds and historical tables are not search inputs"
            )
        if lut.get("search_space_version") != SEARCH_SPACE_VERSION:
            raise ValueError("operator-cost table is bound to the wrong search space")
        if coverage.get("search_space_version") != SEARCH_SPACE_VERSION:
            raise ValueError("coverage audit is bound to the wrong search space")
        if lut.get("search_ready") is not True or coverage.get("search_ready") is not True:
            raise ValueError(
                "fresh ZCU104 operator-cost table is not search_ready; wait for 9/9 accepted rows"
            )
        rows = lut.get("rows")
        mappings = coverage.get("candidate_mappings")
        if not isinstance(rows, list) or len(rows) != 9:
            raise ValueError("operator-cost table must contain exactly 9 rows")
        if not isinstance(mappings, list) or len(mappings) != 20:
            raise ValueError("coverage audit must contain exactly 20 candidate mappings")
        self.rows_by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("operator-cost rows must be JSON objects")
            key = str(row.get("cost_key") or "")
            if not key or key in self.rows_by_key:
                raise ValueError(f"missing or duplicate operator cost key: {key!r}")
            if row.get("search_space_version") != SEARCH_SPACE_VERSION:
                raise ValueError(f"cost row {key} has wrong search-space version")
            if row.get("measurement_status") != "ACCEPTED_FRESH_ZCU104_MEASUREMENT":
                raise ValueError(f"cost row {key} is not an accepted fresh measurement")
            if row.get("gate_status") != "PASS":
                raise ValueError(f"cost row {key} did not pass its hardware gate")
            if row.get("historical_cost_row_allowed") is not False:
                raise ValueError(f"cost row {key} does not explicitly forbid historical reuse")
            expected_identity = EXPECTED_ROWS.get(key)
            observed_identity = (
                row.get("stage"),
                row.get("operator_id"),
                row.get("input_shape_chw"),
                row.get("output_shape_chw"),
                row.get("stride"),
            )
            if expected_identity is None or observed_identity != expected_identity:
                raise ValueError(f"cost row {key} has an identity or exact-shape mismatch")
            _require_exact_target(row)
            for field in REQUIRED_PROVENANCE:
                value = str(row.get(field) or "")
                expected_length = 40 if field == "source_commit" else 64
                if len(value) != expected_length or any(ch not in "0123456789abcdef" for ch in value.lower()):
                    raise ValueError(f"cost row {key} has invalid provenance field {field}")
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"cost row {key} has no metrics object")
            missing = [name for name in REQUIRED_METRICS if metrics.get(name) is None]
            if missing:
                raise ValueError(f"cost row {key} is incomplete: {', '.join(missing)}")
            numeric_metrics = {name: float(metrics[name]) for name in REQUIRED_METRICS}
            if any(not math.isfinite(value) for value in numeric_metrics.values()):
                raise ValueError(f"cost row {key} contains non-finite metrics")
            if numeric_metrics["latency_board_ms"] <= 0:
                raise ValueError(f"cost row {key} has non-positive board latency")
            for name in (
                "lut",
                "dsp",
                "bram36",
                "p_dynamic_w",
                "energy_dynamic_mj_per_op",
            ):
                if numeric_metrics[name] < 0:
                    raise ValueError(f"cost row {key} has negative metric {name}")
            if float(metrics["wns_ns"]) < 0 or float(metrics["tns_ns"]) != 0:
                raise ValueError(f"cost row {key} is not route-clean")
            self.rows_by_key[key] = row
        if set(self.rows_by_key) != set(EXPECTED_ROWS):
            raise ValueError("operator-cost table does not contain the exact frozen 9-key set")
        if coverage.get("operator_cost_payload_sha256") != lut.get("payload_sha256"):
            raise ValueError("coverage audit is not bound to this operator-cost payload")
        self.mapping_by_sha: dict[str, dict[str, Any]] = {}
        for mapping in mappings:
            if not isinstance(mapping, dict):
                raise ValueError("candidate mappings must be JSON objects")
            arch_sha = str(mapping.get("architecture_sha256") or "")
            keys = (mapping.get("stage2_cost_key"), mapping.get("stage4_cost_key"))
            if len(arch_sha) != 64 or arch_sha in self.mapping_by_sha:
                raise ValueError(f"invalid or duplicate architecture SHA256: {arch_sha!r}")
            if any(key not in self.rows_by_key for key in keys):
                raise ValueError(f"candidate {arch_sha} references a missing cost key")
            if mapping.get("mapping_status") != "PASS_MEASURED_COSTS_BOUND":
                raise ValueError(f"candidate {arch_sha} mapping is not measurement-bound")
            self.mapping_by_sha[arch_sha] = mapping
        from hwnas_fpga.fourstage_operator import architecture_sha256, enumerate_cfem20

        expected_mappings: dict[str, dict[str, str]] = {}
        for candidate in enumerate_cfem20():
            stage2 = candidate.architecture.stages[1].blocks[0]
            stage4 = candidate.architecture.stages[3].blocks[0]
            stage2_key = (
                f"stage2__k{stage2.kernel_size}_e{stage2.expand_ratio}__"
                "16x112x112_to_24x56x56__s2"
            )
            stage4_operator_id = (
                f"mbconv_k{stage4.kernel_size}_e{stage4.expand_ratio}"
                if stage4.op == "mbconv"
                else stage4.op
            )
            stage4_key = (
                f"stage4__{stage4_operator_id}__32x28x28_to_32x28x28__s1"
            )
            expected_mappings[architecture_sha256(candidate.architecture)] = {
                "architecture_id": candidate.arch_id,
                "stage2_cost_key": stage2_key,
                "stage4_cost_key": stage4_key,
            }
        if set(self.mapping_by_sha) != set(expected_mappings):
            raise ValueError("candidate mapping is not the exact frozen RAW20 SHA256 set")
        for arch_sha, expected_mapping in expected_mappings.items():
            observed = self.mapping_by_sha[arch_sha]
            for field, expected in expected_mapping.items():
                if observed.get(field) != expected:
                    raise ValueError(
                        f"candidate {arch_sha} has wrong {field}: "
                        f"{observed.get(field)!r} != {expected!r}"
                    )
        coverage_requirements = {
            "operator_key_definition_coverage": 9,
            "raw_candidate_key_mapping_coverage": 20,
            "measured_operator_coverage": 9,
            "measured_raw_candidate_coverage": 20,
        }
        for name, expected_count in coverage_requirements.items():
            record = coverage.get(name, {})
            if (
                int(record.get("covered", 0)) != expected_count
                or int(record.get("expected", 0)) != expected_count
                or record.get("status") != "PASS"
            ):
                raise ValueError(
                    f"coverage audit does not establish PASS {expected_count}/{expected_count} "
                    f"for {name}"
                )
        self._operator_cost_queries = 0
        self._operator_cost_hits = 0
        self._complexity_cache: dict[str, tuple[int, int]] = {}
        if self.constraints is not None and (
            self.constraints.max_memory_bandwidth_gbps is not None
            or self.constraints.max_offchip_mem_mb is not None
        ):
            raise ValueError(
                "CFEM20 exact-shape operator rows do not establish memory-bandwidth "
                "or off-chip-memory gates; those constraints must remain unset"
            )

    def estimate(self, architecture: ArchitectureSpec, search_space: SearchSpace) -> CostEstimate:
        from hwnas_fpga.fourstage_operator import validate_frozen_fourstage

        if int(search_space.config.image_size) != 224:
            raise ValueError("fourstage_cfem20_v1 exact-shape costs require image_size=224")
        validate_frozen_fourstage(architecture)
        arch_sha = canonical_sha256(architecture.to_dict())
        mapping = self.mapping_by_sha.get(arch_sha)
        self._operator_cost_queries += 1
        if mapping is None:
            raise ValueError(
                f"architecture {arch_sha} is outside frozen {SEARCH_SPACE_VERSION} RAW20"
            )
        selected = [
            self.rows_by_key[str(mapping["stage2_cost_key"])],
            self.rows_by_key[str(mapping["stage4_cost_key"])],
        ]
        self._operator_cost_hits += 1
        complexity = self._complexity_cache.get(arch_sha)
        if complexity is None:
            from hwnas_fpga.fourstage_operator import parameter_and_mac_count

            measured_complexity = parameter_and_mac_count(architecture)
            complexity = (
                int(measured_complexity["parameter_count"]),
                int(measured_complexity["macs"]),
            )
            self._complexity_cache[arch_sha] = complexity
        metrics = [row["metrics"] for row in selected]
        latency_ms = sum(float(row["latency_board_ms"]) for row in metrics)
        energy_mj = sum(float(row["energy_dynamic_mj_per_op"]) for row in metrics)
        total_lut = sum(int(row["lut"]) for row in metrics)
        total_dsp = sum(int(row["dsp"]) for row in metrics)
        total_bram = sum(int(row["bram36"]) for row in metrics)
        power_w = max(float(row["p_dynamic_w"]) for row in metrics)
        latency_cycles = int(round(latency_ms * float(self.hardware_spec.clock_mhz) * 1000.0))
        violations = list(
            self._check_constraints(
                latency_ms=latency_ms,
                energy_mj=energy_mj,
                model_size_mb=complexity[0] * self._bytes_per_scalar / (1024**2),
                resource_dsp=total_dsp,
                resource_bram=total_bram,
                resource_lut=total_lut,
                power_w=power_w,
                memory_bandwidth_gbps=0.0,
                offchip_mem_mb=0.0,
            )
        )
        resolved_blocks = search_space.resolve_blocks(architecture)
        violations.extend(self._check_operator_policy(resolved_blocks))
        physical_metrics = {
            "zcu104_operator_cost_rows": 2.0,
            "full_network_route_established": 0.0,
            "full_network_power_established": 0.0,
            "memory_cost_established": 0.0,
        }
        return CostEstimate(
            params=complexity[0],
            macs=complexity[1],
            model_size_mb=complexity[0] * self._bytes_per_scalar / (1024**2),
            peak_activation_bytes=0,
            peak_weight_bytes=0,
            peak_buffer_bytes=0,
            peak_dsp=max(int(row["dsp"]) for row in metrics),
            peak_bram=max(int(row["bram36"]) for row in metrics),
            peak_lut=max(int(row["lut"]) for row in metrics),
            total_dsp=total_dsp,
            total_bram=total_bram,
            total_lut=total_lut,
            latency_cycles=latency_cycles,
            latency_ms=latency_ms,
            power_w=power_w,
            energy_mj=energy_mj,
            memory_bandwidth_gbps=0.0,
            offchip_mem_mb=0.0,
            violations=tuple(dict.fromkeys(violations)),
            per_layer=(),
            physical_metrics=physical_metrics,
        )

    def reset_lut_stats(self) -> None:
        super().reset_lut_stats()
        self._operator_cost_queries = 0
        self._operator_cost_hits = 0

    def get_lut_stats(self) -> dict[str, Any]:
        stats = super().get_lut_stats()
        stats.update(
            {
                "cost_source": "zcu104_cfem20_exact_shape_operator_cost_v1",
                "data_kind": self.data_kind,
                "operator_cost_queries": self._operator_cost_queries,
                "operator_cost_hits": self._operator_cost_hits,
                "operator_rows": len(self.rows_by_key),
                "candidate_mappings": len(self.mapping_by_sha),
                "operator_cost_file": str(self.operator_cost_path),
                "operator_cost_file_sha256": self.operator_cost_file_sha256,
                "coverage_file": str(self.coverage_path),
                "coverage_file_sha256": self.coverage_file_sha256,
                "hardware_cost_scope": "stage2_plus_stage4_searchable_operators_only",
                "aggregation": {
                    "latency_board_ms": "sum",
                    "energy_dynamic_mj_per_op": "sum",
                    "lut_dsp_bram36": "sum",
                    "p_dynamic_w": "max_proxy_not_network_power_measurement",
                },
                "fallback_rate": 0.0,
                "true_misses": 0,
            }
        )
        return stats
