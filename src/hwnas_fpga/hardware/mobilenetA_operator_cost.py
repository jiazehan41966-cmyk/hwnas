"""Formal ZCU104 operator-cost estimator for MobileNetA DMDC/CFEM search."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from hwnas_fpga.hardware.cost import CostEstimate, FPGACostEstimator
from hwnas_fpga.search_space import ArchitectureSpec, ResolvedBlockSpec, SearchSpace


SEARCH_SPACE_VERSION = "mobilenetA_full_dmdc_cfem_v1"
HARDWARE_LUT_VERSION = "mobilenetA_full_dmdc_cfem_v3_gap1280"
SUPPORTED_LUT_VERSIONS = {HARDWARE_LUT_VERSION}
FORMAL_DATA_KIND = "FRESH_ZCU104_HLS_ROUTE_OPERATOR_COST_WITH_RECOVERY"
EXPECTED_BOARD = "ZCU104"
EXPECTED_PART = "xczu7ev-ffvc1156-2-e"
EXPECTED_CLOCK_MHZ = 200.0
EXPECTED_ROWS = 88
ZCU104_LUTRAM_COMPATIBLE_SITES = 101_760
ZCU104_BRAM18_SITES = 624
REQUIRED_COST_FIELDS = (
    "latency_cycles_hls",
    "latency_ms_hls",
    "lut_post_route",
    "ff_post_route",
    "dsp_post_route",
    "bram_tile_post_route",
    "power_w_vivado_vectorless",
)

MAIN_MBCONV_SHAPES = {
    (16, 24, 112, 56, 2): "mbconv_stage2_112_16_24_s2",
    (24, 32, 56, 28, 2): "mbconv_stage3_56_24_32_s2",
    (32, 64, 28, 14, 2): "mbconv_stage4_28_32_64_s2",
    (64, 96, 14, 14, 1): "mbconv_stage5_14_64_96_s1",
    (96, 160, 14, 7, 2): "mbconv_stage6_14_96_160_s2",
    (160, 320, 7, 7, 1): "mbconv_stage7_7_160_320_s1",
}


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing formal MobileNetA operator-cost input: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
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


class ZCU104MobileNetAOperatorCostEstimator(FPGACostEstimator):
    """Aggregate exact formal LUT rows for the MobileNetA search architecture."""

    cost_source = "zcu104_mobilenetA_full_operator_cost_lut_v1"

    def __init__(
        self,
        *args: Any,
        operator_cost_path: str | Path,
        coverage_path: str | Path,
        preflight_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if float(self.hardware_spec.clock_mhz) != 200.0 or "ZCU104" not in self.hardware_spec.name:
            raise ValueError("MobileNetA formal operator costs require the 200 MHz ZCU104 hardware profile")
        if self.lut_query_engine is not None:
            raise ValueError("MobileNetA formal operator costs forbid the legacy/general LUT engine")
        if any(float(value) != 1.0 for value in self.analytic_calibration.values()):
            raise ValueError("MobileNetA formal operator costs forbid analytic calibration reuse")

        self.operator_cost_path, lut = load_hashed_json(operator_cost_path)
        self.coverage_path, coverage = load_hashed_json(coverage_path)
        self.search_ready = bool(lut.get("search_ready"))
        self.operator_cost_payload_sha256 = str(lut.get("payload_sha256") or "")
        self.preflight_path: Path | None = None
        self.preflight_file_sha256: str | None = None
        if preflight_path:
            self.preflight_path, preflight = load_hashed_json(preflight_path)
            if preflight.get("status") != "PASS_FORMAL_ZCU104_OPERATOR_COST_LUT_PREFLIGHT":
                raise ValueError("MobileNetA formal LUT preflight has not passed")
            if preflight.get("lut", {}).get("payload_sha256") != lut.get("payload_sha256"):
                raise ValueError("MobileNetA preflight is not bound to this operator-cost LUT")
            self.preflight_file_sha256 = sha256_file(self.preflight_path)

        self.operator_cost_file_sha256 = sha256_file(self.operator_cost_path)
        self.coverage_file_sha256 = sha256_file(self.coverage_path)
        self.data_kind = str(lut.get("data_kind") or "")
        self.hardware_lut_version = str(lut.get("hardware_lut_version") or "")
        self.search_space_version = str(lut.get("search_space_version") or "")
        self._validate_payloads(lut, coverage)

        rows = lut.get("rows")
        if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
            raise ValueError("MobileNetA operator-cost LUT must contain exactly 88 rows")
        self.rows_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        self.rows_by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            self._validate_row(row)
            key = str(row["row_key"])
            pair = (str(row["operator"]), str(row["shape_name"]))
            if key in self.rows_by_key or pair in self.rows_by_pair:
                raise ValueError(f"duplicate MobileNetA LUT row: {key}")
            self.rows_by_key[key] = dict(row)
            self.rows_by_pair[pair] = dict(row)

        gap = self.rows_by_pair.get(("global_avg_pool", "gap_7_1280"))
        if gap is None:
            raise ValueError("MobileNetA requires measured global_avg_pool/gap_7_1280; the historical 320-channel GAP is not a substitute")
        spec = gap.get("op_spec") or {}
        if (spec.get("in_channels"), spec.get("out_channels"), spec.get("input_resolution")) != (1280, 1280, [7, 7]):
            raise ValueError("gap_7_1280 op_spec must describe 1280 channels at 7x7")

        self._operator_cost_queries = 0
        self._operator_cost_hits = 0
        self._operator_cost_misses = 0
        self.last_selected_row_keys: list[str] = []

    def _validate_payloads(
        self,
        lut: Mapping[str, Any],
        coverage: Mapping[str, Any],
    ) -> None:
        for prefix, payload in (("lut", lut), ("coverage", coverage)):
            if payload.get("data_kind") != FORMAL_DATA_KIND:
                raise ValueError(f"{prefix} is not the formal recovered MobileNetA ZCU104 LUT data kind")
            if payload.get("search_space_version") != SEARCH_SPACE_VERSION:
                raise ValueError(f"{prefix} is bound to the wrong search space")
            if payload.get("hardware_lut_version") not in SUPPORTED_LUT_VERSIONS:
                raise ValueError(f"{prefix} is bound to the wrong hardware LUT version")
            if payload.get("search_ready") is not True:
                raise ValueError(f"{prefix} is not search_ready")
        if coverage.get("operator_cost_payload_sha256") != lut.get("payload_sha256"):
            raise ValueError("coverage audit is not bound to this operator-cost payload")
        if coverage.get("hardware_lut_version") != lut.get("hardware_lut_version"):
            raise ValueError("coverage audit and LUT versions differ")
        if int(coverage.get("expected_rows") or 0) != EXPECTED_ROWS:
            raise ValueError("coverage audit expected row count is not 88")
        if int(coverage.get("ready_rows") or 0) != EXPECTED_ROWS:
            raise ValueError("coverage audit ready row count is not 88")

    def _validate_row(self, row: Any) -> None:
        if not isinstance(row, Mapping):
            raise ValueError("MobileNetA operator-cost rows must be JSON objects")
        for field in ("row_key", "operator", "shape_name", "cost"):
            if not row.get(field):
                raise ValueError(f"MobileNetA LUT row is missing {field}")
        if row.get("ready") is not True:
            raise ValueError(f"MobileNetA LUT row is not ready: {row.get('row_key')}")
        if row.get("target_board") != EXPECTED_BOARD:
            raise ValueError(f"MobileNetA LUT row has wrong board: {row.get('row_key')}")
        if row.get("target_part") != EXPECTED_PART:
            raise ValueError(f"MobileNetA LUT row has wrong part: {row.get('row_key')}")
        if float(row.get("target_clock_mhz") or 0.0) != EXPECTED_CLOCK_MHZ:
            raise ValueError(f"MobileNetA LUT row has wrong clock: {row.get('row_key')}")
        op_spec = row.get("op_spec")
        if isinstance(op_spec, Mapping) and op_spec.get("search_space_version") not in {
            None,
            SEARCH_SPACE_VERSION,
        }:
            raise ValueError(f"MobileNetA LUT row has wrong op_spec search space: {row.get('row_key')}")
        cost = row.get("cost")
        if not isinstance(cost, Mapping):
            raise ValueError(f"MobileNetA LUT row has no cost object: {row.get('row_key')}")
        missing = [field for field in REQUIRED_COST_FIELDS if cost.get(field) is None]
        if missing:
            raise ValueError(f"MobileNetA LUT row is incomplete: {row.get('row_key')}")
        for field in REQUIRED_COST_FIELDS:
            value = float(cost[field])
            if not math.isfinite(value):
                raise ValueError(f"MobileNetA LUT row contains non-finite {field}: {row.get('row_key')}")
        if float(cost["latency_ms_hls"]) <= 0.0 or int(cost["latency_cycles_hls"]) <= 0:
            raise ValueError(f"MobileNetA LUT row has non-positive latency: {row.get('row_key')}")
        for field in ("lut_post_route", "ff_post_route", "dsp_post_route", "bram_tile_post_route"):
            if float(cost[field]) < 0.0:
                raise ValueError(f"MobileNetA LUT row has negative {field}: {row.get('row_key')}")

    def _row(self, operator: str, shape_name: str) -> dict[str, Any]:
        self._operator_cost_queries += 1
        row = self.rows_by_pair.get((operator, shape_name))
        if row is None:
            self._operator_cost_misses += 1
            raise ValueError(f"MobileNetA formal LUT row missing: {operator}/{shape_name}")
        self._operator_cost_hits += 1
        return row

    def _row_for_block(self, block: ResolvedBlockSpec) -> dict[str, Any]:
        if block.op == "conv" and (
            block.in_channels,
            block.out_channels,
            block.input_resolution,
            block.output_resolution,
            block.stride,
            block.kernel_size,
        ) == (32, 16, 112, 112, 1, 1):
            return self._row("pw_conv", "pw_ir16_proj_112_32_16")

        if block.op == "mbconv" and (
            block.in_channels,
            block.out_channels,
            block.input_resolution,
            block.output_resolution,
            block.stride,
        ) == (32, 32, 28, 28, 1):
            if block.expand_ratio != 3 or block.kernel_size not in {3, 5}:
                raise ValueError("target slot MBConv formal LUT supports only k3/e3 and k5/e3")
            return self._row(
                f"mbconv_target_slot_k{block.kernel_size}_e3",
                "target_slot_32x28x28_to_32x28x28_s1",
            )

        if block.op == "dmdc_conv_sonar":
            if (
                block.in_channels,
                block.out_channels,
                block.input_resolution,
                block.output_resolution,
                block.stride,
            ) != (32, 32, 28, 28, 1):
                raise ValueError("DMDC formal LUT row is available only for the target slot")
            return self._row(
                "dmdc_conv_sonar_target_slot",
                "target_slot_32x28x28_to_32x28x32_s1".replace("32x28x32", "32x28x28"),
            )

        if block.op == "cfem_sonar":
            if (
                block.in_channels,
                block.out_channels,
                block.input_resolution,
                block.output_resolution,
                block.stride,
            ) != (32, 32, 28, 28, 1):
                raise ValueError("CFEM formal LUT row is available only for the target slot")
            return self._row(
                "cfem_sonar_target_slot",
                "target_slot_32x28x28_to_32x28x28_s1",
            )

        if block.op == "skip":
            if block.stride != 1 or block.in_channels != block.out_channels:
                raise ValueError("skip formal LUT rows are same-channel stride-1 only")
            return self._row("skip", f"skip_{block.input_resolution}_{block.in_channels}")

        if block.op == "mbconv":
            shape = MAIN_MBCONV_SHAPES.get(
                (
                    block.in_channels,
                    block.out_channels,
                    block.input_resolution,
                    block.output_resolution,
                    block.stride,
                )
            )
            if shape is None:
                raise ValueError(
                    "MobileNetA MBConv block is outside formal LUT shapes: "
                    f"{block.in_channels}->{block.out_channels} "
                    f"{block.input_resolution}->{block.output_resolution} s{block.stride}"
                )
            if block.kernel_size not in {3, 5} or block.expand_ratio not in {3, 6}:
                raise ValueError("MobileNetA MBConv formal LUT supports only k3/k5 and e3/e6")
            return self._row(f"mbconv_e{block.expand_ratio}_k{block.kernel_size}", shape)

        raise ValueError(f"operator is outside MobileNetA formal LUT scope: {block.op}")

    def _selected_rows(
        self,
        architecture: ArchitectureSpec,
        search_space: SearchSpace,
    ) -> list[dict[str, Any]]:
        if search_space.config.image_size != 224:
            raise ValueError("MobileNetA formal LUT requires image_size=224")
        if search_space.config.stage_count != 8:
            raise ValueError("MobileNetA formal LUT requires the 8-stage architecture")
        if architecture.head_conv_channels != 1280:
            raise ValueError("MobileNetA formal LUT requires head_conv_channels=1280")
        if architecture.num_classes not in {None, 8}:
            raise ValueError("MobileNetA formal LUT is bound to the 8-class NKSID head")

        rows: list[dict[str, Any]] = [self._row("stem_conv_k3_s2", "stem_224_1_32_s2")]
        for block in search_space.resolve_blocks(architecture):
            rows.append(self._row_for_block(block))
        rows.extend(
            (
                self._row("pw_conv", "pw_head_7_320_1280"),
                self._row("global_avg_pool", "gap_7_1280"),
                self._row("fc_layer", "fc_1_1280_8"),
            )
        )
        return rows

    @staticmethod
    def _cost(row: Mapping[str, Any]) -> Mapping[str, Any]:
        cost = row.get("cost")
        if not isinstance(cost, Mapping):
            raise ValueError(f"MobileNetA LUT row has invalid cost object: {row.get('row_key')}")
        return cost

    def _operating_latency_ms(self, cost: Mapping[str, Any]) -> float:
        return float(cost["latency_cycles_hls"]) / (float(self.hardware_spec.clock_mhz) * 1000.0)

    def _zcu104_route_pressure_metrics(
        self,
        *,
        architecture: ArchitectureSpec,
        costs: list[Mapping[str, Any]],
        total_lut: int,
        total_bram: int,
    ) -> dict[str, float]:
        physical = self._physical_constraints()
        lutram_sites = self._optional_float(physical.get("board_max_lutram"))
        if lutram_sites is None:
            lutram_sites = float(ZCU104_LUTRAM_COMPATIBLE_SITES)
        bram18_sites = self._optional_float(physical.get("board_max_bram18"))
        if bram18_sites is None:
            bram18_sites = float(ZCU104_BRAM18_SITES)
        head_conv_channels = int(architecture.head_conv_channels or 0)
        head_lut = 0
        for row_key, cost in zip(self.last_selected_row_keys, costs):
            if "pw_head_7_320_1280" in row_key:
                head_lut = int(cost["lut_post_route"])
                break

        # ``total_lut / lutram_sites`` is only a legacy heuristic.  Full-network
        # route failures showed that harness adapters, FIFOs, and storage
        # inference can roughly double the LUT-as-memory demand for this
        # headconv1280 design, so keep the calibrated storage estimate separate.
        operator_lutram_heuristic = (
            float(total_lut) / lutram_sites if lutram_sites > 0 else 0.0
        )
        lut_as_memory_scale = self._optional_float(physical.get("zcu104_lut_as_memory_scale"))
        if lut_as_memory_scale is None:
            lut_as_memory_scale = 1.0
        interface_lut_as_memory = self._optional_float(
            physical.get("zcu104_interface_control_lut_as_memory")
        )
        if interface_lut_as_memory is None:
            interface_lut_as_memory = 0.0
        lut_as_memory_margin = self._optional_float(
            physical.get("zcu104_proxy_error_lut_as_memory_margin")
        )
        if lut_as_memory_margin is None:
            lut_as_memory_margin = 0.0
        predicted_lut_as_memory = (
            float(total_lut) * lut_as_memory_scale
            + interface_lut_as_memory
            + lut_as_memory_margin
        )
        predicted_lut_as_memory_pressure = (
            predicted_lut_as_memory / lutram_sites if lutram_sites > 0 else 0.0
        )

        raw_bram18 = 2.0 * float(total_bram)
        interface_bram18 = self._optional_float(physical.get("zcu104_interface_control_bram18"))
        if interface_bram18 is None:
            interface_bram18 = 0.0
        bram18_margin = self._optional_float(physical.get("zcu104_proxy_error_bram18_margin"))
        if bram18_margin is None:
            bram18_margin = 0.0
        predicted_bram18 = raw_bram18 + interface_bram18 + bram18_margin

        route_lutram_pressure = operator_lutram_heuristic
        route_bram18_pressure = raw_bram18 / bram18_sites if bram18_sites > 0 else 0.0
        predicted_bram18_pressure = predicted_bram18 / bram18_sites if bram18_sites > 0 else 0.0
        head_lut_fraction = float(head_lut) / float(total_lut) if total_lut > 0 else 0.0
        route_overflow_pressure = max(0.0, predicted_lut_as_memory_pressure - 1.0) + max(
            0.0,
            predicted_bram18_pressure - 1.0,
        )
        return {
            "zcu104_route_lutram_pressure": route_lutram_pressure,
            "zcu104_route_bram18_pressure": route_bram18_pressure,
            "zcu104_operator_lutram_heuristic_pressure": operator_lutram_heuristic,
            "zcu104_lut_as_memory_scale": lut_as_memory_scale,
            "zcu104_interface_control_lut_as_memory": interface_lut_as_memory,
            "zcu104_proxy_error_lut_as_memory_margin": lut_as_memory_margin,
            "zcu104_predicted_lut_as_memory": predicted_lut_as_memory,
            "zcu104_predicted_lut_as_memory_pressure": predicted_lut_as_memory_pressure,
            "zcu104_operator_bram18": raw_bram18,
            "zcu104_interface_control_bram18": interface_bram18,
            "zcu104_proxy_error_bram18_margin": bram18_margin,
            "zcu104_predicted_bram18": predicted_bram18,
            "zcu104_predicted_bram18_pressure": predicted_bram18_pressure,
            "zcu104_route_head_lut_fraction": head_lut_fraction,
            "zcu104_route_headconv_channels": float(head_conv_channels),
            "zcu104_route_overflow_pressure": route_overflow_pressure,
        }

    def _block_complexity(self, block: ResolvedBlockSpec) -> tuple[int, int, int]:
        if block.op == "dmdc_conv_sonar":
            channels = int(block.in_channels)
            out_channels = int(block.out_channels)
            resolution = int(block.output_resolution)
            params = 0
            macs = 0
            for dilation_rate in (1, 3, 5):
                full_res_params = (11 * channels * channels) + (channels * channels)
                pooled_params = (
                    channels * channels
                    + channels * channels * dilation_rate * dilation_rate
                )
                params += full_res_params + pooled_params
                macs += full_res_params * resolution * resolution
                macs += channels * channels * dilation_rate * dilation_rate
                macs += channels * channels * dilation_rate**4
            compression_params = channels * 3 * out_channels
            params += compression_params
            macs += compression_params * resolution * resolution
            raw_dsp = max(
                self._conv_dsp(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    depthwise=False,
                ),
                self._conv_dsp(
                    in_channels=channels * 3,
                    out_channels=out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return params, macs, raw_dsp

        if block.op == "cfem_sonar":
            in_channels = int(block.in_channels)
            out_channels = int(block.out_channels)
            resolution = int(block.output_resolution)
            fused_channels = out_channels * 3
            scale_hidden = max(1, fused_channels // 4)
            cbam_hidden = max(1, out_channels // 4)

            branch_params = 3 * (
                in_channels * out_channels
                + out_channels * out_channels * 3 * 3
            )
            scale_attention_params = (
                fused_channels * scale_hidden
                + scale_hidden
                + scale_hidden * fused_channels
                + fused_channels
            )
            projection_params = out_channels * out_channels + in_channels * out_channels
            cbam_params = (
                out_channels * cbam_hidden
                + cbam_hidden
                + cbam_hidden * out_channels
                + out_channels
                + 2 * 7 * 7
                + 1
            )
            params = (
                branch_params
                + scale_attention_params
                + projection_params
                + 2 * cbam_params
            )

            branch_macs = branch_params * resolution * resolution
            scale_attention_macs = 2 * (
                fused_channels * scale_hidden + scale_hidden * fused_channels
            )
            projection_macs = projection_params * resolution * resolution
            cbam_channel_macs = 2 * (
                out_channels * cbam_hidden + cbam_hidden * out_channels
            )
            cbam_spatial_macs = 2 * 7 * 7 * resolution * resolution
            macs = (
                branch_macs
                + scale_attention_macs
                + projection_macs
                + 2 * (cbam_channel_macs + cbam_spatial_macs)
            )
            raw_dsp = max(
                self._conv_dsp(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    depthwise=False,
                ),
                self._conv_dsp(
                    in_channels=fused_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return params, macs, raw_dsp

        return super()._block_complexity(block)

    def estimate(self, architecture: ArchitectureSpec, search_space: SearchSpace) -> CostEstimate:
        self.last_selected_row_keys = []
        analytic = super().estimate(architecture, search_space)
        rows = self._selected_rows(architecture, search_space)
        self.last_selected_row_keys = [str(row["row_key"]) for row in rows]
        costs = [self._cost(row) for row in rows]

        # Preserve report-time fields in the immutable input table; runtime
        # latency and energy use the frozen operating frequency, not HLS Fmax.
        latency_ms = sum(self._operating_latency_ms(cost) for cost in costs)
        latency_cycles = sum(int(cost["latency_cycles_hls"]) for cost in costs)
        total_lut = sum(int(cost["lut_post_route"]) for cost in costs)
        total_ff = sum(int(cost["ff_post_route"]) for cost in costs)
        total_dsp = sum(int(cost["dsp_post_route"]) for cost in costs)
        total_bram = int(math.ceil(sum(float(cost["bram_tile_post_route"]) for cost in costs)))
        peak_lut = max(int(cost["lut_post_route"]) for cost in costs)
        peak_dsp = max(int(cost["dsp_post_route"]) for cost in costs)
        peak_bram = int(math.ceil(max(float(cost["bram_tile_post_route"]) for cost in costs)))
        power_w = max(float(cost["power_w_vivado_vectorless"]) for cost in costs)
        energy_mj = sum(
            self._operating_latency_ms(cost) * float(cost["power_w_vivado_vectorless"])
            for cost in costs
        )

        resolved_blocks = search_space.resolve_blocks(architecture)
        physical_metrics = self._compute_physical_metrics(
            resolved_blocks=resolved_blocks,
            peak_buffer_bytes=analytic.peak_buffer_bytes,
        )
        route_metrics = self._zcu104_route_pressure_metrics(
            architecture=architecture,
            costs=costs,
            total_lut=total_lut,
            total_bram=total_bram,
        )
        base_physical_risk = float(physical_metrics.get("physical_risk", 0.0))
        route_overflow_pressure = float(route_metrics["zcu104_route_overflow_pressure"])
        physical_metrics["physical_risk_without_zcu104_route_gate"] = base_physical_risk
        physical_metrics["physical_risk"] = base_physical_risk + 4.0 * route_overflow_pressure
        physical_metrics.update(
            {
                "zcu104_operator_cost_rows": float(len(rows)),
                "zcu104_operator_cost_total_ff": float(total_ff),
                "full_network_route_established": 0.0,
                "full_network_power_established": 0.0,
                "vivado_power_vectorless_proxy": power_w,
                **route_metrics,
            }
        )
        violations = list(
            self._check_constraints(
                latency_ms=latency_ms,
                energy_mj=energy_mj,
                model_size_mb=analytic.model_size_mb,
                resource_dsp=total_dsp,
                resource_bram=total_bram,
                resource_lut=total_lut,
                power_w=power_w,
                memory_bandwidth_gbps=analytic.memory_bandwidth_gbps,
                offchip_mem_mb=analytic.offchip_mem_mb,
            )
        )
        violations.extend(
            self._check_physical_constraints(
                resource_dsp=total_dsp,
                resource_bram=total_bram,
                resource_lut=total_lut,
                physical_metrics=physical_metrics,
                resolved_blocks=resolved_blocks,
            )
        )
        violations.extend(self._check_operator_policy(resolved_blocks))
        for row in rows:
            if row.get("timing_met") is not True:
                violations.append(f"operator_route_timing_not_clean:{row['row_key']}")

        return CostEstimate(
            params=analytic.params,
            macs=analytic.macs,
            model_size_mb=analytic.model_size_mb,
            peak_activation_bytes=analytic.peak_activation_bytes,
            peak_weight_bytes=analytic.peak_weight_bytes,
            peak_buffer_bytes=analytic.peak_buffer_bytes,
            peak_dsp=peak_dsp,
            peak_bram=peak_bram,
            peak_lut=peak_lut,
            total_dsp=total_dsp,
            total_bram=total_bram,
            total_lut=total_lut,
            latency_cycles=latency_cycles,
            latency_ms=latency_ms,
            power_w=power_w,
            energy_mj=energy_mj,
            memory_bandwidth_gbps=analytic.memory_bandwidth_gbps,
            offchip_mem_mb=analytic.offchip_mem_mb,
            violations=tuple(dict.fromkeys(violations)),
            per_layer=(),
            physical_metrics=physical_metrics,
        )

    def reset_lut_stats(self) -> None:
        super().reset_lut_stats()
        self._operator_cost_queries = 0
        self._operator_cost_hits = 0
        self._operator_cost_misses = 0
        self.last_selected_row_keys = []

    def get_lut_stats(self) -> dict[str, Any]:
        stats = super().get_lut_stats()
        total = self._operator_cost_queries
        stats.update(
            {
                "cost_source": self.cost_source,
                "data_kind": self.data_kind,
                "search_space_version": self.search_space_version,
                "hardware_lut_version": self.hardware_lut_version,
                "search_ready": self.search_ready,
                "operator_rows": len(self.rows_by_key),
                "operator_cost_queries": self._operator_cost_queries,
                "operator_cost_hits": self._operator_cost_hits,
                "operator_cost_misses": self._operator_cost_misses,
                "operator_cost_hit_rate": self._operator_cost_hits / total if total else 0.0,
                "fallback_rate": 0.0 if self._operator_cost_misses == 0 else 1.0,
                "operator_cost_path": str(self.operator_cost_path),
                "operator_cost_file": str(self.operator_cost_path),
                "operator_cost_file_sha256": self.operator_cost_file_sha256,
                "payload_sha256": self.operator_cost_payload_sha256,
                "coverage_path": str(self.coverage_path),
                "coverage_file": str(self.coverage_path),
                "coverage_file_sha256": self.coverage_file_sha256,
                "preflight_path": None if self.preflight_path is None else str(self.preflight_path),
                "preflight_file": None if self.preflight_path is None else str(self.preflight_path),
                "preflight_file_sha256": self.preflight_file_sha256,
                "last_selected_row_count": len(self.last_selected_row_keys),
                "last_selected_row_keys": list(self.last_selected_row_keys),
                "hardware_cost_scope": (
                    "MobileNetA operator HLS/route LUT aggregation; "
                    "board classification and measured power are not established"
                ),
                "power_w_semantics": "vivado_vectorless_proxy_not_measured_board_power",
                "cost_semantics_version": "mobilenetA_fixed_clock_bram18_gap1280_v3",
                "latency_ms_semantics": "sum(hls_cycles) / frozen_clock_mhz / 1000",
                "energy_mj_semantics": "sum(vivado_vectorless_power_w * hls_cycles / frozen_clock_mhz / 1000)",
            }
        )
        return stats
