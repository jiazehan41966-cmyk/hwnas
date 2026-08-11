#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the seven synthesized ZCU104 operator AXI interfaces from retained HLS reports."
    )
    parser.add_argument("--hls-evidence-tar", type=Path, required=True)
    parser.add_argument(
        "--hls-summary",
        type=Path,
        default=Path("artifacts/zcu104_full_campaign_v1/control/remote_hls_execution_complete_20260810.json"),
    )
    parser.add_argument(
        "--route-summary",
        type=Path,
        default=Path("artifacts/zcu104_full_campaign_v1/control/remote_ooc_route_execution_complete_20260810.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/zcu104_dma_shell_v1/operator_interface_audit.json"),
    )
    return parser.parse_args()


def extract_port_bits(report: str, port: str) -> int:
    match = re.search(rf"\|{re.escape(port)}\s*\|\s*(?:in|out)\|\s*(\d+)\|", report)
    if not match:
        raise ValueError(f"Missing interface row for {port}")
    return int(match.group(1))


def extract_port_protocol(report: str, port: str) -> str:
    match = re.search(
        rf"\|{re.escape(port)}\s*\|\s*(?:in|out)\|\s*\d+\|\s*([^|]+)\|",
        report,
    )
    if not match:
        raise ValueError(f"Missing interface row for {port}")
    return match.group(1).strip()


def main() -> None:
    args = parse_args()
    hls_summary = json.loads(args.hls_summary.read_text(encoding="utf-8"))
    route_summary = json.loads(args.route_summary.read_text(encoding="utf-8"))
    hls_rows = {row["cost_key"]: row for row in hls_summary["rows"]}
    route_rows = {row["cost_key"]: row for row in route_summary["rows"]}

    rows: list[dict[str, Any]] = []
    with tarfile.open(args.hls_evidence_tar, "r:gz") as archive:
        metadata_members = sorted(name for name in archive.getnames() if name.endswith("/metadata.json"))
        if len(metadata_members) != 7:
            raise ValueError(f"Expected seven metadata records, found {len(metadata_members)}")

        for metadata_member in metadata_members:
            metadata_file = archive.extractfile(metadata_member)
            if metadata_file is None:
                raise ValueError(f"Cannot extract {metadata_member}")
            metadata = json.loads(metadata_file.read().decode("utf-8"))
            case_root = metadata_member.rsplit("/", 1)[0]
            report_member = (
                f"{case_root}/project/solution1/syn/report/"
                f"{metadata['top_function']}_csynth.rpt"
            )
            report_file = archive.extractfile(report_member)
            if report_file is None:
                raise ValueError(f"Cannot extract {report_member}")
            report = report_file.read().decode("utf-8", errors="replace")

            params = metadata["parameters"]
            stage = "stage2" if metadata["operator"].startswith("mbconv_stage2") else "stage4"
            input_bits = extract_port_bits(report, "input_stream_TDATA")
            output_bits = extract_port_bits(report, "output_stream_TDATA")
            input_words = int(params["feature_h"]) * int(params["feature_w"])
            output_words = int(params["out_h"]) * int(params["out_w"])
            is_skip = metadata["operator"] == "skip_stage4"
            if is_skip:
                weight_elements = 0
                bias_elements = 0
            else:
                expanded_channels = int(params["expanded_channels"])
                input_channels = int(params["in_channels"])
                output_channels = int(params["out_channels"])
                kernel_size = int(params["kernel_size"])
                weight_elements = (
                    expanded_channels * input_channels
                    + expanded_channels * kernel_size * kernel_size
                    + output_channels * expanded_channels
                )
                bias_elements = expanded_channels + expanded_channels + output_channels
            weight_protocol = extract_port_protocol(report, "weights_Addr_A") if "weights_Addr_A" in report else extract_port_protocol(report, "weights")
            bias_protocol = extract_port_protocol(report, "biases_Addr_A") if "biases_Addr_A" in report else extract_port_protocol(report, "biases")
            cost_key = metadata["op_spec"]["cost_key"]

            row = {
                "cost_key": cost_key,
                "case_name": metadata["case_name"],
                "top_function": metadata["top_function"],
                "stage": stage,
                "shape_chw": {
                    "input": [int(params["in_channels"]), int(params["feature_h"]), int(params["feature_w"])],
                    "output": [int(params["out_channels"]), int(params["out_h"]), int(params["out_w"])],
                },
                "tensor_flatten_order": "pixel_major_HWC",
                "activation_format": "signed_int8_twos_complement",
                "input_axis": {
                    "tdata_bits": input_bits,
                    "tkeep_bits": extract_port_bits(report, "input_stream_TKEEP"),
                    "tstrb_bits": extract_port_bits(report, "input_stream_TSTRB"),
                    "tlast_bits": extract_port_bits(report, "input_stream_TLAST"),
                    "words_per_frame": input_words,
                    "bytes_per_frame": input_words * input_bits // 8,
                },
                "output_axis": {
                    "tdata_bits": output_bits,
                    "tkeep_bits": extract_port_bits(report, "output_stream_TKEEP"),
                    "tstrb_bits": extract_port_bits(report, "output_stream_TSTRB"),
                    "tlast_bits": extract_port_bits(report, "output_stream_TLAST"),
                    "words_per_frame": output_words,
                    "bytes_per_frame": output_words * output_bits // 8,
                },
                "control": {
                    "protocol": "AXI4-Lite_ap_ctrl_hs",
                    "address_bits": extract_port_bits(report, "s_axi_control_AWADDR"),
                },
                "parameters": {
                    "weights": {
                        "protocol": weight_protocol,
                        "element_bits": 8,
                        "elements": weight_elements,
                        "bytes": weight_elements,
                    },
                    "biases": {
                        "protocol": bias_protocol,
                        "element_bits": 32,
                        "elements": bias_elements,
                        "bytes": bias_elements * 4,
                    },
                    "note": "Skip operands are unused and synthesize as ap_none scalar pins; MBConv operands expose BRAM master ports.",
                },
                "hls": {
                    "status": hls_rows[cost_key].get("status", "completed"),
                    "latency_cycles": hls_rows[cost_key]["latency_cycles"],
                    "estimated_clock_period_ns": hls_rows[cost_key]["estimated_clock_period_ns"],
                },
                "operator_ooc_route": {
                    "wns_ns": route_rows[cost_key]["wns_ns"],
                    "tns_ns": route_rows[cost_key]["tns_ns"],
                    "lut": route_rows[cost_key]["lut"],
                    "dsp": route_rows[cost_key]["dsp"],
                    "bram36": route_rows[cost_key]["bram36"],
                },
            }
            rows.append(row)

    expected_widths = {
        "stage2": (128, 192),
        "stage4": (256, 256),
    }
    for row in rows:
        expected_input, expected_output = expected_widths[row["stage"]]
        if row["input_axis"]["tdata_bits"] != expected_input:
            raise ValueError(f"Unexpected input width for {row['cost_key']}")
        if row["output_axis"]["tdata_bits"] != expected_output:
            raise ValueError(f"Unexpected output width for {row['cost_key']}")
        for direction in ("input_axis", "output_axis"):
            axis = row[direction]
            if axis["tkeep_bits"] != axis["tdata_bits"] // 8:
                raise ValueError(f"TKEEP width mismatch for {row['cost_key']} {direction}")
            if axis["tstrb_bits"] != axis["tdata_bits"] // 8:
                raise ValueError(f"TSTRB width mismatch for {row['cost_key']} {direction}")
            if axis["tlast_bits"] != 1:
                raise ValueError(f"TLAST width mismatch for {row['cost_key']} {direction}")

    payload = {
        "schema_version": 1,
        "status": "PASS_7_OF_7_EXISTING_OPERATOR_INTERFACES_AUDITED",
        "source": {
            "hls_evidence_tar": str(args.hls_evidence_tar.resolve()),
            "hls_evidence_tar_sha256": sha256_file(args.hls_evidence_tar),
            "hls_summary": str(args.hls_summary.as_posix()),
            "route_summary": str(args.route_summary.as_posix()),
        },
        "common_contract": {
            "stream_order": "pixel_major_HWC",
            "channel_packing": "channel c occupies TDATA[8*c+7:8*c]",
            "activation_format": "signed_int8_twos_complement",
            "input_tlast": "one TLAST on the final pixel word; current cores do not consume input TLAST semantically",
            "output_tlast": "one TLAST on the final output pixel word",
            "full_word_tkeep_tstrb": True,
        },
        "rows": sorted(rows, key=lambda item: item["cost_key"]),
        "blocked_not_audited": [
            "stage4__dmdc_conv_sonar__32x28x28_to_32x28x28__s1",
            "stage4__cfem_sonar__32x28x28_to_32x28x28__s1",
        ],
        "claim_boundary": "The seven rows have retained HLS and operator-only OOC route evidence. This audit is not a DMA-shell implementation, bitstream, board run, or power result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
