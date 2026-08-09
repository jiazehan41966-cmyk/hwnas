"""Board profile templates for FPGA deployment targets."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from hwnas_fpga.interfaces import HardwareSpec


BOARD_PROFILES: dict[str, HardwareSpec] = {
    "zynq7020": HardwareSpec(
        name="Zynq-7020 XC7Z020",
        clock_mhz=200,
        max_lut=53_200,
        max_ff=106_400,
        max_bram=140,
        max_bram18=280,
        max_bram36=140,
        max_uram=0,
        max_dsp=220,
        max_power_w=5.0,
        memory_bandwidth_gbps=4.2,
        offchip_mem_mb=512.0,
    ),
    "kintex7_xc7k325": HardwareSpec(
        name="Kintex-7 XC7K325T",
        clock_mhz=250,
        max_lut=203_800,
        max_ff=407_600,
        max_bram=445,
        max_bram18=890,
        max_bram36=445,
        max_uram=0,
        max_dsp=840,
        max_power_w=12.0,
        memory_bandwidth_gbps=8.0,
        offchip_mem_mb=1024.0,
    ),
    "alinx_av7k325": HardwareSpec(
        name="ALINX AV7K325 (XC7K325T-2FFG900I)",
        clock_mhz=200,
        max_lut=203_800,
        max_ff=407_600,
        max_bram=445,
        max_bram18=890,
        max_bram36=445,
        max_uram=0,
        max_dsp=840,
        max_power_w=12.0,
        memory_bandwidth_gbps=12.8,
        offchip_mem_mb=2048.0,
    ),
    "zcu104_xczu7ev": HardwareSpec(
        name="AMD/Xilinx ZCU104 (XCZU7EV-2FFVC1156)",
        clock_mhz=200,
        max_lut=230_400,
        max_ff=460_800,
        # The legacy estimator consumes BRAM36 blocks through ``max_bram``.
        max_bram=312,
        max_bram18=624,
        max_bram36=312,
        max_uram=96,
        max_dsp=1_728,
        # Do not encode an assumed board-power ceiling as measured evidence.
        max_power_w=None,
        memory_bandwidth_gbps=19.2,
        offchip_mem_mb=4096.0,
    ),
}

BOARD_ALIASES = {
    "xc7z020": "zynq7020",
    "zynq-7020": "zynq7020",
    "zynq_7020": "zynq7020",
    "zynq7020": "zynq7020",
    "xc7k325": "kintex7_xc7k325",
    "xc7k325t": "kintex7_xc7k325",
    "kintex7": "kintex7_xc7k325",
    "kintex-7": "kintex7_xc7k325",
    "kintex7_xc7k325": "kintex7_xc7k325",
    "kintex7_xc7k325t": "kintex7_xc7k325",
    "av7k325": "alinx_av7k325",
    "av7k3252": "alinx_av7k325",
    "alinxav7k325": "alinx_av7k325",
    "alinx_av7k325": "alinx_av7k325",
    "alinx-av7k325": "alinx_av7k325",
    "zcu104": "zcu104_xczu7ev",
    "zcu-104": "zcu104_xczu7ev",
    "xczu7ev": "zcu104_xczu7ev",
    "xczu7evffvc1156": "zcu104_xczu7ev",
    "xczu7ev-ffvc1156-2-e": "zcu104_xczu7ev",
    "zcu104_xczu7ev": "zcu104_xczu7ev",
}


def _normalize_board_name(board_name: str) -> str:
    normalized = board_name.strip().lower().replace(" ", "").replace("/", "_")
    return normalized


def get_board_profile(board_name: str) -> HardwareSpec:
    key = BOARD_ALIASES.get(_normalize_board_name(board_name), _normalize_board_name(board_name))
    if key not in BOARD_PROFILES:
        raise KeyError(f"Unknown board profile: {board_name}")
    return BOARD_PROFILES[key]


def resolve_board_profile(
    board_name: str | None,
    overrides: Mapping[str, Any] | None = None,
) -> HardwareSpec:
    base = get_board_profile(board_name) if board_name else HardwareSpec(name="fpga", clock_mhz=200)
    data = asdict(base)
    for key, value in dict(overrides or {}).items():
        if value is not None and key in data:
            data[key] = value
    return HardwareSpec(**data)


def list_board_profiles() -> dict[str, dict[str, Any]]:
    return {key: asdict(value) for key, value in BOARD_PROFILES.items()}
