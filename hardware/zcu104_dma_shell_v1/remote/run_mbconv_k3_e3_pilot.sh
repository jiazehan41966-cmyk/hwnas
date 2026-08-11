#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
result_root="${repo_root}/results/zcu104_dma_shell_mbconv_k3_e3_pilot_v1"
hls_bin="/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls"
vivado_bin="/opt/Xilinx/Vivado/2022.2/bin/vivado"
hls_tcl="${repo_root}/hardware/zcu104_dma_shell_v1/hls/mbconv_k3_e3/run_hls.tcl"
vivado_tcl="${repo_root}/hardware/zcu104_dma_shell_v1/vivado/build_mbconv_k3_e3_pilot_bd.tcl"

mkdir -p "${result_root}/logs" "${result_root}/environment"

{
  date --iso-8601=seconds
  git -C "${repo_root}" rev-parse HEAD
  git -C "${repo_root}" status --porcelain
  "${hls_bin}" -version | head -n 4
  "${vivado_bin}" -version | head -n 4
  df -h "${repo_root}" | tail -n 1
} > "${result_root}/environment/run_context.txt"

"${hls_bin}" -f "${hls_tcl}" 2>&1 | tee "${result_root}/logs/vitis_hls.log"

component_xml="${result_root}/hls/project/solution1/impl/ip/component.xml"
python3 - "${component_xml}" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

component = Path(sys.argv[1])
if not component.is_file():
    raise SystemExit(f"missing component.xml: {component}")
root = ET.parse(component).getroot()
ns = {"spirit": "http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009"}
def text(name: str) -> str:
    node = root.find(f"spirit:{name}", ns)
    return "" if node is None else (node.text or "").strip()
identity = {key: text(key) for key in ("vendor", "library", "name", "version")}
expected = {
    "vendor": "hwnas.org",
    "library": "hls",
    "name": "zcu104_stage4_k3_e3_kernel",
    "version": "1.0",
}
if identity != expected:
    raise SystemExit(f"VLNV mismatch: observed={identity}, expected={expected}")
out = component.parents[5] / "component_identity.json"
out.write_text(json.dumps({"status": "PASS", "vlnv": ":".join(identity.values())}, indent=2) + "\n")
print(f"COMPONENT_XML={component}")
print(f"VLNV={':'.join(identity.values())}")
PY

"${vivado_bin}" -mode batch -nolog -nojournal -notrace -source "${vivado_tcl}" \
  2>&1 | tee "${result_root}/logs/vivado_build.log"

find "${result_root}" -type f \( \
  -name '*.json' -o -name '*.rpt' -o -name '*.xml' -o -name '*.bit' -o \
  -name '*.xsa' -o -name '*.dcp' -o -name '*.tcl' -o -name '*.log' \
\) -print0 | sort -z | xargs -0 sha256sum > "${result_root}/sha256sums.txt"

printf 'PASS: Stage4 MBConv-K3-E3 ZCU104 DMA shell pilot build\n'
