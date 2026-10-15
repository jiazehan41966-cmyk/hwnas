"""Copy Phase5A evaluation code to a new remote workdir. Does not modify frozen Phase2F artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/phase5a_quantization/phase5a_src_20260920.tar.gz"
HOST = "jiajia@10.70.23.121"
REMOTE = "/home/jiajia/experiments/sonar_phase5a_20260920_v1"
FROZEN = "/home/jiajia/experiments/sonar_phase2c_20260914_v2"

ENV = os.environ.copy()
ENV.update(
    SSH_ASKPASS=r"C:/Users/Lenovo/.codex/local-tools/zcu104-ssh/ssh_askpass.cmd",
    SSH_ASKPASS_REQUIRE="force",
    DISPLAY="codex:0",
)
SSH = [
    "C:/Windows/System32/OpenSSH/ssh.exe",
    "-p",
    "7889",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "ConnectTimeout=20",
    HOST,
]
SCP = [
    "C:/Windows/System32/OpenSSH/scp.exe",
    "-P",
    "7889",
    "-o",
    "StrictHostKeyChecking=yes",
]

SKIP_PARTS = {"__pycache__", ".venv", ".git"}
SKIP_TOP = {"data", "results", "artifacts", "logs", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".onnx"}
INCLUDE_DIRS = ("src", "scripts", "configs", "tools")
INCLUDE_FILES = (
    "reports/phase5a_quantization/CHECKPOINT_SELECTION_MANIFEST.json",
    "reports/phase5a_quantization/CHECKPOINT_SELECTION_AUDIT.md",
    "reports/phase2f_multiseed/network_candidate_set.json",
)
INCLUDE_GLOBS = (
    "reports/fpga_h2_contract/*.json",
    "reports/fpga_h2_contract/*.md",
)
REMOTE_EVAL = """#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jiajia/experiments/sonar_phase5a_20260920_v1
PYTHON=/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u tools/phase5a_quantization/run.py --skip-export
""".replace("\r\n", "\n")


def keep(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.parts and rel.parts[0] in SKIP_TOP:
        return False
    if any(part in SKIP_PARTS for part in rel.parts):
        return False
    return path.suffix.lower() not in SKIP_SUFFIX


def build_bundle() -> None:
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "w:gz") as tf:
        for name in INCLUDE_DIRS:
            base = ROOT / name
            for path in base.rglob("*"):
                if path.is_file() and keep(path):
                    tf.add(path, arcname=path.relative_to(ROOT).as_posix())
        for rel in INCLUDE_FILES:
            path = ROOT / rel
            tf.add(path, arcname=Path(rel).as_posix())
        for pattern in INCLUDE_GLOBS:
            for path in ROOT.glob(pattern):
                if path.is_file():
                    tf.add(path, arcname=path.relative_to(ROOT).as_posix())
    print("bundle", BUNDLE, BUNDLE.stat().st_size)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=ENV, check=True, **kwargs)


def main() -> None:
    build_bundle()
    run(SSH + [f"mkdir -p {REMOTE}"])
    run(SCP + [str(BUNDLE), f"{HOST}:{REMOTE}/phase5a_src_20260920.tar.gz"])
    setup = f"""
set -euo pipefail
cd '{REMOTE}'
tar -xzf phase5a_src_20260920.tar.gz
mkdir -p results artifacts logs reports
ln -sfn '{FROZEN}/data' data
ln -sfn '{FROZEN}/results/phase2e_independent' results/phase2e_independent
ln -sfn '{FROZEN}/results/phase2f_multiseed' results/phase2f_multiseed
cat > tools/phase5a_quantization/remote_eval.sh << 'EOF'
{REMOTE_EVAL}EOF
chmod 755 tools/phase5a_quantization/remote_eval.sh
python3 - << 'PY'
from pathlib import Path
checks = [
    'reports/phase5a_quantization/CHECKPOINT_SELECTION_MANIFEST.json',
    'reports/phase2f_multiseed/network_candidate_set.json',
    'reports/fpga_h2_contract/numeric_contract_v1.json',
    'tools/phase5a_quantization/run.py',
    'src/hwnas_fpga/deploy/h2_ptq.py',
    'data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8/train',
    'data/external_redownload_audit_20260909/roboflow_cylider2/extracted/v6_yolov8/valid',
    'results/phase2e_independent/P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5/best.pt',
    'results/phase2f_multiseed/P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5/seed_43/best.pt',
]
for item in checks:
    path = Path(item)
    print(('OK' if path.exists() else 'MISSING'), item)
PY
"""
    setup = setup.replace("\r\n", "\n").replace("\r", "\n")
    proc = subprocess.run(
        SSH + ["bash", "-s"],
        input=setup.encode("utf-8"),
        env=ENV,
        capture_output=True,
    )
    print("SETUP_RC", proc.returncode)
    print(proc.stdout.decode("utf-8", "replace"))
    print(proc.stderr.decode("utf-8", "replace"))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
