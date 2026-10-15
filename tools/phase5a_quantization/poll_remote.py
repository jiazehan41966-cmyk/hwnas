"""Poll remote Phase5A PTQ progress. Read-only."""

from __future__ import annotations

import os
import subprocess
import sys

HOST = "jiajia@10.70.23.121"
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
POLL = """set -euo pipefail
REMOTE=/home/jiajia/experiments/sonar_phase5a_20260920_v1
echo ---tmux---
tmux ls || true
echo ---gpu---
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo ---ps---
ps -u jiajia -o pid,pcpu,pmem,etime,cmd | grep -E 'run.py --skip-export' | grep -v grep || true
echo ---log_tail---
tail -n 40 "$REMOTE/logs/phase5a_ptq.log" || true
echo ---partial---
python3 - <<'PY'
from pathlib import Path
import json
p=Path('/home/jiajia/experiments/sonar_phase5a_20260920_v1/reports/phase5a_quantization/ptq_results_partial.json')
final=Path('/home/jiajia/experiments/sonar_phase5a_20260920_v1/reports/phase5a_quantization/PHASE5A_FINAL_REVIEW.md')
print('partial_exists', p.exists(), 'final_exists', final.exists())
if p.exists():
    rows=json.loads(p.read_text())
    print('partial_rows', len(rows))
    for row in rows:
        print(row.get('network'), row.get('seed'), row.get('status'), row.get('fp32_mAP50_95'), row.get('int8_mAP50_95'), row.get('delta_mAP50_95'))
PY
""".replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    proc = subprocess.run(SSH + ["bash", "-s"], input=POLL.encode("utf-8"), env=ENV, capture_output=True)
    sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
    sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
