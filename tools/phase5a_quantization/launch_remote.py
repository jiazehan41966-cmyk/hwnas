"""Launch the remote Phase5A PTQ tmux job. Does not train or touch test."""

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
LAUNCH = """set -euo pipefail
REMOTE=/home/jiajia/experiments/sonar_phase5a_20260920_v1
cd "$REMOTE"
mkdir -p logs
if tmux has-session -t phase5a_ptq 2>/dev/null; then
  tmux kill-session -t phase5a_ptq
fi
tmux new-session -d -s phase5a_ptq bash -lc 'cd /home/jiajia/experiments/sonar_phase5a_20260920_v1 && bash tools/phase5a_quantization/remote_eval.sh > logs/phase5a_ptq.log 2>&1'
sleep 3
tmux ls
echo ---log---
sed -n '1,80p' logs/phase5a_ptq.log || true
echo ---ps---
ps -u jiajia -o pid,pcpu,pmem,etime,cmd | grep -E 'phase5a|run.py' | grep -v grep || true
""".replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    proc = subprocess.run(
        SSH + ["bash", "-s"],
        input=LAUNCH.encode("utf-8"),
        env=ENV,
        capture_output=True,
    )
    sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
    sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
