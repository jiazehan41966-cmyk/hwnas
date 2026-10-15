"""Version 2: keep the ARM control plane outside CPU3 and silent while measuring."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from tools.hardware_measurement_pilot import d09_cpu_campaign as campaign
from tools.hardware_measurement_pilot import d09_cpu_quiet_transport as transfer


def configure():
    campaign.RESULT = Path("results/d09_cpu_freeze_20261013/quiet_v2").resolve()
    campaign.transfer = transfer
    campaign.time = SimpleNamespace(monotonic=time.monotonic, sleep=lambda seconds: time.sleep(180))
    original = transfer.prepare

    def prepare(*args, **kwargs):
        kwargs["bridge"] = str(Path(__file__).with_name("board_bridge_stdin.py").resolve())
        kwargs["chunk_bytes"] = 4 * 1024 * 1024
        return original(*args, **kwargs)

    transfer.prepare = prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--case", choices=tuple(campaign.CASES), required=True)
    p.add_argument("--phase", choices=("calibration", "validation"), required=True)
    p.add_argument("--run-id", required=True)
    p = sub.add_parser("run")
    p.add_argument("--case-dir", required=True)
    args = parser.parse_args()
    configure()
    result = (campaign.prepare(args.case, args.phase, args.run_id) if args.action == "prepare"
              else campaign.run(args.case_dir, timeout_seconds=600))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
