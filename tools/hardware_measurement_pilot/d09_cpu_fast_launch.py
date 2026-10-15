"""Select the previously verified stdin bridge without changing the CPU protocol."""
import argparse
import json
from pathlib import Path

from tools.hardware_measurement_pilot import d09_cpu_campaign as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    args = parser.parse_args()
    original = campaign.transfer.prepare
    bridge = str(Path(__file__).with_name("board_bridge_stdin.py").resolve())

    def prepare(*positional, **keywords):
        keywords["bridge"] = bridge
        keywords["chunk_bytes"] = 4 * 1024 * 1024
        return original(*positional, **keywords)

    campaign.transfer.prepare = prepare
    try:
        result = campaign.run(args.case_dir)
    finally:
        campaign.transfer.prepare = original
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
