"""Commands for the independent, data-gated sonar experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import DEFAULT_CONFIG, environment, load_config, read_json, resolve_path, write_json


def main():
    parser = argparse.ArgumentParser(description="独立声呐检测试验：先核验数据，再训练，最后才使用测试数据。")
    parser.add_argument("command", choices=["prepare", "draft", "accept-exploratory", "campaign", "status", "environment"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group-map", type=Path, help="可核验的图片到真实采集序列的对应记录；不是任意随机分组。")
    parser.add_argument("--confirm-protocol", help="仅在明确接受数据协议调整后使用 PUBLIC_SPLIT_EXPLORATORY_V1。")
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve_path(config["output_dir"])
    if args.command == "draft":
        from .exploratory import prepare_exploratory_draft
        result = prepare_exploratory_draft(output / "data_audit" / "manifest.json", output / "exploratory_review",
                                           seed=config["split_seed"], val_fraction=config["validation_fraction"])
        print(result["split_status"])
        return 0
    if args.command == "accept-exploratory":
        from .protocol import accept_exploratory
        print(accept_exploratory(config, args.confirm_protocol))
        return 0
    if args.command == "prepare":
        from .data import prepare_manifest
        manifest = prepare_manifest(resolve_path(config["dataset_root"]), output / "data_audit",
                                    seed=config["split_seed"], val_fraction=config["validation_fraction"],
                                    group_map=args.group_map or config.get("group_map"))
        print(manifest["split_status"])
        return 0 if manifest["split_status"] == "PASS" else 2
    if args.command == "campaign":
        from .campaign import execute_campaign
        return execute_campaign(config)
    if args.command == "environment":
        result = environment()
        write_json(output / "environment_lock.json", result)
        print(result)
        return 0
    for name in ["campaign_status.json", "data_audit/manifest.json"]:
        path = output / name
        if path.exists():
            value = read_json(path)
            print({"file": name, "status": value.get("status", value.get("split_status")),
                   "reason": value.get("reason"), "formal_training_started": value.get("formal_training_started", False)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
